"""实体链路族：实体抽取编排 + 实体表写入。

从 ``brain.core`` 整搬两件：``extract_entities_with_origin``（多路抽取
→ 清洗去重按可信度合并 → 别名归一）与 ``_link_entities``（实体表写入
与倒排维护）。两件的外部能力全部由壳**调用时现取**注入，本模块零
import：

  * extract：``preprocess`` / ``cjk_ratio`` / ``get_nlp``（惰性——不调
    就不加载 spaCy）/ ``title_entities`` / ``heuristic_entities`` /
    ``clean_name`` / ``canonicalize_map`` 是 brain 侧能力回调；
    ``lexicon`` 是词库序列（热重载会整体换新，壳每次现取）；四档
    origin 常量与 ``origin_rank`` / ``cjk_skip_ratio`` 按值传。

逐字保留的语义（动一个就是行为变化，parity 会抓）：

  * extract 四路次序固定：词库子串（大小写不敏感）→ spaCy（**仅英文
    比例高时**，异常吞掉不炸）→ 《》书名号 → 启发式兜底；
  * 去重键是 ``clean 后的 lower().strip()``；**单字符候选丢弃**
    （``len(k) <= 1``）；同键后来者只在**可信度严格更高**时覆盖 origin
    （rank 相同不覆盖——先到先得保首现显示名）；
  * 空 prepared（``strip()`` 后为空）直接返回空 dict，四路都不跑；
  * link：名字先 ``dict.fromkeys`` 去重；origin 缺省视为 heuristic
    （不可信）；升级**只升不降**（严格大于才写）；``mention_count``
    是派生量恒等于 ``len(memory_ids)``（不累加）；``memory_id`` 假值
    不入倒排；``last_seen`` 取 max、倒排为空时 ``first_seen`` 重置。
"""
from __future__ import annotations

from typing import Any, Callable


def extract_entities_with_origin(
    text: str,
    *,
    preprocess: Callable[[str], str],
    cjk_ratio: Callable[[str], float],
    cjk_skip_ratio: float,
    lexicon: Any,
    get_nlp: Callable[[], Any],
    title_entities: Callable[[str], list],
    heuristic_entities: Callable[..., list],
    clean_name: Callable[[str], str],
    canonicalize_map: Callable[[dict], dict],
    origin_rank: dict,
    origin_lexicon: str,
    origin_spacy: str,
    origin_title: str,
    origin_heuristic: str,
) -> dict[str, str]:
    """同 ``extract_entities``，但返回 ``{实体名: origin}``。

    同一实体被多条路径命中时取**可信度最高**的来源（manual > lexicon >
    spacy > heuristic）。返回顺序按可信度降序，与旧 ``extract_entities``
    的排序语义一致。无产出的路径不留下占位键。
    """
    prepared = preprocess(text)
    if not prepared.strip():
        return {}

    is_cjk = cjk_ratio(prepared) > cjk_skip_ratio
    candidates: list[tuple[str, str]] = []

    # 1) 专名词库：子串命中（不区分大小写）
    if lexicon:
        lowered = prepared.lower()
        for term in lexicon:
            if term.lower() in lowered:
                candidates.append((term, origin_lexicon))

    # 2) spaCy：只对英文比例高的文本启用
    if not is_cjk:
        nlp = get_nlp()
        if nlp is not None:
            try:
                doc = nlp(prepared)
                for ent in doc.ents:
                    if ent.label_ in (
                        "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
                        "WORK_OF_ART", "NORP", "FAC",
                    ):
                        candidates.append((ent.text.strip(), origin_spacy))
            except Exception:
                pass

    # 3) 《》书名号：中文明确的"作品名"标记，单列 title 档（受信任）
    for n in title_entities(prepared):
        candidates.append((n, origin_title))

    # 4) 启发式兜底（spaCy 不可用 / 无产出 / 中文文本都走这里）
    for n in heuristic_entities(prepared, cjk=is_cjk):
        candidates.append((n, origin_heuristic))

    out: dict[str, str] = {}        # 显示名（保留原始大小写）-> origin
    by_key: dict[str, str] = {}     # lower -> 显示名（按小写去重）
    for n, origin in candidates:
        cleaned = clean_name(n)
        if not cleaned:
            continue
        k = cleaned.lower().strip()
        if not k or len(k) <= 1:
            continue
        disp = by_key.get(k)
        if disp is None:
            by_key[k] = cleaned
            out[cleaned] = origin
        elif origin_rank[origin] > origin_rank[out[disp]]:
            out[disp] = origin
    # 5) 别名归一：词库写了 ``子安 => 顾言``，这里就把"子安"折成"顾言"。
    #    只归一一边等于没归一——抽取侧不折，库就把一个实体的多个称法拆成
    #    互不连通的档案；查询侧不折，查"子安"照样吃不到"顾言"的实体加成。
    #    两侧都经本方法（recall/block_4/ingest 全都调它），所以天然一致。
    return canonicalize_map(out)


def link_entities(
    names: list[str],
    memory_id: str,
    ts: float,
    origins: dict[str, str] | None = None,
    *,
    entities: dict,
    nid: Callable[[str], str],
    origin_rank: dict,
    origin_heuristic: str,
) -> list[str]:
    """把实体写入实体表并维护倒排索引，返回实体 id 列表。

    ``mention_count`` 的语义是"被多少条记忆提到过"，恒等于
    ``len(memory_ids)``——**不累加而派生**。此前的累加实现会让计数虚高
    （一条记忆即可把计数推到 3 以上），使画像入选门槛形同虚设，
    且在删除记忆后无法自动纠正。名字先去重，避免同一实体重复计数。

    ``origins`` 缺省视为 ``heuristic``（不可信），避免调用方忘记标注就
    白拿信任；同一实体被多次以不同来源命中时**只升级不降级**。
    ``entities`` 是 brain 的实体表本体（就地修改，容器不搬）。
    """
    origins = origins or {}
    ids = []
    for name in dict.fromkeys(names):
        key = name.lower()
        new_origin = origins.get(key, origin_heuristic)
        ent = entities.get(key)
        if ent is None:
            ent = {
                "id": nid("ent"),
                "name": name,
                "key": key,
                "mention_count": 0,
                "memory_ids": [],
                "first_seen": ts,
                "last_seen": ts,
                "origin": new_origin,
            }
            entities[key] = ent
        elif origin_rank.get(new_origin, 0) > origin_rank.get(
            ent.get("origin", origin_heuristic), 0
        ):
            # 升级：候选来源被更可信的来源确认
            ent["origin"] = new_origin
        if memory_id and memory_id not in ent["memory_ids"]:
            ent["memory_ids"].append(memory_id)
        # 派生量：与 memory_ids 保持恒等，删除记忆后不会失真
        ent["mention_count"] = len(ent["memory_ids"])
        ent["last_seen"] = max(ent.get("last_seen", ts), ts)
        if not ent["memory_ids"]:
            ent["first_seen"] = ts
        ids.append(ent["id"])
    return ids
