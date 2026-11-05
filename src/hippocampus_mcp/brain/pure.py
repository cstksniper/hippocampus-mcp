# -*- coding: utf-8 -*-
"""B2-3 P1：HippoBrain 的纯方法（不碰实例状态）

本簇从 ``brain.py`` 逐字搬来。

判定口径是 AST 实证，不是感觉：**既不读写 ``self.*``、也不调用任何
``self`` 方法** —— 它们与 HippoBrain 的实例状态毫无关系，抽出去
必然等价。

用 Mixin 而不是「传 manager 进去」：``self.manager`` 有 62 个读点，
任何改签名的做法都会触发大规模调用点改动，那正是大重构失控的
典型入口。Mixin 里仍然写 ``self``，靠 MRO 解析，调用点一行不动。

⚠️ ``_is_restatement`` **故意留**在 brain.py：``scripts/test_batch2.py``
用 ``src.split("def _is_restatement")[1]`` 定位它做结构守卫，搬走会
IndexError。那是「判据单一真相源」的守卫，比 32 行代码位置更值钱。
"""

from __future__ import annotations

from ..common.constants import *  # noqa: F401,F403  （常量表：见 common/constants.__all__）
from ..common.constants import logger
from ..config import load_config
import hashlib
import json
import os
import re
from typing import Any
import numpy as np
from ..domain import policies as _policies
from ..domain import topic as topic_mod
from ..store.memory import CognitiveType
from ..store import source_sig as _source_sig

#: 用户级继承槽默认条数：param==0 且 env 空时回落这里。
#: 刻意住消费方（resolve_user_scope_limit）而不是 common/constants——
#: 架构守卫钉死 constants 行数天花板，新常量按守卫指引进 domain/config/
#: 消费模块。与 ``Config.user_scope_limit`` 的字面默认同值，对齐由
#: tests/test_u1_user_scope.py C6 钉住（test_config_parity P7 同口径）。
USER_SCOPE_LIMIT_DEFAULT = 3

#: 用户级任务目录默认条数：param==0 且 env 空时回落这里。
#: 目录是**索引**不是正文（设计 §3.4），8 条足够覆盖「手头这条 + 最近几条」；
#: 超出的老任务由 ``task_directory.total`` 自曝后被截断，宿主可按需再查。
#: 与 ``Config.task_directory_limit`` 的字面默认同值，对齐由
#: tests/test_u5_task_directory.py C6 钉住（test_config_parity P7 同口径）。
TASK_DIRECTORY_LIMIT_DEFAULT = 8


class _PureMixin:
    """B2-3 P1：HippoBrain 的纯方法（不碰实例状态）"""

    # ==================================================================
    # 项目专名词库（覆盖报告·高 6）
    # ==================================================================
    @staticmethod
    def _resolve_lexicon_path(explicit: str | None, persist_dir: str | None) -> str | None:
        """解析词库文件路径：显式参数 > env > ``<记忆库>/entity_lexicon.txt``。

        默认路径**每次现算而不缓存解析结果**——若在服务启动后才把
        entity_lexicon.txt 放进记忆库（很常见的操作顺序），缓存的 "None"
        会让 ``reload_lexicon()`` 永远读到空词库，且毫无提示。
        """
        if explicit:
            return explicit
        env_path = load_config().entity_lexicon or None
        if env_path:
            return env_path
        if persist_dir:
            default_lex = os.path.join(persist_dir, "entity_lexicon.txt")
            if os.path.isfile(default_lex):
                return default_lex
        return None




    @staticmethod
    def _lexicon_mtime_of(path: str | None) -> float | None:
        """词库文件的 mtime；文件不存在 / 读不到都返回 None（**不当错误抛**）。"""
        try:
            return os.path.getmtime(path) if path and os.path.isfile(path) else None
        except OSError:
            return None




    @classmethod
    def _parse_lexicon_text(cls, text: str) -> tuple[set[str], dict[str, str], list[str]]:
        """解析词库文本 → ``(词条集合, 别名映射, 问题行)``。

        行语法（UTF-8，``#`` 开头为注释，空行忽略）::

            顾言                     # 普通词条
            子安 => 顾言              # 别名行：左=别名，右=规范名
            陆暄 -> 暄儿             # 箭头写法等价（-> / → / ⇒）

        **别名与规范名都进词条集合**（两者都要参与子串命中，否则"别名根本没被
        认出来"就谈不上归一）；别名映射另存一份 ``{别名lower: 规范名原样}``，
        供抽取侧与查询侧归一。

        为什么要别名：**同一实体的多种称法在任何知识库都存在**（表字/简称/
        全称/旧称/代号）。小说人物只是最显眼的一例——查"子安"吃不到"顾言"
        的实体加成、档案还各存一份互不连通，是通用缺陷，不是某个项目的私事。
        """
        terms: set[str] = set()
        aliases: dict[str, str] = {}
        problems: list[str] = []
        for lineno, raw in enumerate((text or "").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = cls._ALIAS_SEP.search(line)
            if not m:
                terms.add(line.lower())
                continue
            alias = line[:m.start()].strip().lstrip("#").strip()
            canon = line[m.end():].strip()
            if not alias or not canon:
                # 不静默丢弃：写坏了必须让人知道（否则表现为"别名没生效"）
                problems.append(f"第 {lineno} 行别名语法不完整: {line!r}")
                continue
            if alias.lower() == canon.lower():
                terms.add(canon.lower())
                continue
            terms.add(alias.lower())
            terms.add(canon.lower())
            prev = aliases.get(alias.lower())
            if prev is not None and prev.lower() != canon.lower():
                problems.append(
                    f"第 {lineno} 行别名 {alias!r} 重复指向不同规范名"
                    f"（已保留 {prev!r}，忽略 {canon!r}）"
                )
                continue
            aliases[alias.lower()] = canon
        return terms, aliases, problems



    @classmethod
    def _load_lexicon(cls, path: str | None) -> set[str]:
        """读取词库文件，返回小写化词条集合（含别名）。

        文档场景下，靠启发式规则猜中文专名注定噪声缠身；让项目自己给一份
        专名清单（人名 / 地名 / 商号 / 书名），命中即链，既准又便宜。
        """
        return cls._read_lexicon(path)[0]



    @classmethod
    def _read_lexicon(cls, path: str | None) -> tuple[set[str], dict[str, str]]:
        """读取词库文件，返回 ``(词条集合, 别名映射)``。读不到就是空。"""
        if not path or not os.path.isfile(path):
            return set(), {}
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError as e:
            logger.warning("专名词库读取失败 %s: %s", path, e)
            return set(), {}
        terms, aliases, problems = cls._parse_lexicon_text(text)
        for p in problems:
            logger.warning("专名词库 %s %s", path, p)
        return terms, aliases




    @staticmethod
    def _preprocess_for_entities(text: str) -> str:
        """剥离 markdown 结构后再抽实体（覆盖报告·高 6）。

        文档类语料里，表格行（``| 卷末（1664 | ...``）、树形目录
        （``│  ├─ 伏笔表.md``）、围栏代码块会被启发式规则当成专名，
        实测 Top15 实体里 11 个是这类碎片。这些字符不承载"实体"语义，
        在抽取前整段去掉，噪声源就没了。
        """
        if not text:
            return ""
        t = _MD_CODE_FENCE.sub(" ", text)
        lines = []
        for line in t.split("\n"):
            if _MD_TABLE_LINE.match(line):
                continue
            # 行内的树形符号也去掉（"├─ 顾言" → "顾言"）
            if any(ch in line for ch in _MD_TREE_CHARS):
                line = "".join(" " if ch in _MD_TREE_CHARS else ch for ch in line)
            lines.append(line)
        return "\n".join(lines)



    @staticmethod
    def _cjk_ratio(text: str) -> float:
        """中日韩字符占"非空白字符"的比例。"""
        if not text:
            return 0.0
        dense = [c for c in text if not c.isspace()]
        if not dense:
            return 0.0
        cjk = sum(1 for c in dense if "\u4e00" <= c <= "\u9fa5")
        return cjk / len(dense)




    @classmethod
    def _clean_entity_name(cls, name: str) -> str | None:
        """清洗抽出的候选实体：丢弃功能词、结构字符与编号式噪声。

        覆盖报告·高 6 的实测垃圾样本（「├」「│  ├─ 伏笔表.md」「H05」
        「本文件（人读）」「章自检（2026」）都在这里被拦下：
        结构字符走 ``_ENTITY_BAD_CHARS``（markdown 已在预处理里剥离，
        这里兜住残留），编号式走 ``_CODE_LIKE``。
        """
        n = (name or "").strip()
        if not n:
            return None
        # 中文名放 12 字，纯英文名放宽到 30（New York City / SentenceTransformer）
        limit = 30 if n.isascii() else 12
        if not (1 < len(n) <= limit):
            return None
        # 含标点 / markdown 结构字符 → 噪声（实体名不该带这些）
        if any(c in _ENTITY_BAD_CHARS for c in n):
            return None
        # 纯功能词 / 含功能词的短语 → 噪声
        if any(c in n for c in cls._NOISE_CHARS):
            return None
        if re.fullmatch(r"[\d\W_]+", n):
            return None
        if cls._CODE_LIKE.match(n):
            return None
        return n




    @classmethod
    def _title_entities(cls, text: str) -> list[str]:
        """抽取 `《书名号》` 内的专名。中文唯一的显式作品名标记，可信度单列一档。"""
        names = []
        for m in cls._TITLE_RE.finditer(text):
            w = (m.group(1) or "").strip()
            if 1 < len(w) <= 12:
                names.append(w)
        return names



    @classmethod
    def _heuristic_entities(cls, text: str, cjk: bool = True) -> list[str]:
        """中英文启发式实体抽取（spaCy 不可用或非英文时的回退）。

        中文模式：我叫X / X品类 / X公司（句式特征）。
        英文模式：额外启用引号模式 + 连续大写开头的词。

        `《书名》` 不在这里处理——它由 ``_title_entities`` 单独抽出并标 title 档。

        Args:
            cjk: 文本是否以中文为主。为真时**停用引号类模式**（见
                ``_QUOTE_PATTERNS`` 的说明——中文引号是强调标记而非专名标记）。
        """
        names: list[str] = []
        patterns = list(cls._STATEMENT_PATTERNS)
        if not cjk:
            patterns.extend(cls._QUOTE_PATTERNS)
        for pat in patterns:
            for m in re.finditer(pat, text):
                w = (m.group(1) or "").strip()
                if 1 < len(w) <= 12:
                    names.append(w)
        if not cjk:
            for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
                w = m.group(1).strip()
                if w.lower() not in _STOPWORDS and len(w) > 1:
                    names.append(w)
        return names




    @staticmethod
    def _entity_origin(ent: dict[str, Any]) -> str:
        """读取实体来源。**缺省 heuristic** —— 旧库序列化里没有该字段，
        按不可信处理，绝不因为"数据老"就白给信任。"""
        return ent.get("origin", ENTITY_ORIGIN_HEURISTIC)



    @classmethod
    def _entity_trusted(cls, ent: dict[str, Any]) -> bool:
        """该实体是否可信（参与检索加成 / 画像 / 矛盾检测）。"""
        return cls._entity_origin(ent) in TRUSTED_ENTITY_ORIGINS




    @staticmethod
    def _drift_row(source: str, category: str,
                   ref: dict[str, int] | None,
                   sig: dict[str, Any] | None,
                   cur: dict[str, Any] | None,
                   diff: list[str] | None = None) -> dict[str, Any]:
        """漂移审计的一行（实现体在 ``store.source_sig.drift_row``，
        此处为同名委托——C1b 漂移审计随 source_drift 成簇搬 store 侧，
        ``_PureMixin`` 表面不变，行为由 tests/test_c1_source_drift.py 钉住）。"""
        return _source_sig.drift_row(source, category, ref, sig, cur, diff)




    @staticmethod
    def _split_facts(text: str) -> list[str]:
        """把一段文本切成可作为记忆的短句。"""
        text = text.strip()
        if not text:
            return []
        parts = re.split(r"(?<=[。！？!?；;\n])|(?<=[.!?])\s+", text)
        facts, buf = [], ""
        for p in parts:
            p = (p or "").strip()
            if not p:
                continue
            buf = f"{buf} {p}".strip() if buf else p
            if len(buf) >= 24:
                facts.append(buf)
                buf = ""
        if buf:
            if facts and len(buf) < 18:
                facts[-1] = f"{facts[-1]} {buf}".strip()
            else:
                facts.append(buf)
        return facts or [text]



    @staticmethod
    def _guess_cognitive_type(text: str, speaker: str) -> str:
        t = text.lower()
        if speaker in ("user", "assistant"):
            if any(m in t for m in ("we ", "我们一起", "上次我们", "i did", "you did")):
                return CognitiveType.EXPERIENCE.value
        if any(m in t for m in ("i think", "i feel", "probably", "maybe", "我觉得", "我认为", "可能")):
            return CognitiveType.OPINION.value
        return CognitiveType.WORLD.value



    @staticmethod
    def _guess_importance(text: str) -> float:
        """启发式重要性（判定核已迁 ``domain.policies.guess_importance``，
        此处为同名委托，行为经 golden 测试 44 例钉住）。"""
        return _policies.guess_importance(text)




    @staticmethod
    def _tokens_of(s: str) -> set[str]:
        """中文 bigram + 英文词（委托 ``domain.policies.tokens_of``）。"""
        return _policies.tokens_of(s)



    @classmethod
    def _overlap_of_sets(cls, ta: set[str], tb: set[str]) -> float:
        """M2：委托 ``domain.policies.overlap_of_sets``（cls 形参保留防签名变动）。"""
        return _policies.overlap_of_sets(ta, tb)



    @classmethod
    def _token_overlap(cls, a: str, b: str) -> float:
        """实词重叠率（用于判断两条记忆是否在讲同一件事）。"""
        return cls._overlap_of_sets(cls._tokens_of(a), cls._tokens_of(b))




    def _resolve_daily_days(self, daily_days: int = 0) -> int:
        """按日窗口条数：显式参数 > env ``HIPPOCAMPUS_DAILY_ACTIVITY_DAYS`` > 14。

        ``0`` = 按 env / 默认来。**不设上界校验**——要看全量就传个大数，
        反正 ``daily_counts`` 本身最多只有 :data:`DAILY_COUNT_KEEP_DAYS` 天。
        """
        if daily_days:
            return max(int(daily_days), 1)
        raw = load_config().daily_activity_days
        if raw:
            try:
                v = int(raw)
                if v > 0:
                    return v
            except ValueError:
                pass
        return DAILY_ACTIVITY_DEFAULT_DAYS




    def _mmr_select(self, items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """MMR 选择：兼顾相关性与多样性，抑制重复内容霸榜。"""
        selected: list[dict[str, Any]] = []
        pool = list(items)
        while pool and len(selected) < top_k:
            best, best_val = None, -1e9
            for cand in pool:
                relevance = float(cand["score"])
                if not selected:
                    val = relevance
                else:
                    max_sim = 0.0
                    for sel in selected:
                        s = cand.get("_sel_sim", {}).get(sel["memory_id"])
                        if s is None:
                            a, b = cand.get("_emb"), sel.get("_emb")
                            if a is None or b is None:
                                # 缺当前模型向量的候选（BM25 单路命中）不做多样性惩罚
                                continue
                            denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-12
                            s = float(a @ b / denom)
                            cand.setdefault("_sel_sim", {})[sel["memory_id"]] = s
                        max_sim = max(max_sim, s)
                    # 字面强命中（BM25 单路前 N 名）豁免多样性惩罚：
                    # 否则"带章号/数额的那条"会被近义的已选项折叠掉（见常量处实测）。
                    penalty = 0.0 if cand.get("_literal") else 1.0
                    val = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * max_sim * penalty
                if val > best_val:
                    best, best_val = cand, val
            selected.append(best)  # type: ignore[arg-type]
            pool.remove(best)  # type: ignore[arg-type]
        return selected




    @staticmethod
    def _render_blocks(blocks: dict[str, list[str]], token_budget: int,
                       item_max_chars: int = 200) -> str:
        labels = {
            "block_1_current_hits": "📋 当前相关信息",
            "block_2_pending": "❓ 待办 / 未决问题",
            "block_3_timeline": "📍 时间线（最近会话）",
            "block_4_entity_state": "🧠 相关实体状态",
            "block_5_open_topics": "🔄 上次未聊完",
        }
        budget = max(int(token_budget) * 2, 200)  # 中文约 2 字符/token

        # 先整块渲染，再按优先级（block_1 → block_5）装填。
        # 覆盖报告·低 13：原来是 `out[: budget*2]` 硬截断，会把一行切成半句，
        # 宿主拿到"【📋 当前相关信息】\n  - 顾言表字是"这种断头内容。
        # 现在**只按整块/整条丢弃**，绝不切在行中间。
        rendered: list[tuple[str, list[str]]] = []
        for k, label in labels.items():
            vals = [v for v in blocks.get(k, []) if v]
            if not vals:
                continue
            clip = item_max_chars if item_max_chars > 0 else None
            rendered.append((f"【{label}】",
                             [f"  - {v[:clip] if clip else v}" for v in vals]))

        parts: list[str] = []
        used = 0
        for header, lines in rendered:
            # 1) 整块能放下就整块放
            block = header + "\n" + "\n".join(lines)
            gap = 2 if parts else 0
            if used + gap + len(block) <= budget:
                parts.append(block)
                used += gap + len(block)
                continue
            # 2) 放不下整块 → 保留块头、逐条装到放不下为止（条目仍完整）
            kept: list[str] = []
            here = len(header) + gap
            if used + here <= budget:
                for line in lines:
                    if used + here + len(line) + 1 > budget:
                        break
                    kept.append(line)
                    here += len(line) + 1
            if kept:
                parts.append(header + "\n" + "\n".join(kept))
                used += here
            # 3) 连块头都放不下 → 整块丢弃（后面的块优先级更低，自然也不放）
        return "\n\n".join(parts)




    def resolve_knowledge_limit(self, knowledge_limit: int = 0) -> int:
        """知识地图槽条数：显式参数 > env ``HIPPOCAMPUS_KNOWLEDGE_BLOCK_LIMIT`` > 默认 8。

        ``0`` = 按 env / 默认来；**负数 = 关闭该槽**（与 ``pin_limit`` 同款语义）。
        """
        if knowledge_limit:
            return max(int(knowledge_limit), 0)
        raw = load_config().knowledge_block_limit
        if raw:
            try:
                v = int(raw)
                if v != 0:
                    return max(v, 0)
            except ValueError:
                pass
        return KNOWLEDGE_BLOCK_LIMIT_DEFAULT




    def resolve_pin_limit(self, pin_limit: int = 0) -> int:
        """钉住槽条数：显式参数 > env ``HIPPOCAMPUS_PIN_BLOCK_LIMIT`` > 默认 3。

        ``0`` = 按 env / 默认来；**负数 = 关闭该槽**（与 ``recent_limit`` 同款语义）。
        """
        if pin_limit:
            return max(int(pin_limit), 0)
        raw = load_config().pin_block_limit
        if raw:
            try:
                v = int(raw)
                if v > 0:
                    return v
            except ValueError:
                pass
        return PIN_BLOCK_LIMIT_DEFAULT




    def resolve_user_scope_limit(self, user_scope: int = 0) -> int:
        """用户级继承槽条数三态（逐行照抄的三态表）：

        ====================  ==========================================
        输入                  resolve_user_scope_limit
        ====================  ==========================================
        param > 0             param
        param < 0             0（关）
        param == 0，env > 0   env
        param == 0，env < 0   0（关）
        param == 0，env 0/空/非法  3（默认）
        ====================  ==========================================

        与 :meth:`resolve_pin_limit` 的**唯一**差别在 env 负值：pin 落回
        默认（``v > 0`` 门），user_scope 的设计是显式关——``0``（关）与
        "回默认"必须分得开，「注入可关」纪律才有 env 侧的关法。
        env 的空/非法回落由 :func:`config.load_config` 的 ``_int`` 兜
        （回落字段默认 3，与 :data:`USER_SCOPE_LIMIT_DEFAULT` 同值）。
        """
        if user_scope:
            return max(int(user_scope), 0)
        raw = load_config().user_scope_limit
        if raw:
            return max(int(raw), 0)
        return USER_SCOPE_LIMIT_DEFAULT

    def resolve_task_directory_limit(self, task_directory: int = 0) -> int:
        """任务目录槽条数三态（U5，设计 §3.4）：与 ``user_scope`` 同族。

        ====================  ==================================
        输入                  resolve_task_directory_limit
        ====================  ==================================
        param > 0             param
        param < 0             0（关）
        param == 0，env > 0   env
        param == 0，env < 0   0（关）
        param == 0，env 0/空/非法  8（默认）
        ====================  ==================================

        与 :meth:`resolve_user_scope_limit` 逐行同构（含 env 负值＝显式关），
        唯一的差别是默认值 8。env 的空/非法回落由 ``config.load_config`` 的
        ``_int`` 兜（回落字段默认 8，与 :data:`TASK_DIRECTORY_LIMIT_DEFAULT`
        同值）。
        """
        if task_directory:
            return max(int(task_directory), 0)
        raw = load_config().task_directory_limit
        if raw:
            return max(int(raw), 0)
        return TASK_DIRECTORY_LIMIT_DEFAULT




    @staticmethod
    def _tag_ops_brief(ops: list[dict[str, Any]],
                       limit: int = 5) -> list[dict[str, Any]]:
        """账本摘要（给'没有这个 op'的报错带路：宿主知道有哪些 id 可用）。"""
        return [
            {"op_id": o.get("op_id"), "op": o.get("op"), "ts": o.get("ts"),
             "to_change": o.get("to_change"), "applied": o.get("applied"),
             "filter": o.get("filter")}
            for o in ops[-limit:]
        ]




    def _knowledge_playbook(self, pending_n: int, shown_n: int) -> dict[str, Any]:
        """「整理这个库」的操作说明书（给宿主看，跟着走就能完成一轮归纳）。

        刻意写得像菜谱而不是像文档：宿主是照着步骤执行的，含糊一句"你可以
        归纳一下"换不来任何动作。三条规则是从踩过的坑里来的——evidence 凭空
        引用会被整条退回、已有归纳不该重做、写坏了要能撤。
        """
        batch = min(10, max(1, pending_n))
        return {
            "goal": ("把 %d 个知识点归纳成上层结论。**归纳智能在你这边**，"
                     "我只能出素材、做校验、存起来——本服务生产路径不内嵌 LLM。" % pending_n),
            "steps": [
                '1) 取素材：knowledge_topic(action="brief_batch", batch=%d)'
                '（不传 topics 就自动挑最该归纳的 %d 个）' % (batch, batch),
                "2) 你读每个主题的素材，写成一句结论（≤2000 字，写清主体＋事实）",
                "3) 写回：knowledge_topic(action=\"summarize_batch\", "
                "items='[{\"topic_id\":\"...\",\"summary\":\"...\","
                "\"evidence\":[\"...\"]}]')（items 是 JSON 数组字符串）",
                "4) 重复 1-3 直到 pending_count 为 0（每批会自动跳过已写好的）",
            ],
            "rules": [
                "evidence 必须是该主题 brief 里列出的 id——凭空引用会**整条退回**，"
                "一个字段都不改（改一半比不改更难查）",
                "已有归纳且没标 ⚠️ 的会被自动跳过，别浪费力气重做；"
                "要重写就用 topics 显式点名",
                "写坏了一条：knowledge_topic(action=\"revert\", topic_id=...) 撤销",
                "写回即落盘（带账本），不用另外调 save",
                # S3：四字段契约。写进 rules 而不是新增参数文档——宿主照菜谱
                # 走一遍就会用；f/s 两格是已知最便宜的"假簇过滤器"。
                "可选四字段（推荐）：item 里加 \"fras\":{\"f\":\"范畴（当标题）\","
                "\"r\":[\"关联实体/主题\"],\"a\":\"可引用的结论\",\"s\":\"推不出来"
                "但搞错就出事的约束\"}。四键必须齐全；s 确实没有也要写\"无\"。"
                "只传 fras 不传 summary 时，a 就是那句话",
            ],
            "done_when": "pending_count == 0",
        }




    # ------------------------------------------------------------------
    # L0 与 L1 的"冲突即合并"语义对齐
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_raw_log(disk: list[dict], mine: list[dict]) -> list[dict]:
        """两条 L0 流水按 id 取并集（**只做加法，不覆盖、不删除**）。

        为什么 L0 需要这条：L1（``memories.json``）早就有「冲突即合并」的加法
        语义（``manager.retry_save``：取磁盘最新版本 + 补上磁盘没有的 id），
        而 L0（``brain_raw.json``）此前是**直接覆写 / 直接替换**——两个进程各
        自写同一个库时，就会出现「L1 是并集、L0 是较短的那一份」，于是 L1 的
        ``source_ids`` 指不到 L0 原文。中文小说库实测 21 个 id / 36 条 L1 悬空
        （追加五「L0 簿记漂移」），根因就在这层语义不对齐。

        只做加法是刻意的：L0 是 append-only 流水，删除也是软删除（打
        ``forgotten`` 标记，条目仍在），所以并集不会复活任何被删的东西。
        同 id 两条内容不同时**以磁盘为准**（与 retry_save 对 L1 的取舍一致）。
        """
        if not disk:
            return list(mine)
        by_id = {str(r.get("id")): r for r in disk if r.get("id")}
        merged = list(disk)
        for r in mine:
            rid = r.get("id")
            if rid and str(rid) not in by_id:
                merged.append(r)
                by_id[str(rid)] = r
        return merged




    @staticmethod
    def _layer_key(item: dict) -> str:
        """list 层的合并键：有 id 用 id，没有就用内容指纹（避免无 id 层整份被盖）。"""
        rid = item.get("id")
        if rid:
            return "id:" + str(rid)
        fp = {k: item.get(k) for k in
              ("text", "query", "ts", "timestamp", "session_id", "reason", "name")}
        return "fp:" + json.dumps(fp, sort_keys=True, ensure_ascii=False,
                                  default=str)



    @classmethod
    def _merge_list_layer(cls, disk, mine, cap: int = 0):
        """list 层并集：磁盘打底，本进程没见过的补进来；同键以**本进程**为准。

        只做加法是刻意的 —— 这些层都是流水/账本语义，并集不会复活任何被删的
        东西（删除走墓碑或软标记）。
        """
        if not isinstance(disk, list):
            return list(mine) if isinstance(mine, list) else disk
        if not isinstance(mine, list):
            return list(disk)
        out: dict[str, dict] = {}
        order: list[str] = []
        for item in list(disk) + list(mine):
            if not isinstance(item, dict):
                continue
            k = cls._layer_key(item)
            if k not in out:
                order.append(k)
            out[k] = item
        merged = [out[k] for k in order]
        if cap and len(merged) > cap:
            merged = merged[-cap:]
        return merged



    @staticmethod
    def _merge_dict_layer(disk, mine):
        """dict 层并集：磁盘打底，本进程的键覆盖（本地优先）。"""
        if not isinstance(disk, dict):
            return mine
        if not isinstance(mine, dict):
            return disk
        out = dict(disk)
        out.update(mine)
        return out



    @staticmethod
    def _merge_entity_layer(disk, mine):
        """实体层：键并集，``memory_ids`` 取**并集**、``mention_count`` 取 max。

        直接取本地那份会把对方挂上来的记忆 id 抹掉（实测现象）。
        mention_count 反正 load 时会按存活记忆重算（``_recompute_entity_stats``），
        取 max 只是让合并期间不虚降。
        """
        if not isinstance(disk, dict):
            return mine
        if not isinstance(mine, dict):
            return disk
        out: dict[str, Any] = {}
        for k in set(disk) | set(mine):
            a, b = disk.get(k), mine.get(k)
            if a is None:
                out[k] = b
                continue
            if b is None:
                out[k] = a
                continue
            if not (isinstance(a, dict) and isinstance(b, dict)):
                out[k] = b
                continue
            merged = dict(a)
            merged.update(b)
            if "memory_ids" in a or "memory_ids" in b:
                ids = list(a.get("memory_ids") or []) + list(b.get("memory_ids") or [])
                merged["memory_ids"] = sorted({str(x) for x in ids})
            mc = [x for x in (a.get("mention_count"), b.get("mention_count"))
                  if isinstance(x, (int, float))]
            if mc:
                merged["mention_count"] = max(mc)
            out[k] = merged
        return out



    @classmethod
    def _merge_intent_layer(cls, disk, mine):
        """待办层：键并集 + **状态只进不退**。

        为什么要单独一条规则：通用 list 合并是「磁盘打底、本地覆盖」，
        对 append-only 的账本没问题（先来后到无所谓的条目），但有终态的层
        不行 —— A 标记了 done，B 手里那份还是 open，B 一落盘就把 done 抹回
        open，两边全程无报错（两个实例反复复现过）。

        所以这里的取舍是：**status 取 rank 更大的**，其余字段仍以本地为准。
        「完成」是这类语义里唯一不可逆的部分。
        """
        merged = cls._merge_list_layer(disk, mine)
        if not isinstance(disk, list) or not isinstance(merged, list):
            return merged
        disk_by_key: dict[str, dict] = {}
        for r in disk:
            if isinstance(r, dict):
                disk_by_key[cls._layer_key(r)] = r
        rank = cls._INTENT_STATUS_RANK
        out: list[dict] = []
        for it in merged:
            if not isinstance(it, dict):
                out.append(it)
                continue
            other = disk_by_key.get(cls._layer_key(it))
            if isinstance(other, dict):
                mine_rank = rank.get(str(it.get("status") or ""), 0)
                disk_rank = rank.get(str(other.get("status") or ""), 0)
                if disk_rank > mine_rank:
                    fixed = dict(it)
                    fixed["status"] = other["status"]
                    # 完成时的留痕字段也一并取过来，别留下「完成了但没时间」
                    for extra in ("done_at", "resolved_at", "completed_at"):
                        if other.get(extra) and not fixed.get(extra):
                            fixed[extra] = other[extra]
                    it = fixed
            out.append(it)
        return out



    @classmethod
    def _merge_session_layer(cls, disk, mine):
        """会话层：键并集 + **终态优先**。

        对方已经 ended（写了 ended_at）而本地这路还没收尾时，若按"本地优先"
        会把 ended_at 抹掉 —— 已结束的会话在 ``active_sessions`` 里复活。
        """
        out = cls._merge_dict_layer(disk, mine)
        if not isinstance(disk, dict) or not isinstance(out, dict):
            return out
        for k, v in disk.items():
            cur = out.get(k)
            if not (isinstance(cur, dict) and isinstance(v, dict)):
                continue
            if v.get("ended_at") and not cur.get("ended_at"):
                merged = dict(cur)
                merged["ended_at"] = v["ended_at"]
                if v.get("summary") and not merged.get("summary"):
                    merged["summary"] = v["summary"]
                out[k] = merged
        return out



    @staticmethod
    def _merge_counter_layer(disk, mine):
        """按日计数：同键取 **max**，刻意不求和。

        同一个键两侧可能都带着同一份基线（各自 load 时读到的），求和会把
        「该想起率」的分母吹大；取 max 顶多少计，不会虚增。
        """
        if not isinstance(disk, dict):
            return mine
        if not isinstance(mine, dict):
            return disk
        out = dict(disk)
        for k, v in mine.items():
            old = out.get(k)
            if isinstance(old, dict) and isinstance(v, dict):
                merged = dict(old)
                for kk, vv in v.items():
                    if (isinstance(vv, (int, float))
                            and isinstance(old.get(kk), (int, float))):
                        merged[kk] = max(old[kk], vv)
                    else:
                        merged[kk] = vv
                out[k] = merged
            else:
                out[k] = v
        return out




    # ------------------------------------------------------------------
    # 派生层「真删除」账本
    #
    # 并集只会做加法。原来的解法是「合并完按当前存活记忆重裁」，
    # 那只救得了**派生副本**（实体 / 画像，源头在 L1 可重算）。miss_log /
    # sessions / intents 这些**独立账本**没有可重裁的源头，删了就是删了。
    #
    # 实测（_probe_out/_f92_delete_gap_20260929.py）：A 清空 miss_log 并落盘，
    # B（旧快照）一次 save 就把 3 条完整并回来，两边全程无报错。
    # 所以要有一份"哪些键被删过、什么时候删的"账本，合并时据此拦截。
    # ------------------------------------------------------------------

    @staticmethod
    def _del_ledger_key(layer: str, key: str) -> str:
        """账本键：``层文件名::条目键``（与 :meth:`_layer_key` 的取值对齐）。"""
        return f"{layer}::{key}"




    @staticmethod
    def _merge_del_ledger_layer(disk, mine):
        """删除账本：键并集，同键取**更晚**的删除时刻。

        刻意不走 dict 规则的「本地优先」—— 两侧都是删除记录，取更晚的那次
        才不会把对方的删除**记弱**（记弱 = 拦不住旧版本，等于没记）。
        """
        if not isinstance(disk, dict):
            return dict(mine) if isinstance(mine, dict) else {}
        out = dict(disk)
        if isinstance(mine, dict):
            for k, v in mine.items():
                old = out.get(k)
                if isinstance(v, (int, float)) and isinstance(old, (int, float)):
                    out[k] = max(old, v)
                else:
                    out[k] = v
        return out



    @staticmethod
    def _entry_ts(item: dict) -> float:
        """条目自己的时间戳（用于判断"是不是删完之后又重新产生的"）。

        取不到就返回 0.0 —— 保守地当作"很老"，于是只要立过碑就拦住。
        """
        for k in ("ts", "timestamp", "created_at", "started_at", "done_at"):
            v = item.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0




    @staticmethod
    def _merge_topic_layer(disk, mine):
        """主题层：键并集；``member_ids`` 取**并集**，标量取较新的一侧。

        为什么不直接走 dict 规则的「本地优先」：``member_ids`` 是这一层的
        本体——取本地那份会把对方挂上来的成员抹掉（同款事故）。而
        ``summary`` 的取法是**host 优先于时间戳**（见方法体注释）；都不是
        host 时才按 ``updated_at`` 取较新的一侧。
        """
        def _span_pair(raw):
            """把任意形态的 span 收敛成 [lo, hi] 两个 float。

            ``[]`` / ``None`` / 只有一个元素 / 元素是字符串——都挡得住。
            派生层合并是保存链的最后一道工序，它不该因为一条脏数据把整次
            保存带塌（原实现 ``float(span)[0]`` 对 list 直接 TypeError）。
            """
            try:
                s = list(raw or [])
                if len(s) < 2:
                    return [0.0, 0.0]
                return [float(s[0]), float(s[1])]
            except (TypeError, ValueError):
                return [0.0, 0.0]

        if not isinstance(disk, dict):
            return mine
        if not isinstance(mine, dict):
            return disk
        # 世代：重建 ⇒ +1，**高的整代覆盖**。
        # 键并集只在同一代之间做——跨代做并集正是「旧簇复活」的成因：
        # 磁盘 652 个新簇 ∪ 对面内存 698 个旧簇 = 1050 个。而旧写者
        # 的层世代更低，它下次保存时会**主动让位**给磁盘这份，
        # 不会再并一次——这才是「清得掉」。
        e_disk = topic_mod.layer_epoch(disk)
        e_mine = topic_mod.layer_epoch(mine)
        if e_mine > e_disk:
            return dict(mine)
        if e_disk > e_mine:
            return dict(disk)
        out: dict[str, Any] = {}
        for k in set(disk) | set(mine):
            a, b = disk.get(k), mine.get(k)
            if a is None:
                out[k] = b
                continue
            if b is None:
                out[k] = a
                continue
            if not (isinstance(a, dict) and isinstance(b, dict)):
                out[k] = b
                continue
            merged = dict(a)
            merged.update(b)
            ids = list(a.get("member_ids") or []) + list(b.get("member_ids") or [])
            merged["member_ids"] = sorted({str(x) for x in ids})
            merged["size"] = len(merged["member_ids"])
            # 归纳合并：**host 优先于时间戳**。
            # 抽取式代表句下一次 refresh 就能重算（丢了不心疼），宿主归纳是
            # 花力气写的、且本系统永远不会自动重新生成它——丢了就真没了。
            # 因此两侧只要有一侧是 host，就取那一侧；都不是（或都是）才按
            # updated_at 取新。初版只按时间取，现在让 host 归纳变常态
            # 后，巩固刷新会静默把归纳冲成摘录——这正是"最伤信任的静默改写"。
            ka = str(a.get("summary_kind") or "")
            kb = str(b.get("summary_kind") or "")
            if ka == topic_mod.SUMMARY_KIND_HOST and kb != topic_mod.SUMMARY_KIND_HOST:
                newer = a
            elif kb == topic_mod.SUMMARY_KIND_HOST and ka != topic_mod.SUMMARY_KIND_HOST:
                newer = b
            else:
                ta = float(a.get("updated_at") or 0.0)
                tb = float(b.get("updated_at") or 0.0)
                newer = b if tb > ta else a
            merged["summary"] = newer.get("summary", "")
            merged["summary_kind"] = newer.get("summary_kind", "")
            merged["summary_evidence"] = list(newer.get("summary_evidence") or [])
            # 写回元信息跟着选中那一侧走（签名/时间/规模必须与 summary 同源，
            # 否则 stale 判断会拿 A 的归纳配 B 的成员，报出假过期）。
            for f in ("summary_at", "summary_signature", "summary_member_count"):
                merged[f] = newer.get(f, a.get(f))
            # S3：四字段与宿主标题同样跟着选中侧——它们与 summary 是一套
            # （fras.a 往往就是 summary 本体）。跟错侧的后果：地图上显示
            # A 的结论、点进去是 B 的约束，宿主按错的那份做决策。
            if newer.get("summary_fras"):
                merged["summary_fras"] = newer.get("summary_fras")
            else:
                merged.pop("summary_fras", None)
            if str(newer.get("title_kind") or "") == topic_mod.TITLE_KIND_HOST:
                merged["title"] = newer.get("title", merged.get("title"))
                merged["title_kind"] = topic_mod.TITLE_KIND_HOST
            else:
                # 选中侧没有宿主命名 ⇒ 清掉另一侧残留的宿主标题。标题与
                # summary 是一套：留着 B 起的名字配 A 的结论，就是"地图显示
                # A 的结论、顶着一个 B 谁也对不上的名字"。与上面 fras 的
                # pop 是同一条纪律（负对照实测过这个残留路径）。
                merged.pop("title_kind", None)
                merged["title"] = newer.get("title", "")
            merged["recall_hits"] = max(int(a.get("recall_hits") or 0),
                                        int(b.get("recall_hits") or 0))
            # span 取并集（最早, 最晚）。**别写成 float(span)[0]**：span 本身
            # 就是 list，对 list 做 float 直接 TypeError —— 那不是"算错"，
            # 是整次保存在这里炸掉（多进程共库撞上同一个主题才触发）。
            sa, sb = _span_pair(a.get("span")), _span_pair(b.get("span"))
            merged["span"] = [min(sa[0], sb[0]), max(sa[1], sb[1])]
            out[k] = merged
        return out



    @staticmethod
    def _merge_sourcesig_layer(disk, mine):
        """C1a：源签名层合并——同键取 ``sig_at`` 较新者。

        两个进程 ingest 同一个文件时都会记签名：后校验的那份更接近磁盘
        现状。sha1 相同时事实无差别，取谁都对；不同时若误取旧的，drift
        会把一个没变过的文件报成 stale（狼来了两次就没人看审计了）。
        """
        if not isinstance(disk, dict) or not isinstance(mine, dict):
            return mine if isinstance(mine, dict) else disk
        out = dict(disk)
        for k, b in mine.items():
            a = out.get(k)
            if not isinstance(b, dict):
                continue
            if not isinstance(a, dict) or float(b.get("sig_at") or 0.0) >= float(
                    a.get("sig_at") or 0.0):
                out[k] = b
        return out




    @staticmethod
    def _stamp_of(path: str):
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)




    # -- F14-1 层版本号 / G0 派生层脏集 ---------------------------------
    @staticmethod
    def _layer_fingerprint(value: Any) -> str:
        """层内容指纹（＝层版本号）。

        为什么用**内容哈希**而不是逐条脏标记：10 个层形态各异（list / dict /
        counter），逐条标脏要在几十处修改点插桩，漏一处就是静默漏写；而内容
        哈希只有一个语义——「和我上次写下去的那份相比变没变」，不存在漏插。

        ``sort_keys=True`` 是必须的：dict 的插入顺序在不同进程里可能不同，
        不排序会让"内容其实一样"被判成变了（白白重写）。
        """
        try:
            blob = json.dumps(
                value, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), default=str,
            )
        except (TypeError, ValueError):  # pragma: no cover - 兜底
            blob = repr(value)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
