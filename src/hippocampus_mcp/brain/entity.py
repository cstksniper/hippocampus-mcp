# -*- coding: utf-8 -*-
"""B2-3 P2：实体与专名词库簇

本簇从 ``brain.py`` 逐字搬来。

为什么这一簇能整体搬：它写的 5 个属性（``_alias_reverse`` /
``_lexicon_mtime`` / ``entity_lexicon`` / ``entity_lexicon_aliases`` /
``entity_lexicon_path``）**簇外写者只有 ``__init__``** —— 写者成簇，
搬走不会把一个属性的写者拆散到两个文件里。这是 AST 数出来的，
不是按命名猜的。

P1 已经先搬走了这簇里的 9 个纯方法（``_read_lexicon`` /
``_parse_lexicon_text`` / ``_clean_entity_name`` …），本模块
只装剩下那些真正碰实例状态的。

⚠️ ``discover_aliases`` 搬走后，相关的结构守卫断言（原本只扫
brain.py）已同步改为同时扫本文件——否则那个结构守卫会假失败。
"""

from __future__ import annotations

from ..common.constants import *  # noqa: F401,F403  （常量表：见 common/constants.__all__）
from ..common.constants import logger
import os
import re
from typing import Any
from ..domain import aliases as alias_mod
from ..store.memory import MemoryState
from ..config import load_config  # noqa: E402  （env 读收编 config）


class _EntityLexiconMixin:
    """B2-3 P2：实体与专名词库簇"""

    def reload_lexicon(self) -> int:
        """重新加载专名词库（词库文件改了不用重启服务）。返回词条数。

        路径会重新解析：启动时若词库文件还不存在（``path=None``），
        之后补上文件再调用本方法也能生效。
        """
        persist_dir = getattr(self.manager, "persist_dir", None) or self.persist_dir
        self.entity_lexicon_path = self._resolve_lexicon_path(
            self._explicit_lexicon_path, persist_dir
        )
        self.entity_lexicon, self.entity_lexicon_aliases = self._read_lexicon(
            self.entity_lexicon_path
        )
        self._alias_reverse = None      # 反向索引失效（别名映射变了）
        self._lexicon_mtime = self._lexicon_mtime_of(self.entity_lexicon_path)
        return len(self.entity_lexicon)



    def _ensure_default_lexicon_path(self) -> str | None:
        """默认词库路径；**文件不存在就创建它**（解开"没词库就建不了词库"的死锁）。

        返回 None = 连 persist_dir 都没有（纯内存库）或目录不可写 —— 只有这时
        才真的没法落盘，交给调用方报错（报错要说清查过哪三条路径）。
        """
        persist_dir = getattr(self.manager, "persist_dir", None) or self.persist_dir
        if not persist_dir:
            return None
        default_lex = os.path.join(persist_dir, "entity_lexicon.txt")
        if not os.path.isfile(default_lex):
            try:
                with open(default_lex, "a", encoding="utf-8"):
                    pass        # touch：先建一个空词库，随后按正常路径追加
            except OSError:
                return None
        return default_lex




    def _reload_lexicon_if_changed(self) -> int:
        """词库文件变了就热加载一遍（**不新增 MCP 工具**：挂在 ``stats()`` 上顺带做）。

        为什么需要它：daemon 是长驻进程，而词库路径在构造时只解析一次
        （:meth:`_resolve_lexicon_path`）。服务启动那刻 ``entity_lexicon.txt``
        还不存在 ⇒ ``path=None``；用户之后把文件放进记忆库，本进程**不会**
        重新解析，``entity_lexicon_size`` 就恒为 0，而 ``lexicon_add`` 又因为
        ``path=None`` 直接抛错 —— 死锁，只能重启 daemon。
        （实测确认：这正是「词库热加载失效」的真实形态，不是路径没回退。）

        返回词条数。异常一律吞掉并保留原样：自曝归自曝，不能让 ``stats()`` 挂掉。

        线程安全：只**替换** ``entity_lexicon`` 等引用，不原地改集合。并发线程
        可能短暂读到旧引用，但不会崩（daemon 多线程共享 brain 的既有约定）。
        """
        persist_dir = getattr(self.manager, "persist_dir", None) or self.persist_dir
        path = self._resolve_lexicon_path(self._explicit_lexicon_path, persist_dir)
        mtime = self._lexicon_mtime_of(path)
        if path == self.entity_lexicon_path and mtime == self._lexicon_mtime:
            return len(self.entity_lexicon)
        self.entity_lexicon_path = path
        self.entity_lexicon, self.entity_lexicon_aliases = self._read_lexicon(path)
        self._alias_reverse = None
        self._lexicon_mtime = mtime
        return len(self.entity_lexicon)




    def canonical_entity(self, name: str) -> str:
        """把实体名归一到规范名（词库别名行声明的 ``别名 => 主名``）。

        未登记为别名的原样返回。**抽取侧与查询侧都走这里**——只归一一边等于
        没归一：库里挂的是主名，查"别名"照样吃不到实体加成。
        """
        n = (name or "").strip()
        if not n:
            return n
        return self.entity_lexicon_aliases.get(n.lower(), n)




    def _canonicalize_map(self, mapping: dict[str, str]) -> dict[str, str]:
        """把 ``{实体名: origin}`` 里的别名折到规范名，同键取更高可信度来源。"""
        if not getattr(self, "entity_lexicon_aliases", None):
            return mapping
        merged: dict[str, str] = {}
        disp_of: dict[str, str] = {}
        for name, origin in mapping.items():
            canon = self.canonical_entity(name)
            k = canon.lower()
            cur = disp_of.get(k)
            if cur is None:
                disp_of[k] = canon
                merged[canon] = origin
            elif _ORIGIN_RANK[origin] > _ORIGIN_RANK[merged[cur]]:
                merged[cur] = origin
        return merged




    def _trusted_entity_ids(self) -> set[str]:
        """受信任实体的 id 集合。

        每次现算（E 量级约千余，不到 1ms）。不缓存是为了避免"实体表被
        增删改后缓存失效"这类难以发现的陈旧 bug——这里不是热路径。
        """
        return {e["id"] for e in self.entities.values() if self._entity_trusted(e)}



    def entity_origin_stats(self) -> dict[str, int]:
        """按来源统计实体数，用于观察抽取质量（受信任 vs 候选）。"""
        out: dict[str, int] = {}
        for ent in self.entities.values():
            o = self._entity_origin(ent)
            out[o] = out.get(o, 0) + 1
        return out



    def _recompute_entity_stats(self, keys: list[str] | None = None) -> int:
        """重算实体统计（mention_count / first_seen / last_seen）。

        以「当前存活的记忆」为准：已被删除的记忆 id 会从 ``memory_ids``
        里摘除。删除、作废记忆后必须调用，否则倒排索引会留下悬挂引用。

        Args:
            keys: 要重算的实体键列表；None 表示全量重算（加载旧库时自愈用）。

        Returns:
            被摘除的悬挂引用总数。
        """
        items = (
            list(self.entities.items()) if keys is None
            else [(k, self.entities[k]) for k in keys if k in self.entities]
        )
        dropped = 0
        for _, ent in items:
            alive: list[tuple[str, float]] = []
            for mid in ent.get("memory_ids") or []:
                rec = self.manager.store.get(mid)
                if rec is None:
                    dropped += 1
                    continue
                alive.append((mid, rec.timestamp))
            ent["memory_ids"] = [m for m, _ in alive]
            ent["mention_count"] = len(alive)
            if alive:
                stamps = [t for _, t in alive]
                ent["first_seen"] = min(stamps)
                ent["last_seen"] = max(stamps)
        return dropped




    def _task_entity_ids(self, text: str) -> set[str]:
        """文本里**受信任**实体的 id 集合（钉住排序 / 相关性判定共用）。"""
        if not (text or "").strip():
            return set()
        try:
            origins = self.extract_entities_with_origin(text)
        except Exception:  # noqa: BLE001 —— 抽取失败不该让 session_start 失败
            return set()
        out: set[str] = set()
        for name, origin in origins.items():
            if origin not in TRUSTED_ENTITY_ORIGINS:
                continue
            ent = self._resolve_entity(name)
            if ent is not None:
                out.add(ent["id"])
        return out




    def _alias_siblings(self, canon_key: str) -> list[str]:
        """指向该规范名的**别名**实体 key（小写）。惰性建反向索引，reload 时失效。"""
        if not self.entity_lexicon_aliases:
            return []
        rev = getattr(self, "_alias_reverse", None)
        if rev is None:
            rev = {}
            for alias, canon in self.entity_lexicon_aliases.items():
                rev.setdefault(canon.lower(), []).append(alias)
            self._alias_reverse = rev
        return rev.get((canon_key or "").lower(), [])



    def _alias_family(self, key: str) -> list[str]:
        """该实体键**所属的别名同族**（含自身、规范名、全部兄弟别名）。

        ``key`` 可能本身就是别名（历史数据里"子安"独立建过档），此时也要能
        找到它的规范名与兄弟——所以先查别名映射得到规范名，再展开。
        """
        k = (key or "").lower()
        if not k:
            return []
        canon = (self.entity_lexicon_aliases.get(k) or k).lower()
        fam = [canon]
        for a in self._alias_siblings(canon):
            if a not in fam:
                fam.append(a)
        if k not in fam:
            fam.append(k)
        return fam



    def _resolve_entity(self, entity_key: str) -> dict[str, Any] | None:
        """按名字（含别名）找实体条目。

        优先级：规范名实体 → 同族里**提及最多**的那个（历史数据里可能只有
        别名建过档，此时不该报"没有这个实体"）。
        """
        canon = self.canonical_entity(entity_key or "")
        ent = self.entities.get(canon.lower())
        if ent is not None:
            return ent
        cands = [self.entities[k] for k in self._alias_family(entity_key)
                 if k in self.entities]
        if not cands:
            return None
        return max(cands, key=lambda e: e.get("mention_count", 0))



    def _entity_memory_ids(self, ent: dict[str, Any]) -> list[str]:
        """实体**连同其别名同族**的记忆 id（去重保序）。

        读取侧立刻合并，而不是等 ``relink_entities`` 重挂：历史数据里同一个
        实体往往已经按别名拆成了好几份档案，只认规范名会让"刚配好别名"的人
        看到一份残缺档案。词库是配置，配置生效不该依赖一次数据重写。
        """
        ids = list(ent.get("memory_ids", []))
        seen = set(ids)
        for k in self._alias_family(ent.get("key", "")):
            if k == ent.get("key"):
                continue
            s = self.entities.get(k)
            if not s:
                continue
            for mid in s.get("memory_ids", []):
                if mid not in seen:
                    seen.add(mid)
                    ids.append(mid)
        return ids



    def entity_facts(self, entity_key: str, limit: int = 5) -> list[str]:
        """取某实体的活跃事实列表（按时间倒序）。别名会被归一到规范名。"""
        ent = self._resolve_entity(entity_key)
        if not ent:
            return []
        facts = []
        for mid in self._entity_memory_ids(ent):
            rec = self.manager.store.get(mid)
            if rec and rec.state == MemoryState.ACTIVE.value:
                facts.append((rec.timestamp, rec.content))
        facts.sort(key=lambda x: -x[0])
        return [c for _, c in facts[:limit]]



    # ==================================================================
    # 实体档案 / 词库扩充闭环（用层需求）
    # ==================================================================
    def entity_profile(
        self,
        entity_key: str,
        limit: int = 20,
        order: str = "asc",
        include_superseded: bool = False,
    ) -> dict[str, Any]:
        """取某实体的完整档案：把它参与过的记忆**按时间拉一遍**。

        与 :meth:`entity_facts` 的区别：后者只返回内容字符串（给上下文装配用），
        本方法返回带 ``timestamp`` / ``source`` / ``memory_id`` / ``state`` 的
        结构化条目——宿主写人物戏前要"把某人的全部定案按时间拉一遍"时，
        需要能按时间核、能回指原文件、能看出哪条已被作废。

        Args:
            order: ``"asc"``（默认，时间正序，适合拉时间线）/ ``"desc"``（最近的在前）。
            include_superseded: 是否带上已作废条目（默认只看 active）。
            limit: 最多返回条数；传 0 表示不限。
        """
        canon = self.canonical_entity(entity_key or "")
        ent = self._resolve_entity(entity_key)
        if not ent:
            return {"found": False, "entity": entity_key, "facts": [], "total": 0}
        rows: list[dict[str, Any]] = []
        for mid in self._entity_memory_ids(ent):
            rec = self.manager.store.get(mid)
            if rec is None:
                continue
            if not include_superseded and rec.state != MemoryState.ACTIVE.value:
                continue
            rows.append({
                "memory_id": rec.id,
                "content": rec.content,
                "timestamp": rec.timestamp,
                "source": rec.source or "",
                "state": rec.state,
                "importance": rec.importance,
                "superseded_by": rec.superseded_by,
            })
        rows.sort(key=lambda r: r["timestamp"], reverse=(order == "desc"))
        kept = rows[:limit] if limit > 0 else rows
        out = {
            "found": True,
            "entity": ent["name"],
            "key": ent["key"],
            "origin": self._entity_origin(ent),
            "trusted": self._entity_trusted(ent),
            "mention_count": ent.get("mention_count", 0),
            "total": len(rows),
            "count": len(kept),
            "facts": kept,
        }
        # 别名同族：合并了几个别名档案、查的是从哪个名字折过来的——
        # 不显式说出来，宿主会以为"档案只有这些"（其实另一份挂在别名下）。
        sibs = [k for k in self._alias_family(ent.get("key", ""))
                if k != ent.get("key")]
        if sibs:
            out["aliases"] = sibs
            out["mention_count_merged"] = len(self._entity_memory_ids(ent))
        if canon != (entity_key or ""):
            out["resolved_from"] = entity_key
        return out



    def entity_list(
        self,
        prefix: str = "",
        limit: int = 50,
        min_mentions: int = 1,
        trusted_only: bool = False,
    ) -> list[dict[str, Any]]:
        """列出实体（按提及条数降序），用于发现"库里到底有哪些实体 key"。

        没有这个入口时，宿主只能靠猜实体名去调 :meth:`entity_profile`——
        而实体名是抽取出来的，拼写往往和记忆里的写法不一致。
        """
        rows: list[dict[str, Any]] = []
        for ent in self.entities.values():
            if trusted_only and not self._entity_trusted(ent):
                continue
            if prefix and not ent["name"].lower().startswith(prefix.lower()):
                continue
            if ent.get("mention_count", 0) < min_mentions:
                continue
            rows.append({
                "name": ent["name"],
                "key": ent["key"],
                "origin": self._entity_origin(ent),
                "trusted": self._entity_trusted(ent),
                "mention_count": ent.get("mention_count", 0),
                "last_seen": ent.get("last_seen", 0.0),
            })
        rows.sort(key=lambda r: (-r["mention_count"], r["name"]))
        return rows[:limit] if limit > 0 else rows



    def lexicon_add(self, terms: list[str], relink: bool = True) -> dict[str, Any]:
        """向**当前生效的**词库追加专名并热加载。

        坑（实测）：词库不一定在记忆库目录里——宿主实际生效的路径是
        ``data/entity_lexicon_<项目名>.txt``（靠 env / 显式参数生效，记忆库目录里
        **没有** entity_lexicon.txt）。若写回 ``<persist_dir>/entity_lexicon.txt``，
        会凭空造出一个不生效的新文件：宿主以为加成功了，检索毫无变化。
        因此一律写回 ``self.entity_lexicon_path``（实际生效的那个）。

        Args:
            terms: 词条列表。**支持别名行**——``"子安 => 顾言"``（``->`` / ``→``
                等价）。别名会写进词库文件，抽取与查询两侧随之归一。这是
                "同一实体多种称法"的通用解，不限于人名（简称/全称/旧称/代号
                同理）。语法不完整的行**不写盘**，原样放进返回的 ``invalid``。
            relink: 追加后是否对全库重挂实体。**默认必须做**：否则新词只对
                之后写入的记忆生效，存量记忆仍然查不到，闭环断在这里。
                写了别名时还会顺带做一次 :meth:`merge_alias_entities`，
                把历史上按别名拆开的档案并回规范名。

        返回：``added`` 实际写盘的行 · ``skipped_duplicate`` · ``invalid`` ·
        ``alias_count`` 生效别名总数（宿主核对归一是否真的挂了）·
        ``lexicon_size`` / ``path`` · 可选的 ``relink`` / ``alias_merge``。
        """
        entries = list(dict.fromkeys(
            t.strip() for t in (terms or []) if t and t.strip()
        ))
        path = self.entity_lexicon_path
        if not entries:
            return {"added": [], "lexicon_size": len(self.entity_lexicon), "path": path}
        if not path:
            # 自愈：默认路径下文件不存在时 path 就是 None（构造时解析过一次）。
            # 这时应当**创建**默认文件，而不是报错——否则宿主陷入死锁：
            # 没有词库 → 本工具报错 → 建不了词库 → 只能重启 daemon。
            path = self._ensure_default_lexicon_path()
        if not path:
            _dir = getattr(self.manager, "persist_dir", None) or self.persist_dir
            _env = load_config().entity_lexicon or ""
            # 如实说清查过哪三条路——旧文案只说「persist_dir 未设」，而真因
            # 通常是「默认路径下的文件不存在」，把人往完全错误的地方引。
            raise ValueError(
                "cannot append to the lexicon: no writable lexicon path. Searched, in order -- "
                f"1) explicit entity_lexicon_path="
                f"{self._explicit_lexicon_path or 'unset'}; "
                f"2) env HIPPOCAMPUS_ENTITY_LEXICON={_env or 'unset'}; "
                f"3) default <memory library>/entity_lexicon.txt: persist_dir="
                f"{_dir or 'unset'} (creation failed or no permission)."
            )
        existing_terms, existing_aliases = self._read_lexicon(path)
        before = len(existing_terms)
        added: list[str] = []
        invalid: list[str] = []
        has_alias = False
        for e in entries:
            m = self._ALIAS_SEP.search(e)
            if not m:
                if e.lower() in existing_terms:
                    continue
                added.append(e)
                continue
            alias = e[:m.start()].strip().lstrip("#").strip()
            canon = e[m.end():].strip()
            if not alias or not canon:
                invalid.append(e)
                continue
            if alias.lower() == canon.lower():
                if alias.lower() not in existing_terms:
                    added.append(canon)
                continue
            if existing_aliases.get(alias.lower(), "").lower() == canon.lower():
                continue
            # 落盘统一用规范写法（`别名 => 主名`），避免文件里三种箭头混用
            added.append(f"{alias} => {canon}")
            has_alias = True
        if added:
            # 追加到末尾：保留原有注释与分组，不重写整个文件
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n" + "\n".join(added) + "\n")
        size = self.reload_lexicon()
        out: dict[str, Any] = {
            "added": added,
            "skipped_duplicate": len(entries) - len(added) - len(invalid),
            "lexicon_size_before": before,
            "lexicon_size": size,
            "alias_count": len(self.entity_lexicon_aliases),
            "path": path,
        }
        if invalid:
            # 不静默吞：写坏的别名行若当成功返回，宿主会以为归一已生效
            out["invalid"] = invalid
        if relink:
            out["relink"] = self.relink_entities()
        if has_alias:
            out["alias_merge"] = self.merge_alias_entities()
        return out




    def merge_alias_entities(self) -> dict[str, Any]:
        """把别名档案并进规范名档案（消歧后的库整理）。

        读取侧（:meth:`entity_profile` / :meth:`entity_facts`）本来就会合并别名
        同族，所以**不跑它查询也是对的**；跑它是为了把库真正理干净——否则
        ``entity_list`` 会长期并列"子安"与"顾言"两份档案，画像槽位、mention
        计数、实体加成也各算一份。

        动作：①别名条目的 ``memory_ids`` 并入规范名条目（并集，origin 取更高）；
        ②那些记忆的 ``entity_ids`` 把别名 id 换成规范名 id；③删掉别名条目；
        ④全量重算统计。规范名条目尚不存在时直接**改名**（id 与倒排都不动）。

        与 :meth:`forget` 的区别：这里不是删除语义，是**同一性归并**——
        记忆一条不少，只是不再挂在两个名字下。
        """
        if not self.entity_lexicon_aliases:
            return {"merged": 0, "renamed": 0, "remapped_memories": 0}
        merged = renamed = remapped = 0
        for alias, canon in list(self.entity_lexicon_aliases.items()):
            a = self.entities.get(alias)
            if not a:
                continue
            ck = canon.lower()
            c = self.entities.get(ck)
            if c is None:
                a["name"] = canon
                a["key"] = ck
                self.entities[ck] = a
                del self.entities[alias]
                renamed += 1
                continue
            if c is a:
                continue
            for mid in a.get("memory_ids") or []:
                rec = self.manager.store.get(mid)
                if rec is None:
                    continue
                old_ids = list(rec.entity_ids or [])
                new_ids: list[str] = []
                for e in old_ids:
                    v = c["id"] if e == a["id"] else e
                    if v not in new_ids:
                        new_ids.append(v)
                if new_ids != old_ids:
                    self.manager.store.update(mid, entity_ids=new_ids)
                    remapped += 1
                if mid not in c["memory_ids"]:
                    c["memory_ids"].append(mid)
            # 来源只升不降：别名条目若被词库确认过，规范名也该拿到这份信任
            if _ORIGIN_RANK.get(a.get("origin", ENTITY_ORIGIN_HEURISTIC), 0) > \
                    _ORIGIN_RANK.get(c.get("origin", ENTITY_ORIGIN_HEURISTIC), 0):
                c["origin"] = a["origin"]
            del self.entities[alias]
            merged += 1
        if merged or renamed:
            self._recompute_entity_stats()
            self._alias_reverse = None
        return {"merged": merged, "renamed": renamed,
                "remapped_memories": remapped}




    def _alias_direction(
        self,
        a: str,
        b: str,
        in_lexicon: set[str],
        docs: dict[str, int],
    ) -> tuple[str, str, str]:
        """判断这一对里"谁是别名、谁是规范名"，返回 ``(alias, canonical, 依据)``。

        规则按可信度排序，**每一档都能说清楚理由**（依据会出现在候选里，
        因为方向猜错等于把两份档案并错，必须让人一眼能复核）：

          1. 一方已在词库登记为别名 → 规范名取它指向的那个（人已经表过态）
          2. 一方在词库（正式词条）→ 它是规范名
          3. 提及更多的一方 → 规范名（建档 mention_count 与共现条数取大）
          4. 更长的一方 → 规范名（全称优先于简称）
          5. 仍平手 → 字典序，并在依据里注明"需人工确认方向"
        """
        ca = self.canonical_entity(a)
        cb = self.canonical_entity(b)
        a_is_alias = ca.lower() != a.lower()
        b_is_alias = cb.lower() != b.lower()
        if a_is_alias and not b_is_alias:
            return b, ca, f"{a} is already registered in the lexicon as an alias of {ca}"
        if b_is_alias and not a_is_alias:
            return a, cb, f"{b} is already registered in the lexicon as an alias of {cb}"
        a_lex, b_lex = a in in_lexicon, b in in_lexicon
        if a_lex != b_lex:
            return (b, a, "registered in lexicon") if a_lex else (a, b, "registered in lexicon")
        na = max(docs.get(a, 0), self.entities.get(a, {}).get("mention_count", 0))
        nb = max(docs.get(b, 0), self.entities.get(b, {}).get("mention_count", 0))
        if na != nb:
            return ((b, a, f"mentioned more often ({na} > {nb})") if na > nb
                    else (a, b, f"mentioned more often ({nb} > {na})"))
        if len(a) != len(b):
            return ((b, a, f"longer form wins (full name over abbreviation, {len(a)}/{len(b)} chars)")
                    if len(a) > len(b)
                    else (a, b, f"longer form wins (full name over abbreviation, {len(b)}/{len(a)} chars)"))
        if a <= b:
            return b, a, "cannot decide; ordered lexicographically -- confirm the direction manually"
        return a, b, "cannot decide; ordered lexicographically -- confirm the direction manually"



    def discover_aliases(
        self,
        top_n: int = 30,
        min_confidence: float = 0.35,
        min_support: int = 2,
        min_lift: float = 2.0,
        min_containment: float = 0.5,
        allow_stat_only: bool = False,
        max_evidence: int = 3,
        max_names_per_record: int = 12,
        exclude: list[str] | None = None,
        cue_words: list[str] | None = None,
        with_paren: bool = True,
        refine: bool = True,
        max_probe_names: int = 200,
    ) -> dict[str, Any]:
        """挖"同一实体的多种称法"候选。**只出候选 + 证据，不写盘。**

        与 :meth:`lexicon_suggest` 的分工：那个是"这个词还没登记，要不要登记
        成专名"；这个是"这两个名字指同一个东西吧"——问的是**关系**，不是词。

        为什么要自动发现：归一的**机制**（``lexicon_add("别名 => 主名")``）早就
        有了，缺的是**发现**——同一个实体在语料里换了三种写法，人得先读到才会
        想到去登记。这一步把"读全库找称法"的活从人手里拿回来。

        为什么只出候选：别名一旦合并就**改变实体身份**，猜错等于把两个不同
        对象的记忆混成一档且事后难拆。所以这里绝不自动写盘，只给
        ``paste``（可直接粘给 ``lexicon_add`` 的一行）与 ``evidence``（原文）。

        信号与判据见 :mod:`hippocampus_mcp.domain.aliases`；五条硬门控：
          * **形状**：两侧都得是名字形状（纯汉字 2~12 / 纯拉丁），
            且不能是纯数字词（"第三""一百二十"）；
          * **噪声**：两侧都不能命中停用词三层（与 ``lexicon_suggest`` 同源）
            或"角色/关系词"表（``作者/读者/主角/用户…``）；
          * **提示词的位置**：分隔符必须在提示词**之前**（"甲，字乙"）。
            提示词多是常用字（字/号/名/称），"句号""编号""简称"里的那个字
            不是称谓——真实库实测这条是**噪声的大头**：放宽成"任一侧有分隔符"
            时，「碎句号、长段落」「全量换号、活跃文档」「同章全称后简称」
            这类假命中占了候选的一半以上（25 条里 16 条）。丢弃数见
            ``skipped.cue_context`` / ``cue_sep_after_only``。
          * **共现必须有锚**：纯共现对（``cooc``）要求至少一侧已在词库登记，
            或两侧存在包含关系（缩写/全称）——否则"两本书名老一起出现"也会
            进候选。
            （注：**没有**把这条推广到模式类。实测过，不成立——见下。）
          * **统计信号不能单独成立**（``allow_stat_only=False``，默认）：
            ``cooc`` 与 ``abbrev`` 只用来**加强**模式信号。真实库实测
            ``台湾外记 => 台湾`` / ``剡溪雪 => 剡溪`` / ``听女道士卞玉京弹琴歌 =>
            卞玉京`` 这类对，包含度 100%、提升度几百，全都在"共现"上完美成立，
            但它们不是同一个东西——**标题里出现某人的名字，不等于标题是那个
            人的别名**。子串 + 共现推不出"同指"，这是本轮实测最重要的结论。

        为什么**不**给模式类也加"至少一端在库内"的锚（试过，被数据否掉）：
        真实库 50 条候选里 44 条是"双端都没进实体索引"的，而人工判为真的 9 条里
        有 8 条**也是单端未知**（"苏文 => 阿文""明远 => 于山""云锦 => 俞锦"
        全都不在实体索引里）——锚挡噪声的同时会把真候选一起删掉。
        能分开真假的是**提示词的位置**，不是"在不在库里"。

        Args:
            top_n: 最多返回多少个候选，默认 30（按置信度降序）。
            min_confidence: 置信度门槛，默认 0.35。提示词（0.50）单独就能过；
                括号同位语（0.30）**单独过不了**，需要多几条证据（支持度加成）
                或再有别的信号——真实库实测"（一手）""（作者）"这类注释性括号
                太多，让括号单独成立会拉低整张表的可用性。
            allow_stat_only: 是否允许**只靠统计信号**（共现 / 包含）成候选，
                默认 False。见下文"三条硬门控"里的说明——子串 + 共现推不出"同指"。
            min_support: 纯共现对的最少共同出现条数，默认 2。
            min_lift: 纯共现对的提升度门槛，默认 2.0（独立时 ≈1）。
            min_containment: 纯共现对的包含度门槛，默认 0.5 ——
                ``共同出现条数 / 两侧较少的那一方的条数``，即"提到少的那一方时，
                有多大概率另一方也在场"。**与 min_lift 是并列关系，满足其一即可**：
                提升度在名字很少的小语料上天然趋近 1，只看它会把真候选整批漏掉。
            max_evidence: 每个候选最多带几条原文证据，默认 3。
            max_names_per_record: 单条记忆最多取几个名字参与两两配对，默认 12
                —— 成本闸门，防一条"实体清单"式长文本把挖掘拖垮。
            exclude: 本次临时排除词（不落盘），与 ``lexicon_suggest`` 同义。
            cue_words: 本次额外补充的称谓提示词（不落盘）。
            with_paren: 是否扫括号同位语，默认 True。
            refine: 是否做**第二轮全库补扫**，默认 True。未登记的别名在第一轮
                只在那条被模式抓到的记忆里"被认识"，支持度会被严重低估；
                第二轮拿第一轮的候选名（通常几十个）全库精确扫一遍，把
                "在多条记忆里都出现"这件事统计出来。大库上这是主要的耗时项，
                想快就关掉（会明显掉召回）。
            max_probe_names: 第二轮最多补扫多少个候选名，**成本闸门**（默认 200）。

        Returns:
            ``candidates``（alias/canonical/confidence/**band**/signals/support/
            lift/direction_basis/paste/evidence）· ``scanned`` · ``cue_words`` ·
            ``stop_words`` · ``skipped``（各门控挡掉多少，便于判断是不是阈值
            设错了）· ``note``。
        """
        in_lexicon = {t.lower() for t in self.entity_lexicon}
        trusted_keys = {
            k.lower() for k, e in self.entities.items() if self._entity_trusted(e)
        }
        # 已知词只用于**切分辅助**（从"此人是顾言，字子安"里切出"顾言"），
        # 不参与对错判断——所以宁可取得干净些，别把启发式噪声塞进来。
        known = in_lexicon | trusted_keys
        name_disp = {k.lower(): e.get("name", k) for k, e in self.entities.items()}
        lex_disp = {t.lower(): t for t in self.entity_lexicon}

        cue_path = None
        if self.entity_lexicon_path:
            cue_path = os.path.join(
                os.path.dirname(self.entity_lexicon_path), alias_mod.ALIAS_CUE_FILENAME
            )
        cue_set, cue_stat = alias_mod.load_cue_words(cue_path, cue_words)
        stop, stop_grams, stop_stat = self._stop_grams(exclude)
        stop_list = sorted(stop)

        agg: dict[tuple[str, str], dict[str, Any]] = {}

        def _touch(ka: str, kb: str, a: str | None = None, b: str | None = None):
            key = (ka, kb) if ka < kb else (kb, ka)
            ent = agg.get(key)
            if ent is None:
                ent = {"signals": set(), "cues": set(), "evidence": [],
                       "support": 0, "rids": set(), "lift": 0.0, "disp": {}}
                agg[key] = ent
            ent["disp"].setdefault(ka, a or name_disp.get(ka) or lex_disp.get(ka) or ka)
            ent["disp"].setdefault(kb, b or name_disp.get(kb) or lex_disp.get(kb) or kb)
            return ent

        # ---- 扫库：模式类信号 + 共现用的名字集合 --------------------------
        # 两条是**同一份定义**：断言不同会走偏。
        #   形状（name_like + 不是角色/关系词 + 不撞停用词）
        #   —— 只用来决定"这个名字值不值得拿去统计"，不影响最终候选判定。
        def _probe_ok(w: str) -> bool:
            if not alias_mod.name_like(w):
                return False
            if alias_mod.is_alias_noise(w):
                return False
            return not (w in stop_grams or any(s and s in w for s in stop_list))

        id_to_key = {e["id"]: e["key"] for e in self.entities.values()}
        n_records = 0
        n_pattern_records = 0
        probe: set[str] = set()
        pass1: list[tuple[str, set[str]]] = []
        ctx_stat: dict[str, int] = {}
        for rec in self.manager.store.iter_records():
            if rec.state == MemoryState.ARCHIVED.value:
                continue
            n_records += 1
            # 名字集合 = 已建档且受信任的实体 ∪ **本条里模式挖到的名字**。
            # 后者很关键：模式只证明"这一条里它们并列"，共现才能证明
            # "它们在很多条里总是一起出现"——同一对候选两条信号叠加时，
            # 置信度与支持度才反映得出真实强度。
            names = {
                id_to_key[eid] for eid in (rec.entity_ids or [])
                if id_to_key.get(eid) in trusted_keys
            }
            text = self._preprocess_for_entities(rec.content or "")
            hits = (
                alias_mod.mine_patterns(text, cue_set, known,
                                        with_paren=with_paren, stats=ctx_stat)
                if text.strip() else []
            )
            if hits:
                n_pattern_records += 1
            rec_probe: set[str] = set()
            for p in hits:
                ka, kb = p.a.lower(), p.b.lower()
                if ka == kb:
                    continue
                names.add(ka)
                names.add(kb)
                if p.signal == "cue":
                    # **只有提示词信号的名字进补扫候选**。理由：提示词是一句
                    # 明确的"这是同一个东西的另一种叫法"（"甲，字乙"），而括号
                    # 只是并列——"《明季南略》（一手）"里的"一手"是注释不是称法。
                    # 拿括号产物去全库补扫，等于把一个误判**放大成几十条支持度**。
                    rec_probe.add(ka)
                    rec_probe.add(kb)
                ent = _touch(ka, kb, p.a, p.b)
                ent["signals"].add(p.signal)
                if p.cue:
                    ent["cues"].add(p.cue)
                if rec.id not in ent["rids"]:
                    ent["rids"].add(rec.id)
                    ent["support"] += 1
                if len(ent["evidence"]) < max_evidence:
                    ent["evidence"].append({
                        "memory_id": rec.id, "signal": p.signal,
                        "cue": p.cue, "text": p.snippet,
                    })
            pass1.append((rec.id, names))
            probe |= {x for x in rec_probe if _probe_ok(x)}

        # ---- 第二轮：候选名很小，全库精确扫一遍求真实支持度 ---------------
        # 第 1 轮里，**未被识别的别名**（"子安"这种）只在那一条被模式抓到；
        # 它在别的记忆里出现时没人认得，支持度就被严重低估。候选名字集合很小
        # （通常几十个），所以值得再扫一遍库把它们认全——支持度是共现信号
        # 能说话的前提，不补这一轮，"在多条记忆里都成立"就永远看不到。
        cooc_input = pass1
        probe_list: list[str] = []
        if refine and probe:
            probe_list = sorted(probe)[:max_probe_names]
            cooc_input = []
            for rec in self.manager.store.iter_records():
                if rec.state == MemoryState.ARCHIVED.value:
                    continue
                low = (rec.content or "").lower()
                names = {
                    id_to_key[eid] for eid in (rec.entity_ids or [])
                    if id_to_key.get(eid) in trusted_keys
                }
                names |= {n for n in probe_list if n in low}
                cooc_input.append((rec.id, names))

        # ---- 共现 + 提升度 ------------------------------------------------
        pair_counts, docs, n_seen = alias_mod.cooccurrence(
            cooc_input, max_names_per_record=max_names_per_record
        )
        n_cooc_kept = 0
        for (ka, kb), cnt in pair_counts.items():
            if cnt < min_support:
                continue
            da, db = docs.get(ka, 0), docs.get(kb, 0)
            lf = alias_mod.lift_of(cnt, da, db, n_seen)
            # 两条**并列**判据，满足其一即可：
            #   ① 提升度（超过独立假设）——适合"简称与全称一起出现"这类；
            #   ② 包含度 count_ab / min(count_a, count_b) ——"只要提到 a 就总会
            #      提到 b"。小语料上提升度天然趋近 1（库里就这么几个名字），
            #      只看提升度会把真候选整批漏掉。
            contain = cnt / min(da, db) if min(da, db) > 0 else 0.0
            if lf < min_lift and contain < min_containment:
                continue
            rel = alias_mod.abbrev_relation(ka, kb)
            if ka not in in_lexicon and kb not in in_lexicon and not rel:
                # 纯共现必须有锚：否则"两本书名总一起出现"也会进候选
                continue
            n_cooc_kept += 1
            ent = _touch(ka, kb)
            ent["signals"].add("cooc")
            ent["lift"] = max(ent["lift"], lf)
            ent["support"] = max(ent["support"], cnt)
            if rel:
                ent["signals"].add("abbrev")
            if len(ent["evidence"]) < max_evidence:
                ent["evidence"].append({
                    "memory_id": "", "signal": "cooc", "cue": "",
                    "text": f"co-occur {cnt} times (lift {lf:.1f}x / containment {contain:.0%}）",
                })

        # ---- 门控 → 定方向 → 打分 → 排序 ---------------------------------
        skipped = {"declared": 0, "shape": 0, "noise": 0, "stop": 0,
                   "stat_only": 0, "lowconf": 0,
                   # 提示词前面没有分隔符而丢弃的命中数；其中"分隔符只在提示词
                   # 之后"的那个数是**假命中的典型形态**（句号/编号/简称），
                   # 单独露出来，好判断这条门控是不是在误杀。
                   "cue_context": ctx_stat.get("cue_no_sep_before", 0),
                   "cue_sep_after_only": ctx_stat.get("cue_sep_after_only", 0)}
        # 统计信号（共现 / 包含）**只能加强模式信号，不能单独成候选**。
        # 真实库实测：`台湾外记 => 台湾`、`剡溪雪 => 剡溪`、`听女道士卞玉京弹琴歌 =>
        # 卞玉京` 这类"书名/标题包含人名/地名"的对，包含关系与共现都完美成立
        # （包含度 100%、提升度几百），但它们**不是同一个东西**——标题里出现
        # 某人的名字，不代表标题就是那个人的别名。子串+共现推不出"同指"。
        _STAT_SIGNALS = {"cooc", "abbrev"}

        def _bad_stop(w: str) -> bool:
            if w in stop_grams:
                return True
            return any(s and s in w for s in stop_list)

        candidates: list[dict[str, Any]] = []
        for (ka, kb), ent in agg.items():
            if not (alias_mod.name_like(ka) and alias_mod.name_like(kb)):
                skipped["shape"] += 1
                continue
            if alias_mod.is_alias_noise(ka) or alias_mod.is_alias_noise(kb):
                skipped["noise"] += 1
                continue
            if _bad_stop(ka) or _bad_stop(kb):
                skipped["stop"] += 1
                continue
            # 已声明为同一族（两边都归到同一个规范名）→ 本来就是归一好的
            if self.canonical_entity(ka).lower() == self.canonical_entity(kb).lower():
                skipped["declared"] += 1
                continue
            if not allow_stat_only and ent["signals"] <= _STAT_SIGNALS:
                skipped["stat_only"] += 1
                continue
            conf = alias_mod.confidence_of(ent["signals"], ent["support"], ent["lift"])
            if conf < min_confidence:
                skipped["lowconf"] += 1
                continue
            alias, canon, basis = self._alias_direction(ka, kb, in_lexicon, docs)
            disp = ent["disp"]
            a_disp = disp.get(alias.lower(), alias)
            c_disp = disp.get(canon.lower(), canon)
            candidates.append({
                "alias": a_disp,
                "canonical": c_disp,
                "confidence": conf,
                # 三档，给调用方一个**不用自己重算**的分流依据：
                #   high   ≥0.60 多信号（或高支持）——值得直接复核；
                #   medium 0.45~0.60 单提示词成立——**必须逐条看 evidence**；
                #   low    <0.45 仅括号/统计——基本是注释性并列。
                # 为什么要这一档：真实库上单字提示词（字/号/名）歧义大，
                # "明远 => 于山""云锦 => 俞锦" 与 "改为 => 同安""无事 => 自建"
                # 的置信度是同一个 0.52，**分数分不开**，只能靠人看证据。
                "band": ("high" if conf >= 0.60 else
                         "medium" if conf >= 0.45 else "low"),
                "signals": sorted(ent["signals"]),
                "cue_words": sorted(ent["cues"]),
                "support": ent["support"],
                "lift": round(ent["lift"], 2),
                "direction_basis": basis,
                "paste": f"{a_disp} => {c_disp}",
                "evidence": ent["evidence"][:max_evidence],
            })
        candidates.sort(key=lambda c: (-c["confidence"], -c["support"], c["alias"]))
        return {
            "candidates": candidates[:top_n] if top_n > 0 else candidates,
            "scanned": {
                "records": n_records,
                "records_with_pattern": n_pattern_records,
                "records_with_names": n_seen,
                "pairs": len(agg),
                "cooc_pairs_kept": n_cooc_kept,
                # 第二轮补扫用了哪些名字（只来自**提示词**信号，见上）。
                # 暴露出来是为了让"括号产物有没有被拿去做全库放大"可核对。
                "probe_names": probe_list,
            },
            "cue_words": cue_stat,
            "stop_words": stop_stat,
            "skipped": skipped,
            "note": "candidates only, nothing written automatically; after confirming the direction, persist via lexicon_add(paste)",
        }



    def lexicon_suggest(
        self,
        top_n: int = 30,
        min_docs: int = 4,
        min_files: int = 2,
        min_len: int = 3,
        max_len: int = 4,
        exclude: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """列出"高频出现但没登记进词库"的候选专名，供人工过筛后追加。

        **不用启发式抽取做信号源**（实测否决）：
        中文没有分词与 NER，``_heuristic_entities`` 靠引号/句式正则，在真实库
        9359 条上只产出 34 个种类、最高频次 2，且全是"活人的制""人成本逐"这类
        被切碎的片段——它不是专名发现器，拿它做候选必然交付一堆垃圾。

        改用**中文 n-gram 频次统计**（零依赖的新词发现最简形态）：
          1. 按 2~4 字滑窗统计"出现在多少条记忆里"（同一条内去重，避免一条短文
             把某个词刷高）与"跨几个源文件"；
          2. 剔除含噪声字的、**命中停用词的**（内置通用行话 ∪ 项目文件 ∪ exclude）、
             已在词库的、已是受信任实体的；
          3. **碎片抑制**：若某短候选被更长的候选以 >=90% 的频次包含
             （如"活人"总出现在"活人的制"里），判为切片碎片，丢弃。

        Args:
            min_docs: 至少在多少条记忆里出现过，默认 4（主力门槛）。
            min_files: 至少跨几个源文件，默认 2。**只在单个文件里反复出现的
                多半是该文件的小节标题 / 术语**，不是要登记的专名。
            min_len / max_len: 候选长度，默认 3~4。默认不从 2 字起——
                实测 2 字候选几乎被"主角/用户/正文/定案/第一次"这类通用词占满。
            exclude: 临时排除词（一次性排雷，不落盘）。**停用词共三层，逐层增补**：
                ① **内置通用行话**（`_DEFAULT_LEXICON_STOP`，代码里，开箱即用）——
                   "本章/正文/时间线/待核清单/用户定案/第一次"这类跨项目套话，
                   外加一条"纯中文数字+量词"规则（第五节/十二年）；
                ② **项目停用词文件** `<词库同目录>/entity_lexicon_stop.txt`
                   （一行一个，# 注释）——放**项目特有**的结构标签/行话
                   （如"感情线/伏笔表/制度篇"）。内置层只管通用的，项目词放这里；
                ③ 本参数 `exclude`——只想这一次排除的词。
                匹配时会**连带压掉已登记词的切片**（登记"用户定案"即可同时挡住
                "户定案/页定案"这类 3 字碎片），所以只需写完整短语，
                不必再手工枚举前缀碎片。
        """
        in_lexicon = {t.lower() for t in self.entity_lexicon}
        trusted_keys = {
            k.lower() for k, e in self.entities.items() if self._entity_trusted(e)
        }
        # 停用词三层（只做增补，后者不覆盖前者）：
        #   ① **内置**通用行话 `_DEFAULT_LEXICON_STOP`——开箱即用，不需要任何配置。
        #      这就是 D4 之前"噪声高"的根因：老版本只有 ② 和 ③，新项目不写文件
        #      就必然被"用户定案/时间线/第一次"这种词霸榜。
        #   ② 项目停用词文件 `<词库同目录>/entity_lexicon_stop.txt`
        #   ③ 本次调用的 `exclude` 参数（临时排除，不落盘）
        stop, stop_grams, stop_stat = self._stop_grams(exclude)
        logger.info(
            "[lexicon_suggest] %d stopwords (built-in %d / project file %d / excluded this run %d)",
            len(stop), stop_stat["builtin"], stop_stat["file"], stop_stat["extra"],
        )
        noise = tuple(self._NOISE_CHARS)
        # 多统计一层（不输出）：4 字候选要靠 5 字频次才能识别它是不是碎片
        # ——否则"天工开"/"工开物"会各自以 48 条混进候选，而它们只是"天工开物"的切片。
        stat_max = max_len + 1
        docs: dict[str, int] = {}
        # 全量计数（**不做**词库/受信任实体过滤），专供碎片抑制使用。
        # 少了它就会出现这种怪事："天工开物"已在词库里 → 被排除 → 没人能证明
        # "天工开""工开物"只是它的切片 → 两个碎片各自以 48 条混进候选列表。
        raw: dict[str, int] = {}
        files: dict[str, set[str]] = {}
        for rec in self.manager.store.iter_records():
            if rec.state == MemoryState.ARCHIVED.value:
                continue
            src = rec.source or ""
            seen: set[str] = set()
            for run in re.findall(r"[\u4e00-\u9fa5]{2,}", rec.content):
                for n in range(min_len, stat_max + 1):
                    if len(run) < n:
                        continue
                    for i in range(len(run) - n + 1):
                        g = run[i : i + n]
                        if g in seen:
                            continue
                        if any(ch in g for ch in noise) or g.lower() in stop_grams:
                            continue
                        if g[0] in _ORDINAL_HEADS and _ORDINAL_RE.match(g):
                            continue
                        seen.add(g)
                        raw[g] = raw.get(g, 0) + 1
                        if g.lower() in in_lexicon or g.lower() in trusted_keys:
                            continue
                        docs[g] = docs.get(g, 0) + 1
                        if n <= max_len:
                            files.setdefault(g, set()).add(src)

        # 碎片抑制：短候选被更长候选"包住"就丢掉（只在候选集内部比较，不再扫库）
        pool = {t: c for t, c in docs.items() if c >= min_docs}
        # 碎片抑制：判据走 raw（已在词库里的长词也能压住自己的碎片）。
        #
        # 做法是"由长压短"而不是"对每个短候选找超串"——后者要拿几百个短候选
        # 去比几百万个 gram，实测 63 秒。反过来遍历长 gram、拆出它的子串，
        # 并**先把压不住任何东西的低频长 gram 剪掉**（候选的 docs 都 >= min_docs，
        # 抑制它至少需要 0.9*min_docs 的频次），实测回到秒级。
        short_set = {t for t in pool if len(t) < stat_max}
        floor = 0.9 * min_docs
        best: dict[str, int] = {}
        for g, c in raw.items():
            if c < floor or len(g) <= min_len:
                continue
            n = len(g)
            for m in range(min_len, n):
                for i in range(n - m + 1):
                    sub = g[i : i + m]
                    if sub in short_set and c > best.get(sub, 0):
                        best[sub] = c
        dropped = {s for s, c in best.items() if c >= 0.9 * raw.get(s, 0)}

        rows = [
            {
                "term": t,
                "docs": c,
                "files": len(files.get(t, ())),
                "sample_sources": sorted(files.get(t, ()))[:3],
            }
            for t, c in pool.items()
            if len(t) <= max_len and t not in dropped
            and len(files.get(t, ())) >= min_files
        ]
        rows.sort(key=lambda r: (-r["docs"], -r["files"], r["term"]))
        return rows[:top_n] if top_n > 0 else rows
