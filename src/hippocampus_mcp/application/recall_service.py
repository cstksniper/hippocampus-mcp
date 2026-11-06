# -*- coding: utf-8 -*-
"""RecallService：读侧用例（11 工具）。

工具面：recall / get_memory /
get_memory_context / list_memories / list_sources / contradictions /
drift_audit / knowledge_map / knowledge_topic / entity_profile /
entity_list。

本类的每个方法都是**从 server.py 工具体逐字搬来的编排逻辑**（去掉了
``_await_warmup`` / brain 获取 / ``json.dumps``）——只挪位置不改行为，
行为等值由全量回归门禁保证；后续脑方法体上收进本类时，本类 API
不动。

「谁解析参数」的切分（按**现行代码里解析与库解析的先后**
定，保字节级一致）：

* 全部 11 个工具的 JSON 解析都排在库解析**之后**（现行代码先
  ``_brain_for(library)`` 再解析），故解析随编排整体进服务——工厂调用
  即库解析，方法体里的解析自然晚于它，错误次序不变；
* 解析失败的错误体**不带** ``library`` 首键（recall / get_memory_context /
  contradictions 的 filter_tags、knowledge_topic 的 items/fras），服务内
  以裸 dict 返回，与现行工具逐位一致；
* knowledge_topic 的 ``evidence`` / ``topics`` 走 lenient 解析
  （``common.params.parse_str_list_lenient``，永不报错），随编排进服务。

信封（库自曝首键）**按工具逐个保留现行形态**，不做统一——本层是行为等值
迁移，"给读侧工具补 library 首键"是产品决策不是迁移副产物：

* **带**信封：knowledge_map / knowledge_topic（现行经 ``_with_library``）；
* **不带**：其余 9 个（现行直接 ``json.dumps``，含全部错误路径）。
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from ..common.params import (filter_applied, parse_str_list_lenient,
                             parse_tags_json)
from ..common.recency import iso_date, parse_date_bound
from ..domain import metacog as _metacog
from ..domain.context_slots import BLOCK_PLACEHOLDERS
from ..domain import prospective as _prospective
from ..domain.scope_recall import (SCOPE_ALL, SCOPE_PROJECT, SCOPE_USER,
                                   merge_multi_library, merge_scope_hits,
                                   parse_library_spec, validate_scope)
from ..domain.write_quarantine import build_access_event
from ..library import resolve_library, scan_libraries
from ..library.access_log import append_access_event, read_access_events
from ..library.task_directory import load_directory
from .base import BaseService
from .tags import check_tag_keys

logger = logging.getLogger("hippocampus_mcp")

#: D2b：none 档对 user 库的读拒绝文案（单一真相，多工具共用）。
USER_READ_DENIED = (
    "This workspace has none-level read access to the user library (not registered/bound; fail-closed)."
    "Register the task directory from a session in this workspace first (library_create with workspace_root), "
    "or operate from a home workspace (daily assistant host) session.")


class RecallService(BaseService):
    """读用例：检索/单条/列举/矛盾/漂移/实体/知识地图（策略与算法暂居 brain/manager，后续上收）。"""

    # ---------------------------------------------------------------- 语义检索

    def recall(self, *, query: str, top_k: int, use_lnn: bool, sample: bool,
               filter_tags: str, client: str,
               scope: str = SCOPE_PROJECT,
               libraries: str = "",
               occurred_after: str = "",
               occurred_before: str = "") -> dict[str, Any]:
        """向量 + BM25(RRF) + MMR + 实体加成 + 硬门槛（默认滤 superseded/archived）。

        ``client`` 由适配层传 ``session_key()``（读也要走本会话那一路
        LNN 状态，与 store_memory 同一笔接线）。

        scope 三值（枚举不泛化，有意收窄）：

        * ``project``（默认）：单库流，返回体不带 scope/libraries 键
          （保持最简形态）；
        * ``user``：整条流换到用户级库 brain 上跑（组合根注入），成功体加
          ``scope`` + ``libraries`` 自曝；留痕（miss/bump_daily）按意图归
          用户库，stamp_recalled 照盖；
        * ``all``：项目流全前、用户流殿后（硬分段，分数跨库不可比不融合），
          ``top_k`` 是**总数**；两库各自 stamp，miss/
          bump_daily 归项目库（查询语境是项目的）；跨库**精确内容**
          相同的用户级条目标 ``duplicate_of``（只标注不丢弃）；
          成功体加 ``scope`` + ``libraries``（顺序 = 实际打的库序）。

        错误体（scope 非法 / filter_tags 语法错 / 键名校验失败）一律是裸
        dict，**不带**任何自曝键。
        """
        err = validate_scope(scope)
        if err:
            return {"error": err, "input": scope}
        # B1 互斥（拍板②a）：libraries= 是显式多库通道，与双库语法糖
        # scope=user/all 组合没有可预期的语义 → 显式报错，不静默择一。
        spec_names: list[str] = []
        if (libraries or "").strip():
            if scope != SCOPE_PROJECT:
                return {"error": (
                    f"libraries= and scope={scope} are mutually exclusive: "
                    "scope=user/all is sugar for the project+user pair, while "
                    "libraries= is the explicit multi-library channel. Their "
                    "combination has no predictable semantics. Use only "
                    "libraries= for multi-library, or only scope for the paired "
                    "scopes."), "input": libraries}
            spec_names, spec_err = parse_library_spec(libraries)
            if spec_err:
                return {"error": spec_err, "input": libraries}
            if not spec_names:
                return {"error": f"libraries= contains only whitespace: {libraries!r}",
                        "input": libraries}
        parsed_tags, tag_err = parse_tags_json(filter_tags)
        if tag_err:
            # 注意：此错误体**不带** library 首键（与现行工具逐位一致）。
            return {"error": f"filter_tags parse failed: {tag_err}",
                    "input": filter_tags[:200]}
        # 时间窗解析（2026-11-05）：日期格式错**显式报错**，不静默当"不限"。
        # 静默丢参数的后果是「结果看起来像库里真没有」，而实际是条件没生效。
        try:
            win_after = parse_date_bound(occurred_after)
            win_before = parse_date_bound(occurred_before, end_of_day=True)
        except ValueError as e:
            return {"error": f"Invalid time-window argument: {e}",
                    "input": f"{occurred_after!r} / {occurred_before!r}"}
        if win_after > 0 and win_before > 0 and win_after > win_before:
            return {"error": (f"Inverted time-window bounds: occurred_after={occurred_after} "
                              f"is later than occurred_before={occurred_before}"),
                    "input": f"{occurred_after!r} / {occurred_before!r}"}
        tw = (win_after, win_before)
        if scope == SCOPE_PROJECT:
            if spec_names:
                return self._recall_multi(
                    query=query, top_k=top_k, use_lnn=use_lnn, sample=sample,
                    parsed_tags=parsed_tags, client=client, time_window=tw,
                    spec_names=spec_names)
            # D2b：目标 = user 库（library="user"）时按档位执行。
            tier = self._ua(client)
            if tier == "none":
                return {"error": USER_READ_DENIED, "input": scope}
            body = self._recall_one(
                self.brain, query=query, top_k=top_k, use_lnn=use_lnn,
                sample=sample, parsed_tags=parsed_tags, client=client, time_window=tw,
                record_miss=True, exclude_sensitive=(tier == "masked"))
            if tier:
                self._last_route_query = query
                self._log_route(client, scope="single",
                                probed=[self.library_self.get("name", "")],
                                hits={self.library_self.get("name", ""):
                                      int(body.get("count", 0))})
            return body
        user_brain = self.user_brain
        if user_brain is None:
            # 正常到不了这里：库缺失时工厂解析已抛 LibraryResolutionError
            # （显式检索找不到库必须报错）。防御性自曝，不静默回落 project。
            return {"error": "scope=user/all requires the user-level library "
                             "(conventional name user); it was not injected for "
                             "this call.", "input": scope}
        # D2b：用户库流的档位门（scope=user/all 打的就是用户库）。
        tier = self._ua(client)
        if scope == SCOPE_USER:
            if tier == "none":
                return {"error": USER_READ_DENIED, "input": scope}
            body = self._recall_one(
                user_brain, query=query, top_k=top_k, use_lnn=use_lnn,
                sample=sample, parsed_tags=parsed_tags, client=client, time_window=tw,
                record_miss=True, exclude_sensitive=(tier == "masked"))
            if "error" in body:
                return body
            user_name = (self.user_library_self or {}).get("name")
            body["memories"] = [dict(h, library=user_name)
                                for h in body.get("memories") or []]
            body["scope"] = SCOPE_USER
            body["libraries"] = [self.user_library_self]
            self._last_route_query = query
            self._log_route(client, scope=SCOPE_USER, probed=["user"],
                            hits={"user": int(body.get("count", 0))})
            return body
        # scope=all：项目流（全量留痕）+ 用户流（只 stamp，miss/日计数按
        # 查询意图归项目库）→ domain 纯函数硬分段合并。
        # D2b：档位门——none 降级 project-only 并自曝；masked 用户流滤敏感。
        tier = self._ua(client)
        proj_body = self._recall_one(
            self.brain, query=query, top_k=top_k, use_lnn=use_lnn,
            sample=sample, parsed_tags=parsed_tags, client=client, time_window=tw,
            record_miss=True)
        if "error" in proj_body:
            return proj_body
        if tier == "none":
            out = dict(proj_body)
            out["scope"] = SCOPE_ALL
            out["user_access"] = "none"
            out["libraries"] = [self.library_self]
            self._last_route_query = query
            self._log_route(client, scope=SCOPE_ALL,
                            probed=["project", "user(denied)"],
                            hits={"project": int(proj_body.get("count", 0)),
                                  "user": 0})
            return out
        user_body = self._recall_one(
            user_brain, query=query, top_k=top_k, use_lnn=use_lnn,
            sample=sample, parsed_tags=parsed_tags, client=client, time_window=tw,
            record_miss=False, exclude_sensitive=(tier == "masked"))
        if "error" in user_body:
            return user_body
        proj_name = self.library_self.get("name")
        user_name = (self.user_library_self or {}).get("name")
        proj_hits = [dict(h, library=proj_name)
                     for h in proj_body.get("memories") or []]
        user_hits = [dict(h, library=user_name)
                     for h in user_body.get("memories") or []]
        merged = merge_scope_hits(proj_hits, user_hits, top_k=top_k)
        out = {
            "count": len(merged),
            "memories": merged,
            # 回显标签域以项目库为准（查询语境；用户库各自过滤已体现在
            # 各条目的 tags 上——按库独立过滤）。
            "filter_applied": proj_body.get("filter_applied"),
            "scope": SCOPE_ALL,
            "libraries": [self.library_self, self.user_library_self],
        }
        if tier == "masked":
            out["filtered_sensitive"] = int(
                user_body.get("filtered_sensitive", 0))
        self._last_route_query = query
        self._log_route(client, scope=SCOPE_ALL,
                        probed=["project", "user"],
                        hits={"project": int(proj_body.get("count", 0)),
                              "user": int(user_body.get("count", 0))})
        return out

    def _log_route(self, client: str, *, scope: str, probed: list[str],
                   hits: dict[str, int]) -> None:
        """B3：路由事件入 user 库 access_log——「谁在什么时候跨库查了什么、
        各库命中几条」。查询原文只留指纹（build_access_event 口径）；
        审计是旁路，失败静默。"""
        lib = (self.user_library_self or {}).get("path")
        if not lib:
            return
        try:
            ev = build_access_event(
                kind="recall_route", tool="recall", home="route",
                target_name="user", target_path=lib, quarantined=False,
                content=str(getattr(self, "_last_route_query", "") or ""))
            ev["route"] = {"scope": scope, "probed": probed, "hits": hits}
            append_access_event(lib, ev)
        except Exception:  # noqa: BLE001 —— 审计旁路
            pass

    def _recall_multi(self, *, query: str, top_k: int, use_lnn: bool,
                      sample: bool, parsed_tags: dict[str, Any] | None,
                      client: str,
                      spec_names: list[str] | None = None,
                      time_window: tuple[float, float] = (0.0, 0.0)
                      ) -> dict[str, Any]:
        """B1 层次化多库（拍板②a）：家库段全前 + 外围库内 RRF 竞争剩余名额。

        家库段保留现行排序语义与留痕口径（record_miss=True）；外围各流
        record_miss=False（查询意图归家库，与 scope=all 的用户库流同款），
        每流各自过本库标签白名单（``_recall_one`` 逐库口径）。信封自曝
        ``mode`` + ``libraries``（实际打的库序）+ 目录外值 ``unregistered``。
        """
        tw = time_window
        home_body = self._recall_one(
            self.brain, query=query, top_k=top_k, use_lnn=use_lnn,
            sample=sample, parsed_tags=parsed_tags, client=client, time_window=tw,
            record_miss=True)
        if "error" in home_body:
            return home_body
        home_name = self.library_self.get("name")
        home_hits = [dict(h, library=home_name)
                     for h in home_body.get("memories") or []]
        periphery: list[tuple[str, list[dict[str, Any]]]] = []
        probed: list[dict[str, str]] = [self.library_self]
        tier = self._ua(client)
        for info, brain in self.extra_libraries:
            # D2b：外围流若含 user 库，档位同款执行——none 整调用报错
            # （显式点名了被拒的库），masked 该流滤敏感。
            if (info.get("name") or "").strip() == "user" and tier != "full":
                if tier == "none":
                    return {"error": USER_READ_DENIED,
                            "input": ",".join(spec_names or [])}
                body = self._recall_one(
                    brain, query=query, top_k=top_k, use_lnn=use_lnn,
                    sample=sample, parsed_tags=parsed_tags, client=client, time_window=tw,
                    record_miss=False, exclude_sensitive=True)
            else:
                body = self._recall_one(
                    brain, query=query, top_k=top_k, use_lnn=use_lnn,
                    sample=sample, parsed_tags=parsed_tags, client=client, time_window=tw,
                    record_miss=False)
            if "error" in body:
                return body
            name = info.get("name")
            periphery.append((name, [dict(h, library=name)
                                     for h in body.get("memories") or []]))
            probed.append(info)
        merged = merge_multi_library(home_hits, periphery, top_k=top_k,
                                     home_name=home_name)
        out: dict[str, Any] = {
            "count": len(merged),
            "memories": merged,
            # 回显标签域以家库为准（查询语境；各库过滤已体现在条目 tags）。
            "filter_applied": home_body.get("filter_applied"),
            "mode": "multi_library",
            "libraries": probed,
        }
        if self.unregistered_names:
            out["unregistered"] = list(self.unregistered_names)
        self._last_route_query = query
        # 路由事件只落 user 库被探测的调用（access_log 是 user 库的读审计）。
        probed_names = [i.get("name", "") for i in probed]
        if "user" in probed_names:
            self._log_route(client, scope="multi", probed=probed_names,
                            hits={n: len(h) for n, h in periphery})
        return out

    def _recall_one(self, brain: Any, *, query: str, top_k: int,
                    use_lnn: bool, sample: bool,
                    parsed_tags: dict[str, Any] | None, client: str,
                    record_miss: bool,
                    exclude_sensitive: bool = False,
                    time_window: tuple[float, float] = (0.0, 0.0)
                    ) -> dict[str, Any]:
        """单库检索流（recall 的单库执行体）。

        ``brain`` 参数化使 project 路径与用户库路径同一条流；
        ``record_miss=False`` 只给 scope=all 的用户库流用：stamp_recalled
        照盖（库内治理数据，两边都盖），miss/按日计数不记（查询
        意图归项目库）。``filter_tags`` 的 JSON 语法解析在调用方做一次，
        键名白名单校验**逐库**做（schema 是每库一份）。
        """
        mgr = brain.manager
        parsed_tags, key_info = check_tag_keys(brain, parsed_tags)
        if key_info and key_info.get("error"):
            return key_info
        allow_ids = brain._ids_matching_tags(parsed_tags)
        if allow_ids is not None and not allow_ids:
            # 过滤没命中任何条目——matched=0 让调用方当场看出"键名/值对不上"，
            # 而不是误读成"库里真没有"。
            return {"count": 0, "memories": [],
                    "filter_applied": filter_applied(parsed_tags, 0, key_info)}
        results = mgr.recall(
            query=query if query else None,
            top_k=top_k,
            use_lnn=use_lnn,
            sample=sample,
            allow_ids=allow_ids,
            client=client,
            occurred_after=time_window[0],
            occurred_before=time_window[1],
        )
        # RecallResult 只带检索侧字段，source/state 在本体记录上 —— 补回来。
        # 缺 source 时"这条记忆出自哪个文件"无法回查，与 get_memory_context 口径也不一致。
        # 这里按 id 回查代价是一次 dict 命中，不涉及重算。
        memories = []
        top_sim = 0.0
        quarantined_hidden = 0
        for r in results:
            d = r.to_dict()
            rec = mgr.store.get(r.memory_id)
            # D2a 隔离门（"不进召回池"在召回面的唯一执行点）：scope=all/user
            # 与裸 recall 全部走本执行体（mgr.recall 引擎路径），隔离排除
            # 必须在这里——brain.recall 的 collect_recall_items 管不到本路径
            #（E2E 实测抓到的漏点）。getattr 兜底旧记录。
            if rec is not None and getattr(rec, "quarantine", False):
                quarantined_hidden += 1
                continue
            if rec is not None:
                d["source"] = rec.source or ""
                d["state"] = rec.state
                d["cognitive_type"] = rec.cognitive_type
                d["tags"] = dict(rec.tags or {})
            top_sim = max(top_sim, float(r.similarity))
            memories.append(d)
        memories = self._cap_contents(memories)
        # D2b masked 档：sensitive:* 卡不出流（计数自曝，绝不静默）。
        filtered_sensitive = 0
        if exclude_sensitive:
            kept: list[dict[str, Any]] = []
            for d in memories:
                rec = mgr.store.get(d.get("memory_id"))
                if rec is not None and str(
                        getattr(rec, "sensitivity", "") or ""
                ).startswith("sensitive:"):
                    filtered_sensitive += 1
                    continue
                kept.append(d)
            memories = kept
        # 盖上"最后一次被召回"时间戳：本工具直连 manager.recall，不经过
        # brain.recall，得自己记——否则走这条路径的召回不算"被召回过"，
        # 沉寂归档只看得到 get_memory_context / session_start 那几条路。
        mgr.stamp_recalled([r.memory_id for r in results])
        if record_miss:
            # 未命中留痕 + 按日召回计数：这条路径不经过
            # brain.recall，不记的话走 MCP 工具的召回（宿主的**主要**检索
            # 入口）在 miss_log / 按日统计里全是 0。
            if query:
                if brain.miss_log_enabled and top_sim < brain.miss_log_threshold:
                    brain.log_miss(query, top_sim, top_k=top_k,
                                   path="mcp.recall")
            brain.bump_daily("recall")
        out = {
            "count": len(memories),
            "memories": memories,
            # 回显本次标签域与过滤后候选数（matched=None = 无过滤全库）。
            "filter_applied": filter_applied(
                parsed_tags, len(allow_ids) if allow_ids is not None else None,
                key_info),
        }
        if exclude_sensitive:
            # D2b masked 档：被滤条数自曝（0 也带键——档位行为可断言，
            # 不与"没过滤"混淆）。
            out["filtered_sensitive"] = filtered_sensitive
        if time_window[0] > 0 or time_window[1] > 0:
            # 时间窗自曝：**count=0 时尤其重要**——否则「窗口内确实没有」
            # 和「检索没找到」在返回面上长得一模一样，调用方会把前者当后者。
            out["time_window"] = {
                "after": iso_date(time_window[0]) if time_window[0] > 0 else None,
                "before": iso_date(time_window[1]) if time_window[1] > 0 else None,
            }
        return out

    def _cap_contents(self, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """截断护栏：把每条命中记忆的 ``content`` 压到字符上限内。

        为什么要有它：``recall`` 原本**只按条数收口**（``top_k``），单条
        ``content`` 想多长就多长——一条长记忆就能让一次召回上万字符，
        而调用方（宿主上下文）根本没法预判。

        为什么不静默切：切掉的内容宿主看不见，会以为"库里就这些"。所以被
        切的条目必须带 ``content_truncated=true`` + ``content_chars``
        （原文长度），要全文走 ``get_memory(memory_id)``。

        ``0`` = 不截断（**默认**，= 现状零变化）；上限比原文长时不加任何键
        （返回面与关态逐字节一致）。上限由组合根现读
        ``HIPPOCAMPUS_RECALL_CONTENT_MAX_CHARS`` 注入（application 不读 config）。
        """
        cap = int(self.recall_content_max_chars or 0)
        if cap <= 0:
            return memories
        out: list[dict[str, Any]] = []
        for d in memories:
            text = d.get("content")
            if not isinstance(text, str) or len(text) <= cap:
                out.append(d)
                continue
            d = dict(d)
            d["content"] = text[:cap]
            d["content_truncated"] = True
            d["content_chars"] = len(text)
            out.append(d)
        return out

    # ---------------------------------------------------------------- 单条与列举

    def get_memory(self, *, memory_id: str,
                   include_history: bool = False,
                   client: str = "") -> dict[str, Any]:
        """按 ID 取单条完整信息（不存在显式报，不静默返回空）。

        ``include_history=True`` 时附 ``history`` 键——
        mem_history.jsonl 里该记录的改写前值（update/merge，最新在前，
        cap 20）。默认不加键（返回面零变化）；账本是审计面，不进检索。

        D2b：目标 = user 库时按档位——masked 档 sensitive:* 卡显式报错
        （单条读取没有"滤掉"可言，给一半就是泄密）；none 档整体拒绝。
        """
        tier = self._ua(client)
        if tier == "none":
            return {"error": USER_READ_DENIED, "memory_id": memory_id}
        if tier == "masked":
            rec = self.manager.store.get(memory_id)
            if rec is not None and str(
                    getattr(rec, "sensitivity", "") or ""
            ).startswith("sensitive:"):
                return {"error": (
                            "This memory is sensitive and this workspace has "
                            "masked read access (sensitive records are unreadable)."),
                        "memory_id": memory_id}
        record = self.manager.get_memory(memory_id)
        if record is None:
            return {"error": "memory not found", "memory_id": memory_id}
        if include_history:
            record["history"] = self.manager.memory_history(
                memory_id=memory_id, limit=20)
        return record

    def list_memories(self, *, limit: int, source: str,
                      state: str, client: str = "") -> dict[str, Any]:
        """按来源/状态列元数据（不含嵌入向量）。

        D2b masked 档：sensitive:* 行滤除并计数（列表是内容面，同召回口径）。
        """
        memories = self.manager.list_memories(
            limit=limit, source=source or None, state=state or None
        )
        tier = self._ua(client)
        if tier == "masked":
            kept = []
            filtered = 0
            for row in memories:
                rec = self.manager.store.get(row.get("memory_id")
                                             or row.get("id"))
                if rec is not None and str(
                        getattr(rec, "sensitivity", "") or ""
                ).startswith("sensitive:"):
                    filtered += 1
                    continue
                kept.append(row)
            memories = kept
            return {"count": len(memories), "memories": memories,
                    "filtered_sensitive": filtered}
        return {"count": len(memories), "memories": memories}

    def list_sources(self, client: str = "") -> dict[str, Any]:
        """来源标签及各自条数（导入幂等的基础）。

        D2b：来源是 L0 回指的**索引**（能直接定位原话），masked 档一律拒绝
        （fail-closed：内容面已滤但回指面敞着等于没滤）。
        """
        if self._ua(client) in ("masked", "none"):
            return {"error": USER_READ_DENIED}
        raw = self.manager.sources()
        labeled = {("(no source)" if k == "" else k): v for k, v in raw.items() if k != ""}
        return {
            "count": len(labeled),
            "sources": dict(sorted(labeled.items(), key=lambda kv: -kv[1])),
            "unlabeled": raw.get("", 0),
        }

    # ---------------------------------------------------------------- 装配上下文

    def get_memory_context(self, *, query: str, top_k: int,
                           token_budget: int, item_max_chars: int,
                           filter_tags: str,
                           scope: str = SCOPE_PROJECT,
                           client: str = "") -> dict[str, Any]:
        """5 槽位装配（当前命中/待办未决/时间线/实体状态/上次未聊完）。

        ``filter_tags`` 解析错误体不带信封（同 :meth:`recall`）；与 recall
        的差异：不过早返回 count=0——空候选也装配（各槽自曝为空），
        ``filter_applied.matched`` 照样回显。

        scope：**只支持 project / user**（5 槽位装配没有
        "项目排前用户殿后"的自然落点，有意收窄）。scope=user 整条装配换到用户库
        brain 上跑，成功体加 ``scope`` + ``libraries`` 自曝；project 默认
        路径不加自曝键。scope=all 显式报错（错误体不带自曝键）。
        """
        err = validate_scope(scope, allow_all=False)
        if err:
            return {"error": err, "input": scope}
        parsed_tags, tag_err = parse_tags_json(filter_tags)
        if tag_err:
            return {"error": f"filter_tags parse failed: {tag_err}",
                    "input": filter_tags[:200]}
        brain = self.brain
        if scope == SCOPE_USER:
            if self.user_brain is None:
                return {"error": "scope=user requires the user-level library "
                                 "(conventional name user); it was not injected "
                                 "for this call.", "input": scope}
            # D2b：装配面的跨库通道只在 full 档开放——5 槽位装配里没有
            # 「滤一半」的自然落点（同 scope=all 只配 recall 的收窄理由）。
            tier = self._ua(client)
            if tier == "none":
                return {"error": USER_READ_DENIED, "input": scope}
            if tier == "masked":
                return {"error": (
                            "This workspace has masked read access to the user "
                            "library; assembled context (get_memory_context) does "
                            "not support sensitive-level filtering. Use "
                            "recall(scope=\"user\") (which self-reports "
                            "filtered_sensitive) or assemble from a "
                            "home-workspace session."), "input": scope}
            brain = self.user_brain
        parsed_tags, key_info = check_tag_keys(brain, parsed_tags)
        if key_info and key_info.get("error"):
            return key_info
        ctx = brain.get_memory_context(query=query, top_k=top_k,
                                       token_budget=token_budget,
                                       item_max_chars=item_max_chars,
                                       filter_tags=parsed_tags)
        # 前瞻记忆情境触发。查询实体/实词与活跃待办的
        # 触发集相交时，该待办在 block_2 置前回带（cap 5 不膨胀），触发
        # 详情走新键 ``prospective_intents`` 自曝。落在服务层后处理而非
        # brain 包装——brain.get_memory_context 保持薄壳（有 parity 与
        # AST 级测试钉住其结构），context_slots 也不动。
        # 开关由组合根现读注入（application 不 import config）。
        if self.prospective_trigger:
            self._apply_prospective(ctx, brain, query=query,
                                    token_budget=token_budget,
                                    item_max_chars=item_max_chars)
        # 元记忆置信度外显。逐条命中附 ``confidence``，
        # 全体块附收据式 ``confidence`` 自曝块（承 session_start 收据
        # 纪律：回答"这次装配的把握是什么状态、怎么算的"）。只添新键，
        # 向后兼容；不加新工具。公式与校准锚点见 domain/metacog.py。
        self._apply_confidence(ctx, brain)
        # 回显标签域 + 过滤后候选数（与 recall / contradictions 同口径；
        # matched=None = 无过滤，0 = 过滤没命中任何条目）。
        allow = brain._ids_matching_tags(parsed_tags)
        ctx["filter_applied"] = filter_applied(
            parsed_tags, len(allow) if allow is not None else None, key_info)
        if scope == SCOPE_USER:
            ctx["scope"] = SCOPE_USER
            ctx["libraries"] = [self.user_library_self]
        return ctx

    # -------------------------------------------- 前瞻触发 / 置信度

    @staticmethod
    def _apply_prospective(ctx: dict[str, Any], brain: Any, *,
                           query: str, token_budget: int,
                           item_max_chars: int) -> None:
        """block_2 情境触发回带（就地改 ``ctx``；未触发零痕迹不加键）。

        重排后重渲染 ``text``（blocks 与文本双轨同源，改了 blocks 必须
        同步重渲染，``token_estimate`` 一并刷新）。待办条目本身只读——
        不改 text / status，complete_intent 的完成语义不受影响。
        """
        if not (query or "").strip():
            return
        plan = _prospective.prospective_block2(
            getattr(brain, "intents", None) or [],
            query=query,
            extract_entities=brain.extract_entities,
            tokens_of=brain._tokens_of)
        if not plan.triggered:
            return
        blocks = ctx.get("blocks")
        if not isinstance(blocks, dict):
            return
        blocks["block_2_pending"] = plan.lines
        ctx["text"] = brain._render_blocks(blocks, token_budget, item_max_chars)
        ctx["token_estimate"] = len(ctx["text"]) // 2
        ctx["prospective_intents"] = plan.triggered

    @staticmethod
    def _apply_confidence(ctx: dict[str, Any], brain: Any) -> None:
        """逐条命中 + 各块置信度（就地改 ``ctx``；只添键，不改既有键）。

        block_1 = 逐条命中的平均；block_2/3/5 出自 first-party 层给满档；
        block_4（实体状态）由 L1 派生且 entity_facts 只回文本不回 id，
        无法逐条溯源——诚实报 None 并在 basis 写明（收据纪律：
        不可评估 ≠ 0 分）。
        """
        raw_ids = {r.get("id")
                   for r in (getattr(brain, "raw_log", None) or [])
                   if isinstance(r, dict) and r.get("id")}
        vals: list[float] = []
        for h in ctx.get("hits") or []:
            src = h.get("source_ids") or []
            vals.append(_metacog.hit_confidence(
                similarity=float(h.get("similarity") or 0.0),
                state=str(h.get("state") or ""),
                source_ids=src,
                source_resolved=sum(1 for s in src if s in raw_ids),
                pin=bool(h.get("pin"))))
            h["confidence"] = vals[-1]
        blocks = ctx.get("blocks") or {}

        def _first_party(lines: Any, basis: str) -> dict[str, Any]:
            """first-party 块的置信条目：非空给满档，占位/空 = None（非 0 分）。

            basis 与 value 联动：None 时报 empty_*，不给"满档"的假标签。
            """
            lines = lines or []
            if not lines or (len(lines) == 1 and str(lines[0]) in BLOCK_PLACEHOLDERS):
                return {"value": None, "basis": "empty_" + basis}
            return {"value": _metacog.FIRST_PARTY_SCORE, "basis": basis}

        ctx["confidence"] = _metacog.confidence_receipt(
            {
                "block_1_current_hits": {
                    "value": _metacog.mean_confidence(vals),
                    "basis": (f"mean_of_{len(vals)}_hits"
                              if vals else "empty")},
                "block_2_pending": _first_party(
                    blocks.get("block_2_pending"), "first_party_intents"),
                "block_3_timeline": _first_party(
                    blocks.get("block_3_timeline"), "first_party_summaries"),
                "block_4_entity_state": {
                    "value": None,
                    "basis": "derived_from_L1_no_direct_trace"},
                "block_5_open_topics": _first_party(
                    blocks.get("block_5_open_topics"), "first_party_open_topics"),
            },
            computed_at=time.time())

    # ---------------------------------------------------------------- 矛盾与漂移

    def contradictions(self, *, limit: int, source: str,
                       filter_tags: str) -> dict[str, Any]:
        """「同一事实、两种口径」候选对（只标注不裁决，确认后走 update_memory）。"""
        brain = self.brain
        parsed_tags, tag_err = parse_tags_json(filter_tags)
        if tag_err:
            return {"error": f"filter_tags parse failed: {tag_err}",
                    "input": filter_tags[:200]}
        parsed_tags, key_info = check_tag_keys(brain, parsed_tags)
        if key_info and key_info.get("error"):
            return key_info
        result = brain.contradictions(limit=limit, source=source,
                                      filter_tags=parsed_tags)
        # 回显标签域 + 过滤后候选数。candidate_pool 就是"收窄后进比对的条数"，
        # 与 recall / get_memory_context 的 matched 同口径。
        result["filter_applied"] = filter_applied(
            parsed_tags, result.get("candidate_pool"), key_info)
        return result

    def drift_audit(self, *, source: str, limit: int,
                    include_ok: bool, client: str = "") -> dict[str, Any]:
        """源文件漂移审计：签名 vs 磁盘现状，六类回报各带分母。

        D2b（Q-14 倾向落地）：目标 = user 库时审计面**仅家可调**——审计会
        吐 L2/L3 聚合内容（画像/主题），masked 档下等于绕过内容滤除的面。
        待审复审本就该在家工作区做（提权键不在 agent 手里）。

        D2a 起附带**待审队列过目区**（设计稿 §4.3 拍板①）：本库的
        quarantine 卡逐条列出供用户过目（转正走 ``update_memory`` 的
        ``clear_quarantine``，删除走 ``forget_memory``）；user 库还附
        access_log 尾窗——"谁在什么时候对用户库动过笔"在这里可审。
        审计是旁路：本节任何失败不阻断源漂移报告本身。
        """
        # 档位门只对 user 库目标生效（resolver 对非 user 目标返回 ""）——
        # 用 in 判 masked/none，不能用 != "full"（空串档位会被误拒）。
        _tier = self._ua(client)
        if _tier in ("masked", "none"):
            return {"error": (
                "User-library audit (drift_audit) is callable only from a home "
                f"workspace (current access level: {_tier}); audits expose L2/L3 "
                "aggregate content, so masked/none levels are always refused. "
                "Review quarantined items from a home workspace (daily assistant "
                "host) session.")}
        out = self.brain.source_drift(source=source, limit=limit,
                                      include_ok=include_ok)
        try:
            quarantined = []
            for rec in self.manager.store.iter_records():
                if getattr(rec, "quarantine", False):
                    quarantined.append({
                        "memory_id": rec.id,
                        "content_preview": (rec.content or "")[:120],
                        "sensitivity": getattr(rec, "sensitivity", ""),
                        "timestamp": rec.timestamp,
                        "source_ids": list(rec.source_ids or []),
                    })
            quarantined.sort(key=lambda r: -r["timestamp"])
            out["quarantine"] = {
                "count": len(quarantined),
                "items": quarantined[:limit],
                "hint": ("Promote: update_memory(memory_id=..., "
                         "clear_quarantine=True) (home-library session only); "
                         "delete: forget_memory(memory_id=...)."),
            }
            # B2 目录健康：库路径/绑定根失联与僵尸条目（描述审计挂点，
            # 设计稿 §2.2/§3.2——目录 note 会腐化，路径会漂移，要看得见）。
            try:
                ulib = resolve_library("user", libs=scan_libraries())
                ddata, dissue = load_directory(ulib.path)
                health: dict[str, Any] = {"ready": dissue is None,
                                          "entries": 0,
                                          "path_missing": [],
                                          "workspace_missing": []}
                if dissue is None:
                    ents = ddata.get("entries") or []
                    health["entries"] = len(ents)
                    for e in ents:
                        if not isinstance(e, dict):
                            continue
                        if not os.path.isdir(str(e.get("path") or "")):
                            health["path_missing"].append(e.get("task"))
                        wr = str(e.get("workspace_root") or "")
                        if wr and not os.path.isdir(wr):
                            health["workspace_missing"].append(e.get("task"))
                out["directory_health"] = health
            except Exception as e2:  # noqa: BLE001 —— 审计旁路
                out["directory_health"] = {
                    "ready": False, "reason": f"{type(e2).__name__}: {e2}"}
            if (self.library_self or {}).get("name") == "user":
                out["access_log"] = read_access_events(self.persist_dir,
                                                       limit=limit)
        except Exception as e:  # noqa: BLE001 —— 审计旁路绝不炸主报告
            out["quarantine_error"] = f"{type(e).__name__}: {e}"
        return out

    # ---------------------------------------------------------------- 实体档案

    def entity_profile(self, *, entity: str, limit: int, order: str,
                       include_superseded: bool) -> dict[str, Any]:
        """单实体完整档案：参与过的全部记忆按时间拉一遍。"""
        return self.brain.entity_profile(entity, limit=limit, order=order,
                                         include_superseded=include_superseded)

    def entity_list(self, *, prefix: str, limit: int, min_mentions: int,
                    trusted_only: bool) -> dict[str, Any]:
        """已有哪些实体（按提及条数降序；可信来源过滤排噪）。"""
        return self.brain.entity_list(prefix=prefix, limit=limit,
                                      min_mentions=min_mentions,
                                      trusted_only=trusted_only)

    # ---------------------------------------------------------------- 知识地图

    def knowledge_map(self, *, limit: int,
                      include_archived: bool) -> dict[str, Any]:
        """知识结构总览（主题 + 代表句 + playbook）。**带** library 信封。"""
        return self._envelope(self.brain.knowledge_map(
            limit=limit, include_archived=include_archived))

    def knowledge_topic(self, *, topic_id: str, action: str, summary: str,
                        evidence: str, limit: int, topics: str, items: str,
                        batch: int, per_topic: int, fras: str) -> dict[str, Any]:
        """单知识点读写通道：brief / summarize / revert / rebuild / refresh。

        ``items`` / ``fras`` 的解析失败错误体**不带** library 首键——
        静默当空会让宿主以为写回成功了，实际一条都没进库。
        成功路径经 ``_envelope``（与适配层 ``_with_library`` 同规则）。
        """
        # items 是「归纳条目数组」的 JSON 字符串（[{"topic_id":..,"summary":..
        # ,"evidence":[..]}]）。解析失败要**显式报错**。
        parsed_items: Any = None
        raw_items = (items or "").strip()
        if raw_items:
            try:
                parsed_items = json.loads(raw_items)
            except json.JSONDecodeError as exc:
                return {"ok": False, "action": action,
                        "error": f"items is not valid JSON: {exc}"}
            if not isinstance(parsed_items, list):
                return {"ok": False, "action": action,
                        "error": "items must be an array (each item contains topic_id/summary/evidence)"}
        # fras 同上：解析失败显式报错（静默当"没传"会让宿主以为结构写进去了）。
        parsed_fras: Any = None
        raw_fras = (fras or "").strip()
        if raw_fras:
            try:
                parsed_fras = json.loads(raw_fras)
            except json.JSONDecodeError as exc:
                return {"ok": False, "action": action,
                        "error": f"fras is not valid JSON: {exc}"}
            if not isinstance(parsed_fras, dict):
                return {"ok": False, "action": action,
                        "error": "fras must be a JSON object (f/r/a/s keys)"}
        return self._envelope(self.brain.knowledge_topic(
            topic_id=topic_id, action=action, summary=summary,
            evidence=parse_str_list_lenient(evidence), limit=limit,
            topics=parse_str_list_lenient(topics) or None,
            items=parsed_items, batch=batch, per_topic=per_topic,
            fras=parsed_fras))
