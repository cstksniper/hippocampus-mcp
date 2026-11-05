# -*- coding: utf-8 -*-
"""WriteService：唯一写入口径（9 工具）。

工具面：store_memory / update_memory /
update_tags / forget_memory / forget_by_query / ingest / ingest_batch /
retry_save / save_state。

本类的每个方法都是**从 server.py 工具体逐字搬来的编排逻辑**（去掉了
``_await_warmup`` / brain 获取 / ``json.dumps``，``_with_library`` 换成
基类 ``_envelope``）——只挪位置不改行为，行为等值由全量回归门禁
保证；后续脑方法体上收进本类时，本类 API 不动。

「谁解析参数」的切分（按**现行错误路径的信封形状**定，保字节级一致）：

* 错误体带 ``library`` 首键的（ingest / ingest_batch 的 tags、items
  解析）→ 服务内解析（信封在服务侧，适配器不碰）；
* 错误体不带信封的（update_tags 四解析、session_start 的
  filter_tags 解析）→ 留在适配器，且必须排在工厂（库解析）**之前**
  ——现行代码就是这样，调序会让「坏 JSON + 坏库名」的错误体漂移。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ..common.params import parse_tags_json
from ..domain.long_content import (
    POLICY_WARN,
    char_limit,
    plan_long_content,
    parse_policy,
)
from ..domain.sensitive import classify_sensitivity, match_sensitive
from ..domain.write_quarantine import build_access_event, sensitivity_label
from ..library.access_log import append_access_event
from ..store import ConcurrentWriteError
from ..store.memory import MemoryState
from .base import BaseService

logger = logging.getLogger("hippocampus_mcp")


class WriteService(BaseService):
    """写用例：校验→策略→落盘→账本→回显（策略与落盘细节暂居 brain/manager，后续上收）。"""

    # ---------------------------------------------------------------- 单条裸写

    # ---------------------------------------------------------------- D2a 隔离

    def _user_lib_label(self, content: str) -> str:
        """目标 = user 库时的 D1 打标（user 库自动入库默认 personal）。"""
        return sensitivity_label(classify_sensitivity(content or ""))

    def _log_access(self, *, kind: str, tool: str, quarantined: bool,
                    sensitivity: str, content: str,
                    memory_id: str | None = None) -> None:
        """追加写侧审计事件（旁路：失败只 warning，绝不影响主路径）。"""
        if not self.access_log_dir:
            return
        ev = build_access_event(
            kind=kind, tool=tool,
            home="home" if self.is_home_write else "other",
            target_name=(self.library_self or {}).get("name", ""),
            target_path=(self.library_self or {}).get("path", ""),
            quarantined=quarantined, sensitivity=sensitivity,
            content=content)
        if memory_id:
            ev["memory_id"] = memory_id
        append_access_event(self.access_log_dir, ev)

    def _apply_long_content_policy(self, *, content: str, importance: float,
                                   memory_type: str, client: str,
                                   policy: str,
                                   quarantine: bool = False,
                                   sensitivity: str = "") -> dict[str, Any] | None:
        """超长内容按策略处置。**返回 None = 不干预**，走原写入路径。

        只在用户显式开了 ``split`` / ``reject`` 时被调用（``warn`` 整段跳过，
        保持改动前的逐字节行为）。
        """
        mgr = self.manager
        embedder = getattr(mgr, "embedder", None)
        cov_fn = getattr(embedder, "embedding_coverage", None)
        if not callable(cov_fn):
            return None
        try:
            limit = int(getattr(embedder, "max_seq_length", 0) or 0)
        except (TypeError, ValueError):
            limit = 0
        if limit <= 0:
            # 拿不到上限（hash 后端 / 元数据缺失）⇒ 无从判定，不干预。
            return None
        # 字符数都不超过 token 上限 ⇒ 任何语言都不可能超窗（中文 1 字≈1
        # token，英文 1 token≈4 字）。让短卡保持零额外开销。
        if len(content or "") <= limit:
            return None
        try:
            cov = cov_fn(content or "") or {}
        except Exception:  # 自曝绝不能反过来把写入搞挂（与截断自曝同口径）
            return None

        plan = plan_long_content(
            content or "", limit, policy,
            truncated=bool(cov.get("truncated")),
            split_facts=self.split_facts)
        if plan.action == "write":
            return None

        if plan.action == "reject":
            return self._envelope({
                "status": "rejected_long_content",
                "memory_id": None,
                "error": plan.reason,
                "long_content_policy": policy,
                "content_chars": len(content or ""),
                "char_limit": char_limit(limit),
                "content_preview": (content or "")[:100],
            })

        ids: list[str] = []
        for piece in plan.pieces:
            ids.append(mgr.store_memory(
                content=piece, importance=importance,
                memory_type=memory_type, client=client,
                # D2a：拆分产物继承隔离标记——拆分不能变成隔离的绕路口。
                quarantine=quarantine, sensitivity=sensitivity))
        out: dict[str, Any] = {
            "status": "stored_split",
            "memory_id": ids[0] if ids else None,  # 兼容单条回显的旧字段
            "memory_ids": ids,
            "piece_count": len(ids),
            "long_content_policy": policy,
            "long_content_reason": plan.reason,
            "content_chars": len(content or ""),
            "char_limit": char_limit(limit),
            "content_preview": (content or "")[:100],
        }
        if plan.note:
            out["long_content_note"] = plan.note
        if quarantine:
            # 与单条路径同口径：隔离态自曝不缺席（拆分也进待审队列）。
            out["quarantined"] = True
            out["quarantine_note"] = (
                "本批拆分产物已全部进入 user 库**待审队列**：不进召回池、"
                "不进 user_scope 槽；请在 drift_audit 报告中过目后转正或删除。")
        if sensitivity:
            # D1：打标结果自曝（拆分产物与单条同口径，家库写入也带）。
            out["sensitivity"] = sensitivity
        return self._envelope(out)

    def store_memory(self, *, content: str, importance: float,
                     memory_type: str, client: str,
                     origin: dict[str, str] | None = None,
                     idem_key: str = "") -> dict[str, Any]:
        """单条事实裸写（无 L0 溯源、不建实体；长文走 :meth:`ingest`）。

        敏感过滤红线：目标库名为 ``user``（用户级库约定名）
        时走**专属敏感过滤**（堵直写绕路）——命中即拒写，
        返回体自曝命中词与位置（不静默）。**存量库行为不变**：非 user
        库零分支、零 I/O。词表内容由组合根注入
        （application 不读 env）；模式级内置兜底在 domain（词表缺省
        也生效）。工厂层的家库写守卫与本过滤**正交组合**——
        守卫也不得绕过这道过滤另开写口。
        """
        if (self.library_self or {}).get("name") == "user":
            hits = match_sensitive(content or "", words=self.sensitive_words)
            if hits:
                return self._envelope({
                    "status": "blocked_sensitive",
                    "memory_id": None,
                    "hits": hits,
                    "content_preview": (content or "")[:100],
                    "note": ("用户级库不收敏感内容。命中详情见 "
                             "hits；改写后重发，或按提升流人工两步写入。"),
                })
        # D2a/D1：目标 = user 库 ⇒ 打标（user 库自动入库默认 personal，
        # 宁枉勿纵）；非家工作区（工厂注入的隔离判据为真）⇒ 隔离标记
        # + 审计事件。非 user 库目标零分支零开销。
        quarantined = bool(self.user_write_quarantine)
        label = ""
        if (self.library_self or {}).get("name") == "user":
            label = self._user_lib_label(content)
        # 超长内容策略。**warn（默认）整段跳过**——现状是
        # 「先写、再自曝」，前置再算一次 coverage 会多付一次 tokenizer，
        # 且没必要改一条人人依赖的默认路径。只有用户显式开 split/reject
        # 才走下面的分支（白名单语义，写错等同于没写）。
        policy = parse_policy(self.long_content_policy)
        if policy != POLICY_WARN:
            decided = self._apply_long_content_policy(
                content=content, importance=importance,
                memory_type=memory_type, client=client, policy=policy,
                quarantine=quarantined, sensitivity=label)
            if decided is not None:
                if quarantined or label:
                    self._log_access(kind="store_memory", tool="store_memory",
                                     quarantined=quarantined, sensitivity=label,
                                     content=content)
                return decided

        mgr = self.manager
        memory_id = mgr.store_memory(
            content=content,
            importance=importance,
            memory_type=memory_type,
            # client 接线（daemon 多会话实测教训）：不传 client 时 LNN 隐状态
            # 恒落在 __default__ 那一路 —— stdio 下无所谓（一进程一会话），
            # daemon 下 N 条连接就全挤在一路，A 的写入直接改写 B 的检索起点，
            # 无异常、无日志，只掉时序分。
            client=client,
            quarantine=quarantined,
            sensitivity=label,
            origin=origin,
            idem_key=idem_key,
        )
        # P0-2 幂等命中：manager 在嵌入前就返回了既有 id（省掉最贵那一步），
        # 此时 last_truncation 是**上一次调用**的残留值——绝不能拿它拼返回体，
        # 否则宿主会看到一条与本次无关的截断告警。这里走极简回执。
        if getattr(mgr, "last_idem_hit", None):
            return self._envelope({
                "status": "duplicate",
                "idempotent": True,
                "memory_id": memory_id,
                "content_preview": content[:100],
                "note": ("同一 idem_key 的重放：未新建、未重新嵌入，直接回既有"
                         "记忆 id。若这次其实是新内容，说明宿主复用了键。"),
            })
        if quarantined or label:
            self._log_access(kind="store_memory", tool="store_memory",
                             quarantined=quarantined, sensitivity=label,
                             content=content, memory_id=memory_id)
        # 截断自曝。嵌入模型有 token **硬上限**（BGE / MiniLM = 512），
        # 超出的部分在 encode 时被静默丢弃、对向量零贡献——写入成功、回读在库、
        # brain_stats 正常，**只有检索悄悄差一截**。这里把「实际纳入嵌入多少字」
        # 摆在返回体前面，让宿主当场就能看见，而不是等检索不到才发现。
        tcov = getattr(mgr, "last_truncation", None) or {}
        truncated = bool(tcov.get("truncated"))
        out: dict[str, Any] = {
            "status": "stored",
            "memory_id": memory_id,
            "truncated": truncated,
            "max_seq_length": tcov.get("max_seq_length", 0),
            "embedded_chars": tcov.get("embedded_chars", len(content)),
            "content_chars": tcov.get("total_chars", len(content)),
            "content_preview": content[:100],
        }
        if truncated:
            out["truncation_warning"] = (
                f"本条内容 {out['content_chars']} 字超过嵌入模型 token 上限 "
                f"{out['max_seq_length']}，仅前 {out['embedded_chars']} 字进入向量——"
                f"其后的原文写入成功但**检索不到**。请把长内容拆成多张 <500 字的卡。"
            )
        if quarantined:
            # 拍板①：写入返回体自曝隔离态（绝不静默）——「写成功但对所有
            # 下游通道不可见」必须当场可见，否则宿主以为记忆已生效。
            out["quarantined"] = True
            out["quarantine_note"] = (
                "本条已进入 user 库**待审队列**（非家工作区经豁免通道的直写）："
                "不进召回池、不进 user_scope 槽；请让用户在 drift_audit 报告中"
                "过目后经受管通道转正或删除。")
        if label:
            # D1：打标结果自曝（家库写入也带——宿主该知道卡被打了什么标签）。
            out["sensitivity"] = label
        return self._envelope(out)

    # ---------------------------------------------------------------- 字段更新

    def update_memory(self, *, memory_id: str, content: str,
                      importance: float, memory_type: str, state: str,
                      superseded_by: str,
                      clear_quarantine: bool = False) -> dict[str, Any]:
        """只更新提供的字段，其余保持不变（「改成 X」通道）。

        ``clear_quarantine``（D2a 受管转正）：把 user 库的待审卡转正。
        **提权键不在 agent 手里**：只对家库会话开放（或逃生门开——工厂把
        两种情况都算进 ``is_home_write``）；非家会话请求 = 显式报错，
        绝不静默忽略。转正动作本身也进 access_log（谁转的正，可审）。
        """
        mgr = self.manager
        if clear_quarantine:
            if not self.is_home_write:
                return self._envelope({
                    "status": "error",
                    "memory_id": memory_id,
                    "error": ("clear_quarantine 只对家库会话开放：待审转正"
                              "是受管动作，请在 user 库的家工作区（日常助手"
                              "宿主）会话里操作，或开逃生门 env 做运维转正。"),
                })
            rec = mgr.store.get(memory_id)
            if rec is None:
                return self._envelope({"status": "not_found",
                                       "memory_id": memory_id})
            if not getattr(rec, "quarantine", False):
                return self._envelope({
                    "status": "noop", "memory_id": memory_id,
                    "note": "该记忆不在待审队列（quarantine=False），无需转正。"})
            mgr.update_memory(memory_id, quarantine=False)
            self._log_access(kind="quarantine_clear", tool="update_memory",
                             quarantined=False, sensitivity="",
                             content=str(getattr(rec, "content", "")),
                             memory_id=memory_id)
            return self._envelope({
                "status": "quarantine_cleared", "memory_id": memory_id,
                "note": "已转正：该记忆恢复进召回池与 user_scope 槽。"
                        "转正动作已记 access_log。"})
        fields: dict[str, Any] = {}
        warnings: list[str] = []
        if content:
            fields["content"] = content
        if importance >= 0:
            fields["importance"] = importance
        if memory_type:
            fields["memory_type"] = memory_type

        if state:
            allowed = {s.value for s in MemoryState}
            st = state.strip().lower()
            if st not in allowed:
                return self._envelope(
                    {"status": "error", "memory_id": memory_id,
                     "error": f"非法 state: {state!r}",
                     "allowed_states": sorted(allowed)})
            fields["state"] = st
            if st == MemoryState.SUPERSEDED.value and not superseded_by:
                warnings.append(
                    "已标 superseded 但未给 superseded_by：日后无法回溯是被哪条取代的"
                )

        if superseded_by:
            if mgr.store.get(superseded_by) is None:
                return self._envelope(
                    {"status": "error", "memory_id": memory_id,
                     "error": f"superseded_by 指向的记忆不存在: {superseded_by}"})
            fields["superseded_by"] = superseded_by
            if not state:
                # 只给了 superseded_by 没给 state：按语义补齐，而不是留下
                # "有指针但状态仍 active" 的自相矛盾状态。
                fields["state"] = MemoryState.SUPERSEDED.value
                warnings.append("未显式给 state，已按 superseded_by 的语义补为 superseded")

        if not fields:
            # 注意：此错误体**不带** library 首键（与现行工具逐位一致）。
            return {"error": "no fields to update"}

        success = mgr.update_memory(memory_id, **fields)
        payload: dict[str, Any] = {
            "status": "updated" if success else "not_found",
            "memory_id": memory_id,
            "updated_fields": sorted(fields),
        }
        if warnings:
            payload["warnings"] = warnings
        return self._envelope(payload)

    # ---------------------------------------------------------------- 批量标签

    def update_tags(self, *, tags: dict[str, Any] | None, clear_keys: list[str] | None,
                    source: str, source_contains: str, contains: str, state: str,
                    untagged_only: bool, missing_keys: list[str] | None,
                    memory_ids: list[str] | None, mode: str, dry_run: bool,
                    limit: int, confirm: bool, undo: str) -> dict[str, Any]:
        """存量标签回填（默认 dry_run；可撤销，fail-closed）。

        入参的 JSON 解析在适配层完成（其错误体不带信封，见模块 docstring）；
        这里收已解析值。dry_run / confirm 闸门 / 幂等 / 账本 semantics 全在
        ``brain.update_tags``（后续随方法体一起上收）。
        """
        return self._envelope(self.brain.update_tags(
            tags=tags, clear_keys=clear_keys,
            source=source, source_contains=source_contains, contains=contains,
            state=state, untagged_only=untagged_only, missing_keys=missing_keys,
            memory_ids=memory_ids, mode=mode, dry_run=dry_run, limit=limit,
            confirm=confirm, undo=undo,
        ))

    # ---------------------------------------------------------------- 删除

    def forget_memory(self, *, memory_id: str, reason: str) -> dict[str, Any]:
        """按 ID 级联删除（清实体倒排 / L3 派生 / 墓碑 / 恢复被取代者）。"""
        return self._envelope(self.brain.forget(memory_id, reason=reason))

    def forget_by_query(self, *, query: str, top_k: int, dry_run: bool,
                        min_similarity: float, allow_multiple: bool) -> dict[str, Any]:
        """按自然语言定位疑似错误记忆并删除（默认预览、默认单删）。"""
        return self._envelope(self.brain.forget_by_query(
            query, top_k=top_k, dry_run=dry_run, allow_multiple=allow_multiple,
            min_similarity=None if min_similarity < 0 else min_similarity,
        ))

    # ---------------------------------------------------------------- 导入

    def ingest(self, *, text: str, speaker: str, importance: float,
               cognitive_type: str, entities: str, source: str,
               tags: str, auto_tags: bool,
               origin: dict[str, str] | None = None,
               idem_key: str = "") -> dict[str, Any]:
        """推荐写入口：原文进 L0、抽句进 L1、维护实体索引。

        与工具层的切分：``tags`` 的解析错误体带 library 首键，故在服务内
        解析（保字节级一致）；``entities`` / ``importance`` /
        ``cognitive_type`` 的哨兵归一无错误路径，随编排放这里。
        """
        logger.info("[ingest] 开始: %s", text[:40])
        parsed_tags, tag_err = parse_tags_json(tags)
        if tag_err:
            return self._envelope(
                {"error": f"tags 解析失败：{tag_err}", "input": tags[:200]})
        # D2a/D1：目标 = user 库 ⇒ 整批 L1 打标（对原文分类一次，批内共用，
        # 宁枉勿纵）；非家工作区 ⇒ 整批 L1 带隔离标记 + 审计事件。
        quarantined = bool(self.user_write_quarantine)
        label = ""
        if (self.library_self or {}).get("name") == "user":
            label = self._user_lib_label(text)
        result = self.brain.ingest(
            text=text,
            speaker=speaker,
            importance=None if importance < 0 else importance,
            cognitive_type=cognitive_type or None,
            entities=[e.strip() for e in entities.split(",") if e.strip()] or None,
            source=source,
            tags=parsed_tags,
            auto_tags=auto_tags,
            quarantine=quarantined,
            sensitivity=label,
            origin=origin,
            idem_key=idem_key,
        )
        if quarantined or label:
            self._log_access(kind="ingest", tool="ingest",
                             quarantined=quarantined, sensitivity=label,
                             content=text)
            if quarantined:
                result["quarantined"] = True
                result["quarantine_note"] = (
                    "本批 %d 条 L1 全部进入 user 库**待审队列**（非家工作区经"
                    "豁免通道的直写）：不进召回池、不进 user_scope 槽；请在 "
                    "drift_audit 报告中过目后经受管通道转正或删除。"
                    % len(result.get("memory_ids", [])))
            if label:
                # D1：打标结果自曝（家库写入也带——宿主该知道卡被打了什么标签）。
                result["sensitivity"] = label
        logger.info("[ingest] 完成: %d 条记忆", len(result.get("memory_ids", [])))
        # 自动保存被并发保护拒绝时必须当场告警：宿主看到"已写入"会以为落盘了，
        # 实际这批只在内存里，进程重启即丢（此前这里完全静默 —— zcode 反馈）。
        save_err = self.manager.last_save_error
        if save_err:
            result["save_warning"] = (
                f"自动保存失败，本批记忆仅在内存中：{save_err}。"
                f"请调用 save_state；若报 conflict 则先 load_state 或 retry_save。"
            )
            logger.warning("[ingest] %s", result["save_warning"])
        return self._envelope(result)

    def ingest_batch(self, *, items: Any, source: str, tags: str,
                     granularity: str, detect_supersede: bool, save: bool,
                     auto_tags: bool,
                     origin: dict[str, str] | None = None) -> dict[str, Any]:
        """批量写入（逐项错误隔离 + 整批收尾只落盘一次）。

        ``items`` 的 JSON 解析错误体带 library 首键，故在服务内解析
        （理由同 :meth:`ingest`）；``items`` 形态标注 ``Any``：工具签名
        写 ``str``，但历史上 Host 直传数组也走到这（防御分支保留）。
        """
        parsed_tags, tag_err = parse_tags_json(tags)
        if tag_err:
            return self._envelope(
                {"error": f"tags 解析失败：{tag_err}", "input": tags[:200]})
        if isinstance(items, str):
            s = items.strip()
            if not s:
                return self._envelope(
                    {"error": "items 为空；请传 JSON 数组字符串，如 [{\"text\":\"…\"}]"})
            try:
                parsed_items = json.loads(s)
            except Exception as e:  # noqa: BLE001
                # 解析失败必须显式报错——静默当空会让宿主以为整批写进去了。
                return self._envelope(
                    {"error": f"items JSON 解析失败：{e}", "input": s[:200]})
        else:
            parsed_items = items
        try:
            # D2a/D1：与单条 ingest 同口径（标签对批内全部条目原文合并分类
            # 一次；隔离是批次级策略，见 brain.ingest_batch 的注入点注释）。
            quarantined = bool(self.user_write_quarantine)
            label = ""
            if (self.library_self or {}).get("name") == "user":
                label = self._user_lib_label(json.dumps(
                    parsed_items, ensure_ascii=False)[:2000])
            result = self.brain.ingest_batch(
                items=parsed_items,
                source=source,
                tags=parsed_tags,
                granularity=granularity,
                detect_supersede=detect_supersede,
                auto_tags=auto_tags,
                save=save,
                quarantine=quarantined,
                sensitivity=label,
                origin=origin,
            )
        except ValueError as e:
            # 非数组 / 超单批上限：直接回错误，别让宿主以为写成功了。
            return self._envelope({"error": str(e)})
        if quarantined or label:
            self._log_access(kind="ingest_batch", tool="ingest_batch",
                             quarantined=quarantined, sensitivity=label,
                             content=json.dumps(parsed_items,
                                                ensure_ascii=False)[:2000])
            if quarantined:
                result["quarantined"] = True
                result["quarantine_note"] = (
                    "本批全部条目进入 user 库**待审队列**：不进召回池、"
                    "不进 user_scope 槽；请在 drift_audit 报告中过目后"
                    "经受管通道转正或删除。")
        if label:
            # D1：打标结果自曝（家库写入也带——宿主该知道卡被打了什么标签）。
            result["sensitivity"] = label
        logger.info("[ingest_batch] 完成: %d/%d 条，失败 %d",
                    result["count"], result["total_input"], result["failed"])
        return self._envelope(result)

    # ---------------------------------------------------------------- 落盘

    def save_state(self) -> dict[str, Any]:
        """全量持久化（向量库 + LNN + L0~L3 + 实体索引）。"""
        try:
            self.brain.save()
        except ConcurrentWriteError as e:
            return self._envelope(
                {"status": "conflict", "detail": str(e),
                 "hint": "另一个进程改写了同一个记忆库。先 load_state 拿最新状态再重做本次修改。"})
        return self._envelope(
            {"status": "saved", "persist_dir": self.persist_dir})

    def retry_save(self) -> dict[str, Any]:
        """并发冲突后重试：取磁盘最新版，只补「磁盘上没有的增量」。"""
        try:
            result = self.brain.retry_save()
        except ConcurrentWriteError as e:
            return self._envelope(
                {"status": "conflict", "detail": str(e),
                 "hint": "冲突仍未停止：先停掉其它写入进程（关掉另一个 MCP 会话）再重试。"})
        return self._envelope({"status": "retried", **result})
