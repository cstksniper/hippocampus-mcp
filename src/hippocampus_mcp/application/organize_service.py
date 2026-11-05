# -*- coding: utf-8 -*-
"""OrganizeService：库经营侧用例（10 工具）。

工具面：consolidate /
relink_entities / lexicon_add / lexicon_suggest / lexicon_alias_suggest /
map_set / map_get / map_list / map_verify / reset_context。

本类的每个方法都是**从 server.py 工具体逐字搬来的编排逻辑**（去掉了
``_await_warmup`` / brain 获取 / ``json.dumps``）——只挪位置不改行为，
行为等值由全量回归门禁保证；脑方法体后续上收进本类，届时本类 API
不动。

「谁解析参数」的切分（按**现行代码里解析与库解析的先后**
定，保字节级一致）：

* 9 个工具的解析（consolidate 的 filter_tags、词库三工具的
  terms/exclude/cue_words）现行都排在库解析**之后**，故随编排整体进
  服务——工厂调用即库解析，方法体里的解析自然晚于它，错误次序不变；
* ``map_set`` 的 ``fields`` 是唯一例外：现行代码先解析 fields 后才
  ``_brain_for(library)``，且解析失败的错误体**带**
  library 首键。保序（坏 fields + 坏库名 ⇒ 先报 fields 错，不碰库）要求
  协议解析留在适配层，本方法接已解析的 ``fields: dict | None``——与
  update_tags 四段 JSON 解析的同构切法；
* 词库三工具的 terms/exclude/cue_words 走 lenient 解析
  （``common.params.parse_str_list_lenient``，永不报错），随编排进服务。

信封（库自曝首键）**按工具逐个保留现行形态**，不做统一：

* **带**信封：consolidate / relink_entities / lexicon_add / map_set /
  map_verify（现行经 ``_with_library``；map_set 的 fields 解析错与
  brain.map_set 的 ValueError 错误体也带——适配层与服务分别用
  ``_with_library`` / ``_envelope`` 组装，逐位一致）；
* **不带**：lexicon_suggest / lexicon_alias_suggest / map_get /
  map_list / reset_context（现行直接 ``json.dumps``）。
"""
from __future__ import annotations

import json
from typing import Any

from ..common.params import (filter_applied, parse_str_list_lenient,
                             parse_tags_json)
from ..domain.promote_candidates import collect_promote_candidates
from ..domain.sensitive import match_sensitive
from .base import BaseService
from .tags import check_tag_keys


class OrganizeService(BaseService):
    """库经营用例：巩固/实体重挂/词库/信息地图/LNN 上下文重置（策略与算法暂居 brain，后续上收）。"""

    # ---------------------------------------------------------------- 巩固

    def consolidate(self, *, filter_tags: str, max_items: int,
                    dry_run: bool) -> dict[str, Any]:
        """睡眠修剪：标被取代事实、归档沉寂、列矛盾候选（dry_run 只列不改）。

        ``filter_tags`` 解析排在库解析之后（随服务走，保序）；解析错误体
        **不带** library 首键（与现行工具逐位一致）；成功路径经
        ``_envelope``。软降级回显：剥掉的键必须看得见。

        返回体尾附 ``promote_candidates`` 键（扩展返回体不改存量语义；
        **绝不自动提升**——写入走
        ``store_memory(library="user")`` 的人工两步）。判据与标注见
        :func:`domain.promote_candidates.collect_promote_candidates`；
        词表内容 / 稳定期限阈值由组合根注入（application 不读 env）。
        ``brain.consolidate``（session_end 内部路径）**不带**此键——
        候选清单是 MCP 工具面的赠品，不进会话收尾热路径。
        """
        brain = self.brain
        parsed_tags, tag_err = parse_tags_json(filter_tags)
        if tag_err:
            return {"error": f"filter_tags 解析失败：{tag_err}",
                    "input": filter_tags[:200]}
        parsed_tags, key_info = check_tag_keys(brain, parsed_tags)
        if key_info and key_info.get("error"):
            return key_info
        report = brain.consolidate(filter_tags=parsed_tags, max_items=max_items,
                                   dry_run=dry_run)
        # 提升候选。实体表服务层现取组装（每库一份
        # 的 I/O 留在服务层，domain 只收倒排 dict）。
        try:
            entities = brain.entities or {}
            report["promote_candidates"] = collect_promote_candidates(
                list(brain.manager.store.iter_records()),
                summaries=brain.summaries,
                entity_members={name: meta.get("memory_ids") or []
                                for name, meta in entities.items()},
                min_sessions=self.promote_min_sessions,
                sensitive_fn=lambda text: match_sensitive(
                    text, words=self.sensitive_words),
            )
        except Exception as e:  # noqa: BLE001 —— 候选清单是赠品，不炸巩固
            report["promote_candidates"] = []
            report["promote_candidates_error"] = f"{type(e).__name__}: {e}"
        # 软降级必须看得见——否则剥掉的键等于被静默吞掉。
        if key_info and key_info.get("degraded"):
            report["filter_applied"] = filter_applied(parsed_tags, None, key_info)
        return self._envelope(report)

    # ---------------------------------------------------------------- 实体重挂

    def relink_entities(self) -> dict[str, Any]:
        """全库重跑实体抽取并补挂（词库扩充后的收尾动作）。**带**信封。"""
        return self._envelope(self.brain.relink_entities())

    # ---------------------------------------------------------------- 词库三工具

    def lexicon_add(self, *, terms: str,
                    relink: bool) -> dict[str, Any]:
        """词库追加专名并热加载（含别名行）。**带**信封。

        ``terms`` 走 lenient 解析（JSON 数组或逗号串都收）——现行就在
        ``_brain_for`` 之后解析，随编排进服务保序。
        """
        return self._envelope(self.brain.lexicon_add(
            parse_str_list_lenient(terms), relink=relink))

    def lexicon_suggest(self, *, top_n: int, min_docs: int, min_files: int,
                        min_len: int, max_len: int,
                        exclude: str) -> dict[str, Any]:
        """高频未登记候选专名（只展示不落盘）。**不带**信封。"""
        return self.brain.lexicon_suggest(
            top_n=top_n, min_docs=min_docs, min_files=min_files,
            min_len=min_len, max_len=max_len,
            exclude=parse_str_list_lenient(exclude))

    def lexicon_alias_suggest(self, *, top_n: int, min_confidence: float,
                              min_support: int, min_lift: float,
                              min_containment: float, allow_stat_only: bool,
                              exclude: str, cue_words: str,
                              with_paren: bool) -> dict[str, Any]:
        """「同一实体多种称法」候选（只出候选+证据，不自动归并）。**不带**信封。"""
        return self.brain.discover_aliases(
            top_n=top_n, min_confidence=min_confidence, min_support=min_support,
            min_lift=min_lift, min_containment=min_containment,
            allow_stat_only=allow_stat_only,
            exclude=parse_str_list_lenient(exclude),
            cue_words=parse_str_list_lenient(cue_words),
            with_paren=with_paren,
        )

    # ---------------------------------------------------------------- 信息地图

    def map_set(self, *, name: str, key: str, status: str,
                fields: dict | None, desc: str, truth_source: str, note: str,
                updated_by: str) -> dict[str, Any]:
        """改地图条目当前状态（同键再写整体替换）。**带**信封。

        ``fields`` 的 JSON 协议解析在**适配层**（保序：现行先解析 fields
        后才 ``_brain_for``，且解析错也带信封——见模块 docstring）；本方法
        只接已解析的 dict。brain.map_set 的 ValueError（非法 name/key）
        错误体同样带信封，与现行逐位一致。
        """
        try:
            out = self.brain.map_set(name, key, status=status, fields=fields,
                                     desc=desc, truth_source=truth_source,
                                     note=note, updated_by=updated_by)
        except ValueError as exc:
            return self._envelope({"status": "error", "error": str(exc)})
        return self._envelope(out)

    def map_get(self, *, name: str, key: str) -> dict[str, Any]:
        """精确查条目现在是什么（未命中 found=false + reason）。**不带**信封。"""
        return self.brain.map_get(name, key)

    def map_list(self, *, name: str) -> dict[str, Any]:
        """列地图（名+条数+新鲜度+漂移数）或单图全部条目。**不带**信封。"""
        return self.brain.map_list(name=name)

    def map_verify(self, *, name: str) -> dict[str, Any]:
        """核对地图条目与真相源文件（重读比签名，zero-read 不误报）。**带**信封。"""
        return self._envelope(self.brain.map_verify(name=name))

    # ---------------------------------------------------------------- LNN 上下文

    def reset_context(self) -> dict[str, Any]:
        """重置 LNN 时序上下文（清隐藏状态与滑窗，不删记忆）。**不带**信封。"""
        self.manager.reset_context()
        return {"status": "context_reset", "message": "LNN 时序上下文已清空"}
