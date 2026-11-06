# -*- coding: utf-8 -*-
"""BootstrapService：开场包与会话生命周期（5 工具）。

工具面：session_start / session_end /
remember_intent / complete_intent / report_feedback + get_stats。

关于 get_stats 的归属：46 工具五归类里唯一
没被前四组认领的就是它，语义上也讲得通——「开场先看系统状态」。
MaintainService 的 brain_stats 才是维护面统计，两者不重叠。

方法与 server.py 工具体逐字对应（去 warmup / brain 获取 /
``json.dumps``；``_with_library`` → 基类 ``_envelope``）：只挪位置
不改行为，等值由回归门禁保证；``brain.session_start`` 等方法体
后续上收进本类。
"""
from __future__ import annotations

from typing import Any

from ..common.params import filter_applied
from .base import BaseService
from .tags import check_tag_keys


class BootstrapService(BaseService):
    """开场用例：预激活包（收据/概览/画像/地图索引）+ 待办 + 反馈闭环。"""

    # ---------------------------------------------------------------- 开场包

    def session_start(self, *, task: str, recent_limit: int, working_on: str,
                      task_top_k: int, pin_limit: int,
                      filter_tags: dict[str, Any] | None,
                      client: str,
                      user_scope: int = 0,
                      task_directory: int = 0,
                      user_access: str = "full") -> dict[str, Any]:
        """每次会话第一步：拿预激活包（compaction 后凭 session_id 重建）。

        ``filter_tags`` 由适配层解析后传入（其 JSON 语法错误体不带信封，
        见 write_service 模块 docstring 的切分说明）；键名白名单校验
        要查本库 schema，属用例职责，在这里做。

        ``user_scope`` 透传给 ``brain.session_start``：
        用户级继承槽条数三态（0 = 按 env / 正 = 条数 / 负 = 关）。

        ``task_directory`` 同样透传：用户级任务目录槽条数
        三态（与 ``user_scope`` 同族；0 = 按 env / 正 = 条数 /
        负 = 关）。组装与自曝形状全在 brain 侧（``_task_directory_brief``），
        用例层只做参数搬运——与 ``user_scope`` 同一条纪律。

        ``user_access``（D2b）同样透传：本连接对 user 库的读侧档位
        （组合根算好注入），继承槽按档过滤/拒绝；缺省 full = 现状。
        """
        parsed_tags, key_info = check_tag_keys(self.brain, filter_tags)
        if key_info and key_info.get("error"):
            # 注意：此错误体**不带** library 首键（与现行工具逐位一致）。
            return key_info
        started = self.brain.session_start(
            task=task, recent_limit=recent_limit,
            working_on=working_on, task_top_k=task_top_k,
            pin_limit=pin_limit, filter_tags=parsed_tags,
            client=client or None, user_scope=user_scope,
            task_directory=task_directory, user_access=user_access)
        # 软降级必须看得见——否则剥掉的键等于被静默吞掉。
        if key_info and key_info.get("degraded"):
            started["filter_applied"] = filter_applied(parsed_tags, None, key_info)
        return self._envelope(started)

    def session_end(self, *, summary: str, open_topics: list[str],
                    client: str, close_topics: list[str] | None = None,
                    session_id: str = "") -> dict[str, Any]:
        """会话收尾：写摘要、登记未完结话题、更新画像、触发巩固。"""
        return self._envelope(self.brain.session_end(
            summary=summary, open_topics=open_topics,
            close_topics=close_topics, client=client or None,
            session_id=session_id))

    # ---------------------------------------------------------------- 待办意图

    def remember_intent(self, *, text: str, due: str) -> dict[str, Any]:
        """登记一条待办/意图（下次会话提醒一次，不重复催）。"""
        return self._envelope({"message": self.brain.track_intent(text, due)})

    def complete_intent(self, *, keyword: str) -> dict[str, Any]:
        """标记一条待办完成（按关键词匹配）。"""
        return self._envelope({"message": self.brain.resolve_intent(keyword)})

    # ---------------------------------------------------------------- 反馈与状态

    def report_feedback(self, *, memory_id: str, useful: bool) -> dict[str, Any]:
        """有用则提升重要性，无用只降优先级不降权。"""
        ok = self.brain.report_feedback(memory_id, useful)
        return self._envelope(
            {"status": "ok" if ok else "not_found", "memory_id": memory_id})

    def get_stats(self) -> dict[str, Any]:
        """系统状态统计。**不带** library 首键（与现行工具逐位一致）。"""
        return self.manager.get_stats()
