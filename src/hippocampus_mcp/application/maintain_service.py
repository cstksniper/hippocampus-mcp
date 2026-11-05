# -*- coding: utf-8 -*-
"""MaintainService：库状态维护用例（5 工具）。

工具面（原「Maintain 11」的**库状态维护**那一半）：
export_memories / miss_log / brain_stats / save_brain / load_state。

.. note:: session_end 随 BootstrapService 进服务（它是会话生命
   周期收尾，语义属于开场/收尾那一组）。其余 5 个 library_* 工具
   **不进本类**——它们不绑定单一 brain（``library_info`` 刻意不构造
   HippoBrain 以免等嵌入模型 25–33 秒），见 ``library_admin_service``。

「谁解析参数」的切分：本类 5 个工具**没有 JSON 协议解析**（唯一形态转换是
``export_memories`` 的 ``source or None`` / ``state or None`` 空串归一），
不存在「解析排在库解析前还是后」的保序问题——编排整体进
服务，工厂调用即库解析，错误次序不变。

信封（库自曝首键）**按工具逐个保留现行形态**：

* **带**信封：只有 ``save_brain``（现行经 ``_with_library``，含 conflict
  分支）；
* **不带**：export_memories / miss_log / brain_stats / load_state（现行直接
  ``json.dumps``）。

``brain_stats`` 的 ``warmup`` 一节刻意由**注入的探针**产出（组合根传
``server.warmup_status``）：它是**进程级设施状态**（后端预热到哪一步了），
不是库存事实——两者混在一个 dict 里返回是产品决策（少一次往返），但来源必须
看得见，故用构造注入显式标出（与 ``persist_dir`` 同类：环境值，不是领域值）。
"""
from __future__ import annotations

from typing import Any, Callable

from ..store import ConcurrentWriteError
from .base import BaseService


class MaintainService(BaseService):
    """维护用例：导出 / 未命中留痕 / 统计 / 落盘 / 重载（策略与算法暂居 brain，后续上收）。"""

    def __init__(self, *, brain: Any, library_self: dict[str, str],
                 persist_dir: str = "",
                 probe: Callable[[], dict[str, Any]] | None = None,
                 **base_kwargs: Any) -> None:
        """``probe`` = 零等待就绪探针（组合根注入 ``server.warmup_status``）。

        缺省 ``None`` 而非必填：本服务一出就拿 5 个工具用，只有 brain_stats
        真要探针。可选是为了让单测能最小构造（不带探针直接 new）；代价是缺
        探针时 ``brain_stats`` 的 ``warmup`` 节退化为 ``{}`` —— 组合根恒注入，
        这条退化路径在生产上不会被走出。

        ``**base_kwargs`` 透传基类（D2b 的 ``user_access_resolver`` 等可选
        注入由此到达，本服务自己不消费）。
        """
        super().__init__(brain=brain, library_self=library_self,
                         persist_dir=persist_dir, **base_kwargs)
        self.probe = probe

    # ---------------------------------------------------------------- 导出

    def export_memories(self, *, since_ts: float, source: str, state: str,
                        limit: int, client: str = "") -> dict[str, Any]:
        """按时间/来源/状态增量导出（不含向量，可直接落盘另存）。**不带**信封。

        边界是「含等于」（多条同时间戳时那批最后一条会重复一次，按 id 去重
        即可）——这条语义写在工具 docstring 里，服务侧只做编排，不重述不改写。

        D2b：目标是 user 库时按档位拒绝（设计稿 §4.2：导出是批量通道，
        masked 档「要么拒绝要么仅限非敏感」取拒绝——批量逐行滤的账算不过来，
        泄露面却一样大）。
        """
        tier = self._ua(client)
        if tier in ("masked", "none"):
            return {"error": (
                f"用户级库的导出在当前工作区被拒（读侧档位 {tier}）：导出是"
                "批量通道，masked/none 档一律拒绝。请在 home 工作区（日常"
                "助手宿主）会话里导出。")}
        rows = self.brain.export_memories(since=since_ts, source=source or None,
                                          state=state or None, limit=limit)
        return {"count": len(rows), "memories": rows}

    # ---------------------------------------------------------------- 留痕

    def miss_log(self, *, limit: int, clear: bool) -> dict[str, Any]:
        """查看/清空「召回不到」的查询留痕。**不带**信封。

        ``clear`` 走 ``brain.clear_miss_log()`` 而不是把 list 直接清空：
        清空是删除，得先立碑，否则别的进程的旧快照一落盘就把它并回来。
        """
        brain = self.brain
        if clear:
            return {"cleared": brain.clear_miss_log()}
        rows = sorted(brain.miss_log, key=lambda r: -r.get("ts", 0.0))
        return {
            "count": len(rows),
            "threshold": brain.miss_log_threshold,
            "enabled": brain.miss_log_enabled,
            "entries": rows[:limit] if limit > 0 else rows,
        }

    # ---------------------------------------------------------------- 统计

    def brain_stats(self, *, daily_days: int) -> dict[str, Any]:
        """四层 + 索引层规模统计，兼作后端就绪探针。**不带**信封。

        ``warmup`` 一节来自注入探针：只依赖库存层级（几秒应答），所以它
        同时回答「后端准备好没有」——``ready=false`` 时其余带向量的工具会
        返回「稍后重试」。
        """
        out = self.brain.stats(daily_days=daily_days)
        out["warmup"] = self.probe() if self.probe else {}
        return out

    # ---------------------------------------------------------------- 落盘 / 重载

    def save_brain(self) -> dict[str, Any]:
        """持久化整个记忆底座（与 save_state 等价）。**带**信封。

        ConcurrentWriteError 的 conflict 体与 WriteService.save_state 逐字同
        文案（hint 指向 load_state）——两个工具本来就是同一语义的两个入口。
        """
        try:
            self.brain.save()
        except ConcurrentWriteError as e:
            return self._envelope(
                {"status": "conflict", "detail": str(e),
                 "hint": "另一个进程改写了同一个记忆库。先 load_state 拿最新状态再重做本次修改。"})
        return self._envelope({"status": "saved", "persist_dir": self.persist_dir})

    def load_state(self) -> dict[str, Any]:
        """从磁盘恢复全状态，覆盖内存。**不带**信封。

        加载翻车时不静默：``store.last_load_issue`` 有值就把隔离文件与原因
        一并回给调用方（因为「loadsucceeded」和「加载但丢了一层」在返回体里
        不该长得一样）。
        """
        brain = self.brain
        brain.load()
        issue = brain.manager.store.last_load_issue
        payload: dict[str, Any] = {"status": "loaded", "stats": brain.stats()}
        if issue:
            payload["load_issue"] = issue
        return payload
