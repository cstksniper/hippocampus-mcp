# -*- coding: utf-8 -*-
"""用例服务基类：组合根注入的 runtime + 库自曝 + 统一信封。

注入面（刻意最小化，按需扩大）：

* ``brain``     —— 目标库的 HippoBrain 实例（``Any``：本包不 import
  brain，见包 docstring 纪律 1）。它目前仍是「每库门面」；方法体将
  逐个搬进各 Service，此注入对象随之逐步瘦身为纯端口组合。
* ``library_self`` —— 库自曝 ``{id, name, path}``，由组合根用适配层
  的解析结果算好传入（服务不碰库发现/会话粘性那套连接态状态）。
* ``persist_dir``  —— 组合根读一次的配置值（save_state 回显用）。
  不传整个 Config：用例要什么注入什么，YAGNI。

``manager`` 不是第二个注入项——它就是 ``brain.manager``（server 的
``_manager_for`` 一直是这个等式），在此派生只为少打一层。
"""
from __future__ import annotations

from typing import Any


class BaseService:
    """用例服务基类。所有方法返回**待序列化的 dict**（信封已装好）。"""

    def __init__(self, *, brain: Any, library_self: dict[str, str],
                 persist_dir: str = "",
                 user_brain: Any = None,
                 user_library_self: dict[str, str] | None = None,
                 sensitive_words: tuple[str, ...] = (),
                 promote_min_sessions: int = 2,
                 long_content_policy: str = "",
                 prospective_trigger: bool = True,
                 split_facts: Any = None,
                 recall_content_max_chars: int = 0,
                 recall_fusion_enabled: bool = True,
                 recall_query_fanout_enabled: bool = False,
                 user_write_quarantine: bool = False,
                 is_home_write: bool = True,
                 access_log_dir: str = "",
                 extra_libraries: list[tuple[dict[str, str], Any]] | None = None,
                 unregistered_names: list[str] | None = None,
                 user_access_resolver: Any = None) -> None:
        self.brain = brain
        self.manager = brain.manager
        self.library_self = library_self
        self.persist_dir = persist_dir
        # 跨库检索的用户库第二 brain。只有
        # ``_recall_service(scope=user/all)`` 工厂会注入（组合根解析约定名
        # user 的库并经注册表构造——显式检索意图付首访加载成本）；
        # 其余服务不传，None = 本服务没有跨库能力（用不到就别碰）。
        self.user_brain = user_brain
        self.user_library_self = user_library_self
        # 敏感双门禁 + 提升稳定期限。application 不许
        # import config（依赖纪律）⇒ 由组合根（server 工厂）每次现读
        # 注入：词表**内容**（服务层读文件）与 min_sessions——与
        # load_config 不缓存语义一致。不用这两个旋钮的服务不受影响。
        self.sensitive_words = sensitive_words
        self.promote_min_sessions = promote_min_sessions
        # 超长内容处置策略（warn/split/reject，默认 warn = 现状）。
        # application 不读 config ⇒ 由组合根现读注入（同 sensitive_words）。
        self.long_content_policy = long_content_policy or ""
        # 前瞻记忆情境触发开关（get_memory_context
        # 的 block_2 回带）。同上由组合根现读注入；默认 True = 生效
        # （登记侧触发集派生在 brain.track_intent，brain 侧直读 config）。
        # 只有 RecallService 用这个旋钮，其余服务不受影响。
        self.prospective_trigger = bool(prospective_trigger)
        # recall 单条 content 的字符上限（截断护栏）。0 = 不截断（现状）。
        # application 不读 config ⇒ 由组合根现读注入（同 long_content_policy）；
        # 只有 RecallService.recall 用，其余方法零变化。
        self.recall_content_max_chars = int(recall_content_max_chars or 0)
        # R1: production recall shares the evaluation scoring pipeline.
        # application must not read config -> the composition root
        # injects load_config().recall_fusion on every call.
        self.recall_fusion_enabled = bool(recall_fusion_enabled)
        # V2: query fanout (keyword variant + RRF merge) for natural-sentence
        # queries. Default off — injected by the composition root the same
        # way; only RecallService._recall_one consumes it.
        self.recall_query_fanout_enabled = bool(recall_query_fanout_enabled)
        # 分句器（brain._split_facts 本体，含「短句并入上一条」等怪癖）。
        # domain 不许另写一份分句真相 ⇒ 只能注入；None 时按长度硬切并自曝。
        self.split_facts = split_facts
        # D2a 写侧隔离三件（组合根工厂现算现注入，application 不判定）：
        # user_write_quarantine —— 本次写目标 = user 库且非家库且逃生门关
        #   （domain/write_quarantine.should_quarantine 的结果）：store/ingest
        #   落卡带 quarantine 标记 + 追加审计事件；
        # is_home_write —— 目标即家库（或逃生门开）：update_memory 的
        #   clear_quarantine 只对它开放（提权键不在 agent 手里）；
        # access_log_dir —— 审计事件落哪个库目录（目标 = user 库时才有值）。
        self.user_write_quarantine = bool(user_write_quarantine)
        self.is_home_write = bool(is_home_write)
        self.access_log_dir = access_log_dir or ""
        # B1 多库检索（只有 _recall_service 工厂会注入，其余服务 None）：
        # extra_libraries —— 外围库的 [(library_self, brain)]（组合根已解析，
        #   解析失败在工厂显式报错，服务层不再寻址）；
        # unregistered_names —— libraries= 值里**不在任务目录**的名字
        #   （软校验口径：三态寻址可解析即可用，目录外值信封自曝不拦截）。
        self.extra_libraries = list(extra_libraries or [])
        self.unregistered_names = list(unregistered_names or [])
        # D2b 读侧档位（只有 recall/maintain 工厂会注入）：callable(client) →
        # "full"/"masked"/"none"；非 user 库目标返回 ""（无档位语境）。
        # 判定真相在 server._user_access_tier（连接级绑定状态在那边）。
        self.user_access_resolver = user_access_resolver

    def _ua(self, client: str | None) -> str:
        """当前调用的 user 库读侧档位。resolver 缺席 = 非用户库语境（""）；
        resolver 调用失败 = 无法证明有资格 → fail-closed 判 none。"""
        if self.user_access_resolver is None:
            return ""
        try:
            return str(self.user_access_resolver(client) or "")
        except Exception:  # noqa: BLE001 —— 提权方向永不自动放行
            return "none"

    def _envelope(self, payload: Any) -> Any:
        """给写类返回体插 ``library`` 首键（库自曝）。**只有 dict 才插**。

        非 dict 返回（错误分支的字符串、列表等）原样返回 —— 不加键，
        免得把错误路径的结构改坏。与 server.py ``_with_library`` 逐位一致
        （响应组装归用例层，工具只序列化）。
        """
        if isinstance(payload, dict):
            return {"library": self.library_self, **payload}
        return payload
