# -*- coding: utf-8 -*-
"""工具剖面门控（server.py 剖面门控区块的实现本体）。

这里实现：剖面状态（``_TOOL_PROFILE`` / ``_TOOL_GROUPS`` /
``_PROFILE_GROUPS`` / ``_COMMON_TOOLS_DEFAULT`` / ``_COMMON_TOOLS`` 与
``HIPPOCAMPUS_COMMON_TOOLS`` env 覆盖）、判定函数 ``_tool_enabled``、
``server.tool`` 的剖面感知包装工厂 :func:`make_gated_tool`。

🔴 现取纪律：``_tool_enabled`` / ``make_gated_tool`` 产物对门控状态的
读取一律 ``s._X``（server 命名空间的再导出绑定）——测试
``monkeypatch.setattr(server, "_TOOL_PROFILE", ...)" /
``server._TOOL_GROUPS = {...}" 照常生效（教训：把状态快照进新模块 =
monkeypatch 静默失效）。``_original_tool`` 由 server.py 在
``server.tool`` 替换**之前**赋值（现取源），包装工厂在**调用时**才读它。

剖面语义：
  默认 common：工具面 token 税是硬约束——full 面 46 个工具 ≈15.8k token
  常驻宿主上下文，而一线观察的日常会话只用到其中 6~9 个。未显式配置时只
  注册 _COMMON_TOOLS 的 10 条（+hippo_help/hippo_extra 两个派发器，共 12 个
  可见）；被隐藏的 36 个**一个不丢**，经 hippo_extra 派发照常可调。
  想要回全量 46 个：.mcp.json 的 env 里加 ``HIPPOCAMPUS_TOOL_PROFILE=full``。
  可选剖面（显式设置才生效；未设置/空串/纯空白都走默认 common）：
    common  —— 日常高频 10 条：读+写+会话开收尾+串库护栏；其余走 hippo_extra
    core    —— 仅检索/会话/库基（纯对话场景）
    writer  —— core + 写入/更新/遗忘/摄入
    lexicon —— core + 词库配置（低频）
    entity  —— core + 实体图
    admin   —— core + 运维/导出
    map     —— core + 信息地图层（map_set/get/list/verify）
    minimal —— 仅 core
    full    —— 全部 46 个（历史默认行为）
  未知剖面名 → 全量兜底：宁可多暴露也不误关，拼错名字的代价只是没省到 token。
"""
from __future__ import annotations

from typing import Any

from hippocampus_mcp.config import load_config

#: 默认 common（见模块 docstring）。
_TOOL_PROFILE = (load_config().tool_profile or "").strip().lower() or "common"

# 工具 → 所属剖面分组（未列出的工具按"保守保留"处理，见 _tool_enabled）
_TOOL_GROUPS: dict[str, str] = {
    # core
    "recall": "core", "get_memory_context": "core", "get_memory": "core",
    "list_memories": "core", "session_start": "core", "session_end": "core",
    "library_use": "core", "library_whoami": "core", "library_list": "core",
    "library_create": "core", "brain_stats": "core", "save_brain": "core",
    "knowledge_map": "core", "knowledge_topic": "core",
    # writer
    "store_memory": "writer", "update_memory": "writer", "forget_memory": "writer",
    "forget_by_query": "writer", "ingest": "writer", "ingest_batch": "writer",
    "list_sources": "writer", "update_tags": "writer",
    # lexicon
    "lexicon_add": "lexicon", "lexicon_suggest": "lexicon",
    "lexicon_alias_suggest": "lexicon",
    # entity
    "entity_profile": "entity", "entity_list": "entity", "relink_entities": "entity",
    "contradictions": "entity",
    # admin
    "export_memories": "admin", "reset_context": "admin", "miss_log": "admin",
    "consolidate": "admin", "report_feedback": "admin", "remember_intent": "admin",
    "complete_intent": "admin", "library_info": "admin",
    "get_stats": "admin", "save_state": "admin", "load_state": "admin",
    "retry_save": "admin",
    # 源漂移审计（只读）。归 admin 组 ⇒ common 剖面
    # 零 token 增加，full 剖面与 hippo_extra 都能触达。
    "drift_audit": "admin",
    # 信息地图层（状态寄存器）。归独立 map 组
    # ⇒ common 面零 token 增加，full 与 hippo_extra 可触达；显式设
    # HIPPOCAMPUS_TOOL_PROFILE=map 得 core+map 的窄面。
    "map_set": "map", "map_get": "map",
    "map_list": "map", "map_verify": "map",
}
# 每个剖面 = {core} ∪ 自身组；full 为 None（全部）
_PROFILE_GROUPS: dict[str, set[str] | None] = {
    "core": {"core"}, "minimal": {"core"},
    "writer": {"core", "writer"}, "lexicon": {"core", "lexicon"},
    "entity": {"core", "entity"}, "admin": {"core", "admin"},
    "map": {"core", "map"},
    "full": None,  # None = 全部
}

# 常用集（一线观察：zcode 日常实际用到的不超过 8 条）—— 要增删改这一行即可。
# 它**不是** {core} ∪ {writer} 的推演结果，而是独立维护的"日常高频"名单。
# 被它排除的工具并非不可用：隐藏的那部分由 hippo_extra 派发补回来。
_COMMON_TOOLS_DEFAULT = frozenset({
    # —— 读写主干 8 条（一线观察的日常面）——
    "recall", "store_memory", "ingest", "session_start",
    "get_memory_context", "update_memory", "forget_memory", "list_memories",
    # —— 串库护栏 ——
    # 工作区引导文件优先级高于 HIPPOCAMPUS_PERSIST_DIR，解析不明时只 warning 后
    # 回落默认库（对用户＝静默写错库）。所以"我在哪个库"必须能零门槛地问，
    # 否则"进会话先确认库"这条纪律会在 common 剖面下被打断。
    # 代价约 200 token，换的是不可静默写错库。
    "library_whoami",
    # —— 维护半环 ——
    # 原名单按"一线观察的日常 8 条"定，只覆盖了开场的 session_start，
    # 漏掉闭环的另一半：没有 session_end ⇒ 会话永不收尾——L2 摘要不写、
    # L3 画像不更新、consolidate 不触发，且**全程不报错**（静默缺席）。
    # 维护工具不进常驻面，agent 就只想不起来调它；这一条必须可见。
    "session_end",
})
_COMMON_TOOLS: frozenset[str] = _COMMON_TOOLS_DEFAULT
# 试错期不用改代码：HIPPOCAMPUS_COMMON_TOOLS="recall,store_memory,..." 直接覆盖常用集。
_env_common = load_config().common_tools.strip()
if _env_common:
    _COMMON_TOOLS = frozenset(x.strip() for x in _env_common.split(",") if x.strip())


def _srv() -> Any:
    """server 模块本体（现取入口，见模块 docstring 的现取纪律）。"""
    from hippocampus_mcp import server as _s

    return _s


def _tool_enabled(name: str) -> bool:
    """该工具在当前剖面下是否注册为 MCP 工具。

    门控状态经 :func:`_srv` 现取——patch ``server._TOOL_PROFILE``
    / ``server._COMMON_TOOLS`` / ``server._TOOL_GROUPS`` 立即生效。
    """
    s = _srv()
    if s._TOOL_PROFILE == "full":
        return True
    if s._TOOL_PROFILE == "common":
        return name in s._COMMON_TOOLS
    groups = s._PROFILE_GROUPS.get(s._TOOL_PROFILE)
    if groups is None:
        return True  # 未知剖面 → 全量兜底，避免误关
    grp = s._TOOL_GROUPS.get(name)
    if grp is None:
        return True  # 未归类工具 → 保守保留
    return grp in groups


def make_gated_tool() -> Any:
    """构造 ``server.tool`` 的剖面感知包装（server.tool 赋值前调用）。

    被关掉的 tool 不注册（函数定义仍在，可被内部 import 调用与
    hippo_extra 派发）。FastMCP 运行时派发走 ``_tool_manager.call_tool``，
    不会在运行期再调 ``self.tool``，因此包装是安全的。

    🔴 ``_original_tool`` 在**调用时**从 ``server._original_tool`` 现取
    （server.py 必须在 ``server.tool`` 替换之前赋好它）——测试替换
    ``server._original_tool`` 即可拦截注册行为。
    """
    def _gated_tool(structured_output=False, **kwargs):
        s = _srv()
        original = s._original_tool
        if callable(structured_output):  # 兼容裸 server.tool（无括号）写法
            return original(structured_output, **kwargs)

        def deco(fn):
            if not s._tool_enabled(fn.__name__):
                return fn  # 跳过 MCP 注册；函数定义仍在，可被内部 import 调用
            return original(structured_output=structured_output, **kwargs)(fn)

        return deco

    return _gated_tool
