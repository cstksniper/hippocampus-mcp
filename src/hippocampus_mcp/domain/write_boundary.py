# -*- coding: utf-8 -*-
"""跨任务协同的家库写边界：**纯判定**，零 I/O。

一句话语义：

    家库可写、他库只读、跨库写只走显式受管通道；
    ``library_use`` 只影响寻址（读落哪），**写永远钉在家库**。

本模块只做「两个路径是不是同一个库」「这次写入允不允许」这类**纯判定**，
与 :mod:`domain.scope_recall` / :mod:`domain.user_scope` 同口径：
单测可在零 I/O 下把边界行为钉死，策略变更不需要碰注册表与磁盘。
路径解析这些脏活留在组合根（``server_runtime/composition.py``），它把结果
喂进来——本模块**不 import library**（架构约束：domain 只准依赖 common）。

三个外来包袱，按只挪位置不改行为的纪律原样承接：

* **归一用 ``os.path.normcase``**（注册表同款口径，见
  ``library.registry`` 对盘符大小写的处理）：Windows 上 ``D:\\x`` 与
  ``d:\\x`` 是同一个库，不归一就会出现「两个 brain 指向同一份 memories.json」。
* **user 库是唯一角色豁免**（§2.4 过渡规则）：目标库是约定名
  ``user`` 的用户级库 ⇒ ``managed-write`` 放行；其他他库一律 read-only。
  豁免的是**通道角色**不是工具——U3 的敏感过滤后续加在这条路径上。
* **逃生门是三值判定的输入**：``escape_open=True`` ⇒ 一律放行
  （``HIPPOCAMPUS_CROSS_LIBRARY_WRITE=1``，运维搬库用），由组合根读 config
  喂进来，本模块不知道 env 的存在（domain 不许 import config，P6 台账纪律）。
"""

from __future__ import annotations

import os
import sys

__all__ = [
    "CrossLibraryWriteDenied",
    "MANAGED_WRITE",
    "READ_ONLY",
    "READ_WRITE",
    "USER_LIBRARY_NAME",
    "WRITE_CLASS_TOOLS",
    "boundary_error_message",
    "caller_tool_name",
    "classify_write_permission",
    "library_display",
    "same_library",
]

#: 用户级库的约定名。角色豁免按**清单名**判，不按 ``lib_`` id
#: 判——id 是建库时随机派生的，同一个用户级库换个部署就换 id，而名字是约定。
USER_LIBRARY_NAME = "user"

READ_WRITE = "read-write"
MANAGED_WRITE = "managed-write"
READ_ONLY = "read-only"


class CrossLibraryWriteDenied(ValueError):
    """写类工具试图写「既不是家库、也不是受管通道」的库时抛（§2.2）。

    继承 ``ValueError`` 是为了和 :class:`library.LibraryResolutionError`
    同族——两条都是「这个调用不能照原样执行」的显式拒绝，都不是系统故障。
    区分二者靠**类名**（谁都能看清是解析不明还是跨界），所以守卫文案
    必须走 :func:`boundary_error_message`，不能抛出光秃秃的一行。
    """

#: 写类工具单一真相源（实施时逐一复核过）。
#:
#: 复核是在 46 个工具上逐个跑的：先把「工具 → 用例工厂」映射全量导出
#: （AST 直读 ``server.py``），再逐个看它们在 application 层是否真的写盘。
#: 相比设计稿的 19 个，**复核新增 2 个**：
#:
#: * ``miss_log`` —— ``clear=True`` 走 ``brain.clear_miss_log()``
#:   （``maintain_service.py:70``），**清空=删除**，是真写，不是只读查看；
#: * ``reset_context`` —— 虽不删记忆，但会改他库 brain 的内存时序状态
#:   （``manager.reset_context()``），auto_save 开启时可能顺势落盘。
#:   这条按保守侧收：宁可让罕见的他库诊断调用显式报错，也不留一个能改他库
#:   状态的后门（“设计稿没点名”不是放任的理由）。
#:
#: 其余 12 个/未列入的 25 个工具是纯读或盘无关（``library_*`` 五个、
#: recall 面 11 个、 ``get_stats`` / ``brain_stats`` / ``export_memories`` /
#: ``map_get`` / ``map_list`` / ``map_verify`` / ``lexicon_suggest`` /
#: ``lexicon_alias_suggest`` / ``session_start``）。
#:
#: 表里的每个名字都必须是 ``server.py`` 里真实存在的工具体——由
#: ``tests/test_u4_write_boundary.py`` 的 AST 断言钉住（防止拼错名字后
#: 该工具静默失去守卫，也防止工具被摘掉后留下僵尸表项）。
WRITE_CLASS_TOOLS: frozenset[str] = frozenset({
    # —— Write 面 9 个（含两个持久化）——
    "store_memory", "update_memory", "update_tags",
    "forget_memory", "forget_by_query",
    "ingest", "ingest_batch",
    "save_state", "retry_save",
    # —— Organize 面 4 个 ——
    "consolidate", "relink_entities", "lexicon_add", "map_set",
    # —— Bootstrap 面 4 个 ——
    "session_end", "remember_intent", "complete_intent", "report_feedback",
    # —— Maintain 面 4 个（持久化类，§2.5 明言一并钉死）——
    "save_brain", "load_state", "miss_log", "reset_context",
})


def same_library(a: str, b: str) -> bool:
    """两条路径是不是同一个库（大小写不敏感的归一比对）。

    ``normpath`` 先吃掉 ``./`` 与重复分隔符，``normcase`` 再吃掉盘符/后缀
    大小写差异——「指定但解析不明一律报错」那条纪律管解析，这一条管
    **已解析出来的两条路径能否视为同一目标**。空串不兜底成 False-value
    暧昧：空路径两两不等（``"" == ""`` 的 True 会让空目标绕过守卫）。
    """
    if not a or not b:
        return False
    return os.path.normcase(os.path.normpath(a)) == \
        os.path.normcase(os.path.normpath(b))


def classify_write_permission(*, target_path: str, home_path: str,
                              target_name: str = "",
                              escape_open: bool = False) -> str:
    """本次写入落到 ``target_path`` 的权限档。三值语义见 §2.4。

    Args:
        target_path: 解析后的目标库绝对路径（必填；空 = 解析不明，由调用方
            在更上游就抛 ``LibraryResolutionError``）。
        home_path: 家库绝对路径（连接默认库，``default_library_path()``）。
        target_name: 目标库的清单名（供 user 角色识别，可空）。
        escape_open: 逃生门是否开着（``HIPPOCAMPUS_CROSS_LIBRARY_WRITE=1``）。

    Returns:
        ``read-write``（家库）/ ``managed-write``（user 受管通道）/
        ``read-only``（其他他库 ⇒ 调用方须显式报错）。

    判定顺序是不能调的：**逃生门 → 家库 → user 角色 → 兜底 read-only**。
    逃生门在最前，因为它就是「整片放开」的运维开关；家库先于 user，因为
    当 user 库恰好就是家库时应报 ``read-write`` —— 那就是一次本库写，不该
    显示成受管通道。
    """
    if escape_open:
        return READ_WRITE
    if same_library(target_path, home_path):
        return READ_WRITE
    if (target_name or "").strip() == USER_LIBRARY_NAME:
        return MANAGED_WRITE
    return READ_ONLY


def library_display(name: str, path: str) -> str:
    """给报错文案用的一句话指代：有名就带名，没名至少给绝对路径。

    .path 永远带上——同名的两个库（不同路径）是常态，只给名字会让
    「目标库是谁」这个问题重新变得含糊。
    """
    name = (name or "").strip()
    return f"{name}（{path}）" if name else (path or "(unknown)")


def boundary_error_message(*, tool: str, target_display: str,
                           home_display: str) -> str:
    """拦下来的报错文案（§2.2：必须说清目标库是谁、家库是谁、怎么办）。

    三件事缺一不可——只说「不允许」的报错会让 agent 去试别的参数碰运气，
    而串库事故的代价是数据进错库后很难追回。
    """
    return (
        f"{tool} 的写入被家库写边界拦下：目标库「{target_display}」不是本次"
        f"连接的家库（home，当前为「{home_display}」）。\n"
        f"library_use 只影响读落在哪，**写永远钉在家库**；跨库写只走受管"
        f"通道（user 库受管提升 / 目录登记 / 进程内部动作）。\n"
        f"要写这个库：去它自己的工作区开会话（那里的家库就是它），或联系"
        f"用户开逃生门 env HIPPOCAMPUS_CROSS_LIBRARY_WRITE=1（运维搬库用，"
        f"默认关）。"
    )


def caller_tool_name(depth: int = 2) -> str:
    """取调用链上第 ``depth`` 帧的函数名（供守卫判断「谁在写」）。

    ``depth=3`` 对应「工具体 → 用例工厂 → 守卫 → 本函数」这条四帧链条
    （0=本函数），与设计 §2.2「守卫放服务工厂层，不在各工具体里铺开」
    配套：四个写类相关工厂各自调一次守卫即可，21 个工具体一行不改。
    注意守卫是**多出来的一帧**——写成 2 就退到工厂名，见
    ``tests/test_u4_write_boundary.py`` 的深度断言。

    取不到（栈没那么深，例如脚本直接调工厂）时返回 ``""``——组合根据此
    放行，因为那种调用本来就不是从工具面进来的。这个 fail-open
    **由 AST 断言代为看守**（详见 :data:`WRITE_CLASS_TOOLS` 注释）：只要
    M4 的「单一调用路径」纪律在（工具体直调工厂、中间不套壳），
    ``depth=2`` 就恒等成立。
    """
    try:
        return sys._getframe(depth).f_code.co_name
    except ValueError:  # pragma: no cover —— 栈没那么深
        return ""
