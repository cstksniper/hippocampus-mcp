# -*- coding: utf-8 -*-
"""MCP Server 入口.

启动一个符合 Model Context Protocol 标准的 stdio 服务端,
为 AI Agent 提供记忆增强能力.

暴露的 MCP 工具:
    - store_memory:    存储新记忆 (文本 → 嵌入 → LNN → 向量库)
    - recall:          回忆相关记忆 (LNN 查询 + 打分 + 排序)
    - get_memory:      按 ID 获取单条记忆
    - update_memory:   更新记忆字段
    - list_memories:   列出所有记忆
    - reset_context:   重置 LNN 上下文 (新对话)
    - get_stats:       获取系统状态
    - save_state:      持久化所有状态
    - load_state:      从磁盘恢复状态

启动方式:
    python -m hippocampus_mcp.server

    或在 MCP 客户端配置中指定:
    {
        "command": "python",
        "args": ["-m", "hippocampus_mcp.server"],
        "env": {"HIPPOCAMPUS_PERSIST_DIR": "/path/to/data"}
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

# 库发现层与多库注册表（**均为纯标准库**，不碰 torch，
# 所以可以安全地放在顶层导入，不会把加载时间顶到 MCP 握手路径上）。
from hippocampus_mcp import library as _library
from hippocampus_mcp.library import LibraryResolutionError  # noqa: F401（工具签名要抛）
from hippocampus_mcp.library.registry import BrainRegistry, DEFAULT_KEY
# 任务目录 sidecar 的读写原语（library 层，纯标准库）。
# 放在这里而不是组合根：组合根有「顶层 import 白名单」（仅
# logging/os/typing/application），而登记点只需经 _LIB_PORTS 的活引用
# ``_task_directory_register`` 被调用——与家库写边界判定同一手法。
from hippocampus_mcp.library import task_directory as _task_directory
# session_key 随多库寻址区在 server_runtime/routing.py（其内部
# 用的 session_ctx 也随迁），本行别名成为死导入，移除——f9_4e/f9_5a 一律
# 直接 ``from hippocampus_mcp.common import session_ctx``（全仓无
# ``S._session_ctx`` 引用，无 patch 面损失）。
from .config import load_config
# 门控/派发/头部/入口在 profiles.py / dispatch.py /
# stdout_guard.py / entry.py。状态经再导出留在本模块命名空间（patch 面
# 不变），四个模块一律经 _srv() 现取——_dispatch_map 经 vars(server)
# 现取工具函数（快照替身会吞新注册的工具，目标测试 N1 钉死）；
# _gated_tool 的 _original_tool 在 server.tool 替换前赋值、调用时现取；
# main 留壳调 entry.run_main()（cli/launcher 的 import 零改动）。
from .server_runtime import entry as _entry
from .server_runtime import profiles as _profiles  # noqa: F401（工厂宿主）
from .server_runtime import stdout_guard as _sg
from .server_runtime.dispatch import (_dispatch_map,  # noqa: F401（再导出）
                                      _hippo_extra, _hippo_help)
from .server_runtime.profiles import (_COMMON_TOOLS,  # noqa: F401（再导出）
                                      _COMMON_TOOLS_DEFAULT,
                                      _PROFILE_GROUPS, _TOOL_GROUPS,
                                      _TOOL_PROFILE, make_gated_tool)
from .server_runtime.profiles import _tool_enabled  # noqa: F401（再导出）
from .server_runtime.stdout_guard import (_ThreadAwareStdout,  # noqa: F401（再导出）
                                          configure_logging, ensure_paths)

# sys.path 兜底（libs append + src insert）在
# server_runtime/stdout_guard.py，传入本文件路径（原 ``__file__`` 语义）。
_sg.ensure_paths(__file__)

try:  # mcp >= 1.0（新版 SDK）
    from mcp.server.fastmcp import FastMCP as MCPServer
except ImportError:  # pragma: no cover - 老版本 SDK
    try:
        from mcp.server import MCPServer  # type: ignore
    except ImportError:
        from mcp.server.lowlevel import Server as MCPServer  # type: ignore

# 重型依赖（``brain`` → ``manager`` → ``torch``，本机热态就要 5.7s）**故意不在此处导入**：
# 顶层每多 1 秒，MCP 的 ``initialize`` 就晚 1 秒应答，直接吃掉宿主的工具注册窗口
# （约 15s）。改成在 ``get_brain()`` 里按需导入 —— auto 模式下首次导入发生在
# 预热线程（事件循环已启动），实测该路径不会触发 scipy 那类 LoadLibrary 卡死
# （对照见 ``_preload_import_chain`` 的说明与 diag 脚本）。
if TYPE_CHECKING:  # 仅供静态检查，运行时不执行（注解已被 future import 字符串化）
    from hippocampus_mcp.brain import HippoBrain
    from hippocampus_mcp.engine.manager import MemoryManager

from hippocampus_mcp.engine.embeddings import ENV_MODEL  # noqa: F401（_EMB_MODEL 走 config；保留再导出防包外断裂）
# 包外请从 ``hippocampus_mcp.store`` 取 ConcurrentWriteError，
# 不要从 server 符号导入。

# 参数解析助手与键名校验在 common/params.py 与 application/tags.py——
# 工具层与用例层共用同一份实现，这里以**别名 import** 保留调用点名字
# 不变；服务层不 import 本模块（依赖方向纪律）。update_tags 的四段解析
# 走 parse_tagops_args（逐段严格解析、第一处错误即停，消息文本逐位
# 不变）；session_start 仍用 ``parse_tags_json``。严格版与 lenient 版
# 两个名字各归其主，避免相互遮蔽。
from .application import (BootstrapService, LibraryAdminService,
                          LibraryPorts, MaintainService, OrganizeService,
                          RecallService, WriteService)
from .common.params import parse_str_list_lenient as _parse_str_list
from .common.params import parse_tagops_args as _parse_tagops_args
from .common.params import parse_tags_json as _parse_tags_json
# 预热区（后台线程/工具侧等待/就绪探针/看门狗/预载/层级表/
# 异常类/torch 兜底）整体拆去 server_runtime/warmup.py。**状态与模式常量
# 刻意留在本模块**（测试对本命名空间的 monkeypatch 面：_WARMUP_MODE /
# _warmup_ready / _warmup_started / _warmup_error / _warmup_fatal /
# _warmup_phase / _store_ready / _WARMUP_WAIT_* / _warmup_libs … 全部经
# warmup._srv() 现取生效，见该模块 docstring）。此处按 import 名原样
# 再导出——46 个工具体的 await _await_warmup() 字面、srv.STORE_ONLY_TOOLS /
# srv.WarmupPendingError 等既有引用零改动。
from .server_runtime.warmup import (STORE_ONLY_TOOLS,  # noqa: F401（再导出）
                                     WarmupFailedError, WarmupPendingError,
                                     _await_warmup, _preload_import_chain,
                                     _prepare_torch, _start_warmup_watchdog,
                                     maybe_warmup_library,
                                     start_background_warmup, warmup_status)

# 多库寻址（引导文件/默认库解析/会话 key/库清单/
# 三级优先/粘性绑定/whoami 自曝/信封装配/端口 I/O 助手）与组合根（_LIB_PORTS
# + 6 个用例工厂 + _build_brain）拆去 server_runtime/routing.py +
# composition.py。**状态与 patch 面刻意留在本模块**（_DEFAULT_LIB_PATH /
# _conn_* / _library_cache* / _registry 三件 / _brain_for / get_brain /
# _brain / _PERSIST_DIR / 配置常量），两个新模块一律经 _srv() 现取（见
# 各自 docstring）。此处按 import 名原样再导出——工具体调用点、
# m4_batch1/4 的 hasattr、f10/f9_4e/workspace_guide 的直调零改动。
from .server_runtime.routing import (GUIDE_FILENAME,  # noqa: F401（再导出）
                                     _ensure_default_dir,
                                     _find_workspace_guide, _layer_files,
                                     _library_binding,
                                     _library_binding_lock, _library_self,
                                     _prefetch_library,
                                     _resolve_default_lib_path, _with_library,
                                     clear_library_binding,
                                     resolve_default_lib_ex,
                                     get_library_binding, libraries,
                                     resolve_library_path, session_key,
                                     set_library_binding)
from .server_runtime.composition import (_LIB_PORTS,  # noqa: F401（再导出）
                                         _bootstrap_service, _build_brain,
                                         _maintain_service, _organize_service,
                                         _recall_service, _write_service)
# 家库写边界的**纯判定**（写类工具清单 / 归一比对 /
# 三值权限 / 拒绝文案）在 domain/write_boundary.py——放那儿是因为它能零 I/O
# 单测，边界行为的每一次调整都不必碰注册表与磁盘。
# 为什么不直接 import 进 composition.py：组合根有 S4.2「顶层 import 无越层」
# 断言（白名单仅 logging/os/typing/application），加了就翻这条钉子。**本模块
# 先 import，组合根再经 _srv() 现取**——与 U2 合流时 ``s._library`` 的手法同款，
# monkeypatch 面一路穿透到守卫。
from .domain.write_boundary import (CrossLibraryWriteDenied,  # noqa: F401（再导出）
                                    READ_ONLY, USER_LIBRARY_NAME,
                                    WRITE_CLASS_TOOLS, boundary_error_message,
                                    caller_tool_name, classify_write_permission,
                                    library_display, same_library)
# D2a 写侧隔离的纯判据同款手法：domain 真相先 import 进本模块，
# 组合根经 _srv() 现取（S4.2 白名单不放 domain import，monkeypatch 穿透）。
from .domain.write_quarantine import should_quarantine  # noqa: F401（再导出）
from .common.session_task import bind_task, clear_task, current_task
from .common.write_origin import build_origin
# C1 工作区绑定反查的纯判定（domain/task_directory.resolve_workspace）
# + D2b 读侧档位判定（resolve_user_access）。
from .domain.task_directory import (normalize_entries, resolve_user_access,
                                    resolve_workspace)
# B1 libraries= 语法解析（组合根经 _srv() 现取，S4.2 白名单同款手法）。
from .domain.scope_recall import parse_library_spec  # noqa: F401（再导出）
# 注：_library_admin_service 留在本模块（见其 docstring 的说明），
# 不在 composition 再导出名内。

# ------------------------------------------------------------------
# 日志配置（handler/级别装配在 stdout_guard.configure_logging，
# 逐位一致；_ThreadAwareStdout 类同迁、按 import 名再导出）
# ------------------------------------------------------------------
logger = _sg.configure_logging()


# ------------------------------------------------------------------
# 全局配置（唯一的记忆管理器在下方 get_brain() 中惰性创建）
# ------------------------------------------------------------------
_PERSIST_DIR = load_config().persist_dir

# 评测最优配置 (LoCoMo 调优结果)
# 注：嵌入维度与神经元数已由嵌入模型/默认值决定（HIPPOCAMPUS_EMB_DIM /
# HIPPOCAMPUS_N_NEURONS 不再生效，保留仅为向后兼容旧配置文件）。
_LAMBDA_DECAY = load_config().lambda_decay  # 90天
_MU_IMPORTANCE = load_config().mu_importance  # 180天
_W_SIM = load_config().w_sim
_W_TIME = load_config().w_time
_W_RECALL = load_config().w_recall
_CRL_MODEL_PATH = load_config().crl_model
_EMB_MODEL = load_config().emb_model  # 如 BAAI/bge-base-zh-v1.5（留空则跟随库内模型）
# 每 10 次写入自动落盘（覆盖报告·追加 4）。
# **MCP 场景下默认开**：宿主进程随时可能被回收（实测 ingest 2 条 → 进程退出 →
# 磁盘条数回退），而写入是稀疏的（对话级），代价摊薄后每条约 +0.15s
# （大库实测全量落盘也在秒级，不会挡住会话）。
# 库层（HippoBrain/MemoryManager）默认仍是 False——评测与批量导入不该被
# 写盘打断；``ingest_batch`` 也会在批内临时关掉它。
_AUTO_SAVE = load_config().auto_save
# 启动时自动补算缺失向量。库里向量与当前模型不符时（例如换模型后首次启动），
# 补算是分钟级且同步阻塞，MCP 客户端会等到超时——大库请先离线跑
# scripts/reembed.py，再设 HIPPOCAMPUS_AUTO_BACKFILL=0 启动服务。
_AUTO_BACKFILL = load_config().auto_backfill
# 导航类内容（INDEX / 目录 / 表头）的召回折扣，1.0 = 不降权（追加八）。
# 只有导入端用 ``import_md.py --nav-files`` 给那些记忆打上 ``_nav`` 标签后
# 才有对象可降——两头都得开着，少一头则静默无效（导入侧会打印它用到的值）。
_NAV_PENALTY = load_config().nav_penalty
# 事实更新检测开关（批量导入文档的场景应关闭）
_SUPERSEDE = load_config().supersede

# ------------------------------------------------------------------
# 预热策略（追加 10：握手不能被预热阻塞）
# ------------------------------------------------------------------
# 背景：预热＝加载全库记忆 + 嵌入模型 + spaCy，实测 30-36 秒。原先放在
# ``server.run()`` **之前**同步执行，等于把 MCP 握手一起堵住了——宿主侧
# 会话启动后约 15 秒就完成工具注册，结果本服务的工具一次都没注册上，
# 表现为「连接状态成功，但会话里没有工具」（ZCode 实测）。
#
# 三种模式（环境变量 HIPPOCAMPUS_WARMUP）：
#   auto（默认）　起后台线程预热，握手立即应答；首个工具调用若预热尚未完成，
#                 在 worker 线程里等待（不占事件循环，不阻塞其他 MCP 消息）。
#                 ➜ 宿主能注册到工具；最坏情况下把等待挪到第一次调用，且只等一次。
#   off　　　　　 完全不预热，握手立即应答；首次工具调用时惰性初始化（约 25-30 秒）。
#   sync　　　　  回归旧行为：预热完才握手。仅离线脚本 / 评测用，**不要给宿主**。
#
# 注意：``auto`` 下工具调用仍要保证「预热完成才执行」，否则第一次调用会与
# 预热线程并发构造 HippoBrain——见 ``get_brain`` 的单例锁。
_WARMUP_MODE = load_config().warmup.strip().lower()
if _WARMUP_MODE not in ("auto", "off", "sync"):
    logger.warning("Unknown HIPPOCAMPUS_WARMUP=%r; falling back to auto", _WARMUP_MODE)
    _WARMUP_MODE = "auto"
# 工具调用等待后台预热的上限。
#
# **这个值必须小于宿主的调用超时**（ZCode 是 120s），否则"等"与"挂死"在宿主
# 侧长得一模一样：调用方先超时，拿到的是一条超时错误，而不是"后端仍在预热"。
# 原默认 300s 正是"冷启动期并发首调撞 120s 超时（重试即好）"的直接成因。
_WARMUP_WAIT_TIMEOUT = load_config().warmup_timeout

# 超预算后的处置策略：
#   raise（默认）　抛出 WarmupPendingError。宿主拿到的是**可解析的明确状态**
#                 （已等多久、卡在哪个阶段、可以重试），而不是一条超时。
#                 并发首调也自然去重：每个等待者都在同一个 Event 上等，
#                 不会各自触发一次加载（惊群）。
#   release       　旧行为：放行，转由惰性初始化接手（慢，但"一定会执行"）。
#                 只有明确知道模型加载就是很慢、且宿主超时足够长时才用。
_WARMUP_WAIT_POLICY = load_config().warmup_wait.strip().lower()
if _WARMUP_WAIT_POLICY not in ("raise", "release"):
    logger.warning("Unknown HIPPOCAMPUS_WARMUP_WAIT=%r; falling back to raise", _WARMUP_WAIT_POLICY)
    _WARMUP_WAIT_POLICY = "raise"

# 进程起点时间戳：自曝「spawn → 站到 server.run() 之前」的**真实**耗时。
# 要区分「本端慢」与「宿主聚合时机」，靠 _HANDSHAKE_BASE_COLD 估算不算证据。
_PROCESS_T0 = time.time()

# 宿主留给「握手 + 工具注册」的窗口（秒），ZCode 实测约 15s。
# 站到 server.run() 之前的耗时超过它，本服务的工具就可能被整个漏注册。
_HANDSHAKE_WINDOW = load_config().handshake_window

_warmup_ready = threading.Event()   # 完整就绪（含嵌入模型）
_store_ready = threading.Event()    # 库存就绪（不含嵌入模型，通常早 20-60 秒）
_warmup_started = False  # 只有 main() 真正起过后台预热，工具才需要等待
# 预热失败的**原因**（空 = 没失败）。失败要能立即说清楚，而不是让调用方
# 等满超时再收到一句"超时"——那会把"配置问题"伪装成"性能问题"。
_warmup_error: str = ""
_warmup_phase: str = ""            # 当前/最后进入的阶段名（超时消息里带上）
# 失败是否**致命**（核心层没建起来）。区分致命与非致命很关键：
#   · 致命（构造 HippoBrain 就失败）→ 后续任何工具都不可能成功，必须当场报错；
#   · 非致命（spaCy 加载失败 / 首次推理失败）→ 核心仍可用，应当放行，
#     只把原因留在 ready 探针里。一刀切抛错会把"降级可用"误判成"服务挂了"。
_warmup_fatal = False
# 诊断开关：预热迟迟不完成时，按 30s/60s/120s … 打印所有线程的 Python 栈。
# 背景：模型构造在**后台线程**里实测比在主线程慢一个数量级（29s → >250s），
# 而"卡住"和"慢"在宿主侧看起来一样（工具调用无响应）——没有栈就只能猜。
# 默认关闭，出问题时设 HIPPOCAMPUS_WARMUP_DIAG=1 重启即可拿到现场。
_WARMUP_DIAG = load_config().warmup_diag

# 预热异常类（WarmupPendingError / WarmupFailedError）、torch 单线程兜底
# （_prepare_torch）与预热区全部逻辑在
# server_runtime/warmup.py（顶部 import 再导出，命名空间不变）。
# 原「单例构造锁 _INIT_LOCK」随多库注册表（每库独立锁）退役，
# 全仓已无引用，一并清除。

# 刻意留下：_brain_class 是 ``_build_brain``
# 的类解析点（``globals().get("HippoBrain")`` 语义，测试的
# ``server.HippoBrain = Fake`` 替换靠它生效）；get_manager 是
# test_warmup / test_recall_consistency / f10 的直调面；
# _PRELOAD / _PRELOAD_BUDGET / _HANDSHAKE_BASE_COLD 由 warmup.py 经
# _srv() 现取（档位语义与实测数据见 warmup.py 内注释）。


def _brain_class() -> Any:
    """解析 ``HippoBrain`` 类：**模块属性优先**，其次才真正导入。

    为什么属性优先：本模块不再在顶层导入 ``brain``（那会把 torch 的 ~5.7s 顶到
    MCP 握手路径上）。但测试与诊断脚本需要替换实现（``server.HippoBrain = Fake``，
    见 ``scripts/test_warmup.py`` 的并发单例用例），所以保留"先看模块属性"的语义。
    未见替换时才做真正的导入（此后 ``sys.modules`` 已缓存，重复调用几乎零成本）。
    """
    cls = globals().get("HippoBrain")
    if cls is not None:
        return cls
    from hippocampus_mcp.brain import HippoBrain as _cls

    return _cls


def get_manager() -> MemoryManager:
    """获取全局 MemoryManager 单例。

    **v0.4 起与 v0.3 用层共用同一个实例**（即 ``get_brain().manager``）。
    此前本函数与 ``get_brain()`` 各自 ``new`` 一个 MemoryManager，却指向
    同一个 ``memories.json``：结果是 ingest 写入的记忆 recall 搜不到，
    并且任一侧 ``save`` 都会整体覆写另一侧（实测互删记录、数据丢失）。
    现在整个进程只有一份内存状态、一份存储。

    配置来源统一为环境变量（见 ``get_brain``），旧路径不再单独持有参数。
    """
    return _brain_for(None).manager


# 预载档位值本体（``HIPPOCAMPUS_PRELOAD`` 解析结果）与预算：无 patch 面但属
# 头部配置区，server_runtime/warmup.py 的 _preload_list / _preload_import_chain
# 经 _srv() 现取（档位语义、为什么不能再收窄与实测数据见 warmup.py 内注释）。
_PRELOAD = load_config().preload.strip()

# 站内默认可接受的预载耗时上限（秒）：超过它握手就可能错过宿主注册窗口。
# 用于在日志里给出明确警告，而不是让运维自己去算。
_PRELOAD_BUDGET = load_config().preload_budget

# "spawn → 预载开始"这段固定开销（解释器启动 + 顶层 import），冷启动实测约 5.7s
# （其中 import mcp 1.6s）。torch 不放顶层导入（预热区才加载），否则这段是 11.4s。
# 只用于把"预载耗时"换算成"握手总耗时"写进告警，方便运维判断会不会超窗口。
_HANDSHAKE_BASE_COLD = 5.7


# ------------------------------------------------------------------
# 创建 MCP Server 并注册工具
# ------------------------------------------------------------------
#: 一次性递送的全局说明（会话级常驻，**不要**往里塞每次调用都要重复的内容）。
#: 为什么要放在这儿：`session_start` 每次返回都要付 token，而"压缩后怎么恢复"
#: 这类说明对宿主是**一次性**的——挪到 instructions 只付一次。返回体
#: `receipt.resume_hint` 只留一句指向本工具的指针（键位不动，形状探针仍绿）。
_INSTRUCTIONS = (
    "This service is a memory layer for AI agents. "
    "Write with ingest, read with get_memory_context, raw retrieval with recall, "
    "open sessions with session_start and close them with session_end. "
    "If tools such as remember_intent / consolidate are not in this session's tool list, "
    "call hippo_help for the catalog, then invoke via hippo_extra(action=\"name\", args=\"{...}\") -- no capability is lost. "
    "If the host context was compacted: do not copy memory content; keep only the identity fields from the session_start "
    "receipt (session_id / assembled_at / blocks / memory_active / "
    "knowledge_epoch) and call session_start once more with session_id to rebuild the opening pack;"
    "If memory_active or knowledge_epoch differs from before going offline, the library was changed by another session; trust this delivery."
)


def _create_server() -> Any:
    """兼容不同版本的 mcp SDK 构造参数。"""
    try:
        return MCPServer(
            name="hippocampus-mcp",
            version="0.1.0",
            description="Agent memory layer: four-layer memory + entity index + session lifecycle",
            instructions=_INSTRUCTIONS,
        )
    except TypeError:
        # FastMCP（mcp >= 1.0）：只接受 name/instructions
        return MCPServer(name="hippocampus-mcp", instructions=_INSTRUCTIONS)


server = _create_server()


# ------------------------------------------------------------------
# 工具剖面门控（状态/判定/包装在 profiles.py，经 _srv() 现取）
# ------------------------------------------------------------------
# 剖面语义（D 方案）、_TOOL_GROUPS 分组表、常用集与 HIPPOCAMPUS_COMMON_TOOLS
# 覆盖、_tool_enabled 判定与 _gated_tool 包装工厂的完整说明见
# server_runtime/profiles.py；状态经再导出留在本模块命名空间（上方 import
# 块），测试对本命名空间的 patch 面不变。_original_tool 必须在
# server.tool 替换之前赋值（_gated_tool 调用时现取它）。
_original_tool = server.tool
_gated_tool = make_gated_tool()
server.tool = _gated_tool


@server.tool(structured_output=False)
async def store_memory(
    content: str,
    importance: float = 0.5,
    memory_type: str = "general", library: str = "",
    idem_key: str = "",
) -> str:
    """Store one fact without raw-log provenance or entity linking.
For long text or conversation transcripts, prefer ``ingest``.

Returns include truncation self-report: embeddings have a 512-token hard limit.
If ``truncated=true``, only the first ``embedded_chars`` characters contributed to
the vector; the remaining text is split into additional <500-character cards.

Args:
    content: Memory text (required).
    importance: 0.0 to 1.0; default 0.5.
    memory_type: episodic / semantic / procedural / general; default general.
    idem_key: Idempotency key. Retrying the same request key does not create a
        duplicate or re-embed; it returns the existing ``memory_id`` with
        ``status=duplicate``. Key by request, not by content hash.
    """
    await _await_warmup()
    svc = _write_service(library)
    return json.dumps(svc.store_memory(
        content=content, importance=importance, memory_type=memory_type,
        # 接线见 WriteService.store_memory：client 决定 LNN 隐状态
        # 落在哪一路（daemon 下 N 条连接不能全挤在 __default__）。
        client=session_key(),
        # P0-1 写入凭据：谁在什么写入面写的（只写不筛，不进检索）。
        origin=build_origin(session=session_key(), task=_origin_task(), surface="store_memory"),
        idem_key=idem_key), ensure_ascii=False)


@server.tool(structured_output=False)
async def recall(
    query: str = "",
    top_k: int = 5,
    use_lnn: bool = True,
    sample: bool = False,
    filter_tags: str = "", library: str = "",
    scope: str = "project", libraries: str = "",
    occurred_after: str = "", occurred_before: str = "",
) -> str:
    """Semantic search for prior memories.
Combines vector search, BM25 (RRF), MMR deduplication, entity boosts, and a hard
relevance floor. By default ``superseded`` and ``archived`` records are filtered out.

Returns ``{count, memories[]}`` (not ``hits``). For assembled context, use
``get_memory_context``. Overly long content is reported via ``content_truncated``.

Args:
    query: Query text. Empty means implicit LNN temporal-state recall only.
    top_k: Number of results; default 5.
    use_lnn: Include LNN temporal state; default true.
    sample: Distance-weighted sampling instead of greedy ranking; default false.
    filter_tags: JSON object string narrowing candidates before retrieval, e.g.
        ``{"chapter": 43}``. All keys are AND; multiple values in one key are OR.
    scope: ``project`` (default), ``user``, or ``all``. For user/all, the response
        self-reports ``scope`` and ``libraries``; ``top_k`` is the total count.
    libraries: Explicit multi-library fan-out, comma-separated (up to 4). Each
        result is marked with its ``library``. Mutually exclusive with
        ``library=`` and ``scope=user|all``.
    occurred_after / occurred_before: Inclusive UTC date window in YYYY-MM-DD.
        Empty means unlimited. A time window narrows candidates; it does not
        bypass pinned memories.
    """
    await _await_warmup()
    # filter_tags 解析、键名白名单校验、候选收窄与 miss/计数留痕
    # 全部在 RecallService.recall（解析排在库解析之后，保序）；client 由
    # 适配层取 session_key() 传入（读侧接线）。
    svc = _recall_service(library, scope, libraries)
    return json.dumps(svc.recall(
        query=query, top_k=top_k, use_lnn=use_lnn, sample=sample,
        filter_tags=filter_tags, client=session_key(), scope=scope,
        libraries=libraries,
        occurred_after=occurred_after, occurred_before=occurred_before),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def get_memory(memory_id: str, library: str = "",
                     include_history: bool = False) -> str:
    """Fetch one memory by exact ID.
Returns the stored record, including state, source ids, entity ids, tags, and
provenance metadata when present.
    """
    await _await_warmup()
    svc = _recall_service(library)
    return json.dumps(svc.get_memory(memory_id=memory_id,
                                     include_history=include_history,
                                     client=session_key()),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def update_memory(
    memory_id: str,
    content: str = "",
    importance: float = -1.0,
    memory_type: str = "",
    state: str = "",
    superseded_by: str = "",
    clear_quarantine: bool = False, library: str = "",
) -> str:
    """Update selected fields of an existing memory.
Only supplied fields change; omitted fields remain untouched.

Use ``state="superseded"`` when a newer fact replaces an older one: the old
record is retained but leaves default recall. Provide ``superseded_by`` with the
replacement memory ID when available. Avoid overwriting content in place when
the intent is correction, because that destroys the audit trail.

Args:
    memory_id: ID of the memory to update.
    content: New content; omitted means no content change and no re-embedding.
    importance: New importance; pass -1 or omit to leave unchanged.
    memory_type: New memory type; omitted means unchanged.
    state: ``active``, ``superseded``, or ``archived``. Other values error.
    superseded_by: New memory ID that replaces this one; used with
        ``state="superseded"``.
    clear_quarantine: Promote a quarantined user-library write. Allowed only
        from a home-library session; non-home callers receive an explicit error.
    """
    await _await_warmup()
    svc = _write_service(library)
    # 校验（MemoryState 合法性 / superseded_by 存在性）与字段组装在
    # WriteService.update_memory；本工具只剩「调用 → 序列化」。
    return json.dumps(svc.update_memory(
        memory_id=memory_id, content=content, importance=importance,
        memory_type=memory_type, state=state, superseded_by=superseded_by,
        clear_quarantine=clear_quarantine),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def update_tags(
    tags: str = "",
    clear_keys: str = "",
    source: str = "",
    source_contains: str = "",
    contains: str = "",
    state: str = "",
    untagged_only: bool = False,
    missing_keys: str = "",
    memory_ids: str = "",
    mode: str = "merge",
    dry_run: bool = True,
    limit: int = 0,
    confirm: bool = False,
    undo: str = "", library: str = "",
) -> str:
    """Batch-update tags on existing memories.
Executes an explicit dry-run plan by default. Because tags are a flat mapping and
there is no automatic transaction, operations above the configured threshold
require explicit confirmation. The response includes a full before/after ledger
and an ``op_id`` for reversible undo.

Args:
    tags: JSON object of tag key/value updates.
    clear_keys: JSON array of tag keys to remove.
    missing_keys: JSON array of keys that must be removed if present.
    memory_ids: JSON array restricting the operation to specific memories.
    confirm: Required confirmation when the operation exceeds the safety threshold.
    dry_run: Preview the plan without writing; default true.
    undo: Undo a previous operation by ledger ``op_id`` or ``"last"``.
    untagged_only: Restrict the operation to records with no tags.
    """
    await _await_warmup()
    # 四段 JSON 解析走共用助手（行为逐位不变）：其错误体
    # **不带** library 首键，且现行代码就排在库解析之前（先报语法错，
    # 不碰库）——保序即保字节级一致。助手内部走严格版，坏 JSON
    # 显式报错，不与 knowledge_topic 那侧的 lenient 版混名。
    parsed, arg_err = _parse_tagops_args(tags, clear_keys, missing_keys, memory_ids)
    if arg_err:
        return json.dumps({"ok": False, "error": arg_err}, ensure_ascii=False)

    svc = _write_service(library)
    return json.dumps(svc.update_tags(
        tags=parsed["tags"], clear_keys=parsed["clear_keys"],
        source=source, source_contains=source_contains, contains=contains,
        state=state, untagged_only=untagged_only, missing_keys=parsed["missing_keys"],
        memory_ids=parsed["memory_ids"], mode=mode, dry_run=dry_run, limit=limit,
        confirm=confirm, undo=undo), ensure_ascii=False)


@server.tool(structured_output=False)
async def forget_memory(memory_id: str, reason: str = "", library: str = "") -> str:
    """Delete one memory and clean every derived copy.
This is the correct entry point for “forget this.” It cascades through entity
indexes, profiles, topic layers, and other derived state, writes a tombstone to
prevent resurrection from old snapshots, and restores records that this memory
had superseded.

Args:
    memory_id: ID of the memory to forget.
    reason: Why it is being deleted; retained in the tombstone/audit trail.
    """
    await _await_warmup()
    svc = _write_service(library)
    return json.dumps(svc.forget_memory(memory_id=memory_id, reason=reason),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def forget_by_query(query: str, top_k: int = 3, dry_run: bool = True,
                          min_similarity: float = -1.0,
                          allow_multiple: bool = False, library: str = "") -> str:
    """Find and optionally delete memories matching a query.
Preview is the default: it returns ranked candidates without deleting. Pass
``dry_run=false`` to delete the selected candidates. This is useful for “that
whole topic is wrong,” but review the preview before deleting.

Args:
    query: Query text used to locate candidates.
    top_k: Maximum candidates to inspect; default 3.
    dry_run: When true, return candidates only; default true.
    confirm: Confirmation required for a non-dry-run destructive operation.
    """
    await _await_warmup()
    svc = _write_service(library)
    return json.dumps(svc.forget_by_query(
        query=query, top_k=top_k, dry_run=dry_run,
        min_similarity=min_similarity, allow_multiple=allow_multiple),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def list_memories(
    limit: int = 50, source: str = "", state: str = "", library: str = ""
) -> str:
    """List memories with optional filters.
Use ``state="superseded"`` to inspect replaced records; they are excluded from
default recall but retained for history and audit.

Args:
    limit: Maximum number of records; default 50.
    source: Exact source filter, such as a file path.
    state: ``active``, ``superseded``, ``archived``, or empty for all.
    library: Optional library override.
    """
    await _await_warmup()
    svc = _recall_service(library)
    return json.dumps(svc.list_memories(
        limit=limit, source=source, state=state, client=session_key()),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def list_sources(library: str = "") -> str:
    """List known source labels and record counts.
Useful before bulk import or source-scoped cleanup. The response is metadata
only; use ``list_memories(source=...)`` to inspect records from one source.
    """
    await _await_warmup()
    svc = _recall_service(library)
    return json.dumps(svc.list_sources(client=session_key()),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def reset_context(library: str = "") -> str:
    """Reset the LNN temporal context for the current library.
This clears hidden state and recent-context windows. It does not delete
memories or alter stored records.
    """
    await _await_warmup()
    svc = _organize_service(library)
    # 注意：返回体**不带** library 首键（与现行工具逐位一致）。
    return json.dumps(svc.reset_context(), ensure_ascii=False)


@server.tool(structured_output=False)
async def get_stats(library: str = "") -> str:
    """Return basic service/library statistics.
For richer self-reporting, use ``brain_stats``.
    """
    await _await_warmup()
    svc = _bootstrap_service(library)
    # 注意：stats 返回体**不带** library 首键（与现行工具逐位一致）。
    return json.dumps(svc.get_stats(), ensure_ascii=False)


@server.tool(structured_output=False)
async def save_state(library: str = "") -> str:
    """Persist the current library state.
Normally automatic, but useful after large in-memory operations or before
external inspection. Returns a save receipt/report.
    """
    await _await_warmup()
    svc = _write_service(library)
    # ConcurrentWriteError 的 conflict 信封在 WriteService.save_state 内组装
    # （hint 指向 load_state / retry_save），工具只序列化。
    return json.dumps(svc.save_state(), ensure_ascii=False)


@server.tool(structured_output=False)
async def retry_save(library: str = "") -> str:
    """Retry the most recent failed save.
Use after fixing disk, permission, or concurrent-write issues. The response
reports whether the retry succeeded and which layers were written.
    """
    await _await_warmup()
    svc = _write_service(library)
    return json.dumps(svc.retry_save(), ensure_ascii=False)


@server.tool(structured_output=False)
async def load_state(library: str = "") -> str:
    """Reload persisted state from disk.
Use after another process modified the library or after recovering from a failed
save. Avoid reloading while another writer is mid-operation.
    """
    await _await_warmup()
    svc = _maintain_service(library)
    # 加载翻车（load_issue）与成功路径都在服务内组装，工具只序列化；
    # 返回体**不带** library 首键（与现行工具逐位一致）。
    return json.dumps(svc.load_state(), ensure_ascii=False)


# ------------------------------------------------------------------
# v0.3 用层工具：会话生命周期 + 5 槽位上下文 + 巩固 + 反馈
# ------------------------------------------------------------------
# ==================================================================
# 多库：进程内注册表 + 调用侧寻址（主通道）
# ==================================================================
# 默认库 = env HIPPOCAMPUS_PERSIST_DIR。留空（没指定）时落到它，
# 行为与多库之前**逐位一致** —— 兼容性是验收项。
_DEFAULT_LIB_PATH = _resolve_default_lib_path()

# 连接级默认库：session key -> 库路径（修 daemon 复用串库）。
# 为什么必须有：daemon 一条进程服务 N 条连接，``_DEFAULT_LIB_PATH`` 是**进程级**
# 常量（模块导入时按当时那个 launcher 的 env/cwd 算出）。第二个工作区复用同一个
# daemon 时，若没有这份连接级覆盖，它的写入会落进第一个工作区的库 —— 静默串库。
# 由 launcher 的带外握手行（``session_ctx.HELLO_PREFIX``）在连接建立时写入。
_conn_default_lib: dict[str, str] = {}
_conn_lib_lock = threading.RLock()

# C1 工作区绑定：session key -> 反查结果（session_start 自曝用）。
# 状态机四值 + declared：bound（目录 workspace_root 精确命中，library=命中库）
# ／unbound（目录在但没这条工作区）／ambiguous（多条命中——**不取第一个**，
# 歧义的家库比没有家库更危险，回落声明库并自曝）／no_cwd／no_user_library
# ／directory_unreadable（坏 sidecar ≠ 空，U5 教训）／lookup_error。
# declared = launcher HELLO 声明的库（env 口径）——反查命中时与 library
# 不同，这就是「绑库而非装机」覆盖 env 声明的可见证据。
_conn_workspace_binding: dict[str, dict[str, Any]] = {}


def _resolve_workspace_binding(cwd: str) -> tuple[str, dict[str, Any]]:
    """按连接工作区路径反查任务目录定家库（C1，设计稿 §3.2）。

    反查是**增强**不是前提：任何失败（user 库失联 / sidecar 坏 / 意外异常）
    都返回空路径 + 带原因的状态，绝不阻断连接建立——与 bridge 握手失败
    回落进程默认库同一纪律。判据（normcase 精确全等、歧义不裁决）在
    domain/task_directory.resolve_workspace，这里只做发现与 I/O。

    Returns:
        ``(path, info)``。path 非空 = 命中即家库；info 恒带 status（自曝原料）。
    """
    info: dict[str, Any] = {"cwd": cwd or "", "status": "no_cwd",
                            "library": "", "task": ""}
    if not (cwd or "").strip():
        return "", info
    try:
        user_lib = None
        for cand in libraries(False):
            if (cand.name or "").strip() == USER_LIBRARY_NAME:
                user_lib = cand
                break
        if user_lib is None:
            info["status"] = "no_user_library"
            return "", info
        data, issue = _task_directory.load_directory(user_lib.path)
        if issue:
            info["status"] = "directory_unreadable"
            info["reason"] = issue
            return "", info
        entry, status = resolve_workspace(
            normalize_entries((data or {}).get("entries")), cwd)
        info["status"] = status
        if entry is not None:
            info["library"] = entry["path"]
            info["task"] = entry["task"]
            # D2b：命中条目的读侧档位声明随绑定记录走（""=未声明→masked）。
            info["access"] = entry.get("access", "")
            return entry["path"], info
        return "", info
    except Exception as exc:  # noqa: BLE001 —— 反查绝不阻断连接
        logger.warning("workspace binding reverse lookup failed (falling back to the declared library): %r", exc)
        info["status"] = "lookup_error"
        info["reason"] = f"{type(exc).__name__}: {exc}"
        return "", info

# **按库**去重的预热登记。
# 背景：main() 那次预热跑在进程启动期，contextvar 为空 ⇒ session_key() 恒为
# __default__ ⇒ 用的是**进程级** _DEFAULT_LIB_PATH（＝第一个 launcher 的库）。
# daemon 被第二个工作区复用时，它的库**永远不会被预热**，首次调用自付 25–33s
# 加载（撞宿主工具超时，同型问题）。于是每条连接声明自己的库时补一次预热。
# 集合而非计数：同一库被 N 条连接声明，只预热一次。
_warmup_libs: set[str] = set()
_warmup_libs_lock = threading.Lock()


def set_connection_default_library(persist_dir: str = "", cwd: str = "",
                                   session: str | None = None) -> str:
    """给**当前连接**声明默认库（daemon 化后每条连接各自一份）。

    C1 起多一层**工作区绑定反查**：任务目录（user 库 sidecar）里
    ``workspace_root`` 与本连接 cwd 精确命中的条目，其 ``path`` 接管
    连接默认库——「绑库而非装机」的解析依据（设计稿 §3.2）。优先级：
    **目录反查命中 > launcher 声明（env 口径）> 进程级默认**。反查命中
    覆盖 env 声明是拍板过的裁决：用户级安装下 launcher 恒带 env 声明，
    若声明优先则绑定永不触发；显式性由 session_start 的
    ``workspace_binding`` 自曝兜底（declared 与 library 并列可见）。
    未命中/歧义/任何失败都回落声明库并记录状态——**未命中绝不落到
    user 库**（设计稿 §3.2 红线），status 留给开场包自曝。
    """
    path, fallback = resolve_default_lib_ex(persist_dir or None, cwd or None)
    declared = path
    key = session_key(session)
    hit, binding = _resolve_workspace_binding(cwd or "")
    binding["declared"] = declared
    if fallback:
        # 实测反馈 2026-10-31：声明库解析失败原本只落服务端日志（agent
        # 看不见），declared 还被改写成回落路径、原始声明名丢失。三键
        # 进 binding（task_directory 槽的 workspace 透传可见），warning
        # 文本另经 _inject_fallback_warning 顶到 session_start 返回体。
        binding["declared_raw"] = fallback["declared_raw"]
        binding["fallback_reason"] = fallback["reason"]
        binding["warning"] = _fallback_warning_text(fallback, declared)
    if hit:
        path = hit
        logger.info("workspace binding active: cwd=%s -> home library=%s (task=%s, overriding the declared library %s)",
                    binding.get("cwd"), path, binding.get("task"), declared)
    with _conn_lib_lock:
        _conn_default_lib[key] = path
        _conn_workspace_binding[key] = binding
    # I3：daemon 被多个工作区复用时，main() 那次进程级预热覆盖不到这个库。
    # 连接刚声明完就补一次（按库去重，失败只落一行 warning）。
    maybe_warmup_library(path)
    return path


def default_library_path(client: str | None = None) -> str:
    """本次调用语境下的默认库：连接级 > 进程级。"""
    with _conn_lib_lock:
        cd = _conn_default_lib.get(session_key(client))
    if cd:
        return cd
    return _DEFAULT_LIB_PATH


def _workspace_binding_brief(client: str | None = None) -> dict[str, Any]:
    """本连接的工作区绑定状态（session_start 的 task_directory 槽自曝用）。

    无 HELLO 的直连客户端（测试/排障）没有绑定记录 → ``undeclared``——
    不是错误，是「这条连接没走过声明通道」。
    """
    with _conn_lib_lock:
        rec = _conn_workspace_binding.get(session_key(client))
    if not rec:
        return {"status": "undeclared", "cwd": "", "library": "",
                "task": "", "declared": ""}
    return dict(rec)


def _origin_task() -> str:
    """写入凭据 ``origin.task`` 的取值：会话声明 > 工作区绑定映射 > 空。

    两个来源都是**宿主显式登记**的，不从正文猜（``common/write_origin``
    口径 3）：

    - 会话声明 ``session_start(task=...)``：本次会话在干什么，动态；
    - 任务目录的「任务 → 库」映射（``_conn_workspace_binding`` 的 task）：
      这个库属于哪个工作项，稳定。

    Super-agent 的 Ticket 更接近后者，但宿主若能显式声明任务，动态值更准，
    故优先前者。定义在文件下半部，上方三个写工具按名字运行时解析调用，
    与定义顺序无关。

    key 一律走 ``session_key()``（**不带 client**）——三个写工具都没有
    ``client`` 参数，若混用 ``session_key(client)``，宿主传了 client 就会
    错位，表现为 task 静默读不到（C1 绑定同一命名空间，不能各用一套）。
    """
    declared = current_task()
    if declared:
        return declared
    with _conn_lib_lock:
        rec = _conn_workspace_binding.get(session_key()) or {}
    return str(rec.get("task") or "").strip()


def _user_access_tier(info: dict[str, str] | None,
                      client: str | None = None) -> str:
    """D2b：本连接对 user 库的读侧档位（非 user 库目标返回空串）。

    判据单一真相在 domain/task_directory.resolve_user_access；这里只做
    发现（家库比对、绑定记录、逃生门）。档位随 client 解析——多库 daemon
    下每条连接各有绑定。
    """
    info = info or {}
    if (info.get("name") or "").strip() != USER_LIBRARY_NAME:
        return ""
    home = default_library_path(client)
    home_is_user = same_library(home, info.get("path", ""))
    with _conn_lib_lock:
        rec = _conn_workspace_binding.get(session_key(client)) or {}
    return resolve_user_access(
        home_is_user=home_is_user,
        binding_status=str(rec.get("status") or "undeclared"),
        entry_access=str(rec.get("access") or ""),
        escape_open=bool(load_config().cross_library_write))


def _fallback_warning_text(fallback: dict[str, str], declared: str) -> str:
    """声明库解析回落的 agent 可见告警文本（改动二，2026-10-31）。

    措辞与 D2b 现实对齐：unbound 回落形态下写走待审隔离，不是全量直写
    ——说「进待审」，不说「将错库」。回落库名取路径尾段做展示（user 级
    接入下 env 默认库目录名即 `user`）。
    """
    raw = fallback.get("declared_raw") or "(empty)"
    tail = os.path.basename((declared or "").rstrip("\\/")) or declared or "(unknown)"
    return (f"The workspace-declared library {raw!r} is unregistered or failed to resolve ({fallback.get('reason')}); "
            f"fell back to the default library {tail!r}: reads land there and writes go to the review queue (not direct writes). "
            f"Register the library via library_create, or check the library field of the "
            f"workspace .hippocampus.json.")


def _inject_session_disclosures(payload: str, client: str | None) -> str:
    """session_start 的全部注入（工具体 ≤20 行纪律的外移口）。

    顺序：先 C1 工作区绑定自曝（task_directory 槽内），后声明库回落
    ``⚠``（返回体顶层）——后者改写返回体结构，放最后避免被前者覆盖。
    """
    return _inject_fallback_warning(
        _inject_workspace_disclosure(payload, client), client)


def _inject_fallback_warning(payload: str, client: str | None) -> str:
    """实测反馈 2026-10-31：本连接发生过声明库回落时，session_start 返回体
    **顶层**放 ``⚠`` 告警块。

    **与 task_directory 槽位解耦**（槽关也注入）：误回落比省 token 贵——
    这是对「关＝零痕迹」口径的一处**有意例外**，注入前提是「这条连接真的
    发生了回落」（binding 带 warning），不是常驻键。自曝是赠品：任何异常
    原样返回，绝不炸开场包。
    """
    try:
        with _conn_lib_lock:
            rec = _conn_workspace_binding.get(session_key(client)) or {}
        warning = rec.get("warning")
        if not warning:
            return payload
        obj = json.loads(payload) if isinstance(payload, str) else payload
        if isinstance(obj, dict) and "⚠" not in obj:
            return json.dumps({"⚠": str(warning), **obj}, ensure_ascii=False)
    except Exception:  # noqa: BLE001 —— 自曝绝不炸开场包
        pass
    return payload if isinstance(payload, str) else json.dumps(
        payload, ensure_ascii=False)


def _inject_workspace_disclosure(payload: str, client: str | None) -> str:
    """C1：task_directory 槽 ready 时注入 ``workspace`` 绑定自曝。

    槽关 / 不 ready = 零痕迹（不加键，U5「关＝零痕迹」同口径）。自曝是
    赠品：任何异常原样返回 payload，绝不炸开场包。**独立成函数是 M6
    S1.2「工具体 ≤20 行」纪律的要求**——工具体只留一行调用。
    """
    try:
        obj = json.loads(payload) if isinstance(payload, str) else payload
        if isinstance(obj, dict):
            td = obj.get("task_directory")
            if isinstance(td, dict) and td.get("ready"):
                td["workspace"] = _workspace_binding_brief(client)
                return json.dumps(obj, ensure_ascii=False)
    except Exception:  # noqa: BLE001 —— 自曝绝不炸开场包
        pass
    return payload if isinstance(payload, str) else json.dumps(
        payload, ensure_ascii=False)


# 库清单缓存：扫描要 walk 磁盘，每次调用都重扫会把 list 类工具拖慢。
_library_cache: list[Any] | None = None
_library_cache_at = 0.0
_LIBRARY_CACHE_TTL = load_config().library_cache_ttl

_registry: BrainRegistry | None = None
_registry_lock = threading.RLock()


def _registry_get() -> BrainRegistry:
    """进程内多库注册表（惰性）。factory 注入使测试可塞假 brain。"""
    global _registry
    if _registry is not None:
        return _registry
    with _registry_lock:
        if _registry is None:
            _registry = BrainRegistry(factory=_build_brain)
    return _registry


def _brain_for(library: str | None = None,
               client: str | None = None) -> HippoBrain:
    """取本次调用目标库的 brain（多库入口）。

    **同一库路径只构造一次**：两个 brain 指向同一份 ``memories.json``
    是 v0.4 明确禁止的（写入互不可见 + 整体覆写丢数据）。
    每库独立锁 ⇒ 库 A 正在加载（25–33 秒）不会挡住库 B 的调用。
    """
    return _registry_get().get(resolve_library_path(library, client))


def _manager_for(library: str | None = None,
                 client: str | None = None) -> MemoryManager:
    """取本次调用目标库的 MemoryManager（与 brain 同一份内存状态）。"""
    return _brain_for(library, client).manager

def _map_fields_error(message: str, library: str | None = None) -> str:
    """map_set fields 解析失败的错误体（共用助手产出，序列化逐位不变）。

    两处 fields 错误（坏 JSON / 非 JSON 对象）现行都经
    ``json.dumps(_with_library({"status": "error", "error": ...}, library),
    ensure_ascii=False)`` 组装——错误体**带** library 首键（这是 map_set 与
    update_tags 两个保序工具的唯一差异点，有探针用例钉死）。抽成助手只省
    工具体行数，dict 键序（status→error）与序列化参数原样保留。

    注意 fields 的 ``json.loads`` 解析本身**刻意不抽**：工具层
    用例（test_m4_batch3_tool_face T4）以源码字面钉住「json.loads(fields)
    在 _organize_service(library) 之前」（保序：坏库名+坏 fields 先抛库错
    ——信封组装即库解析），抽走字面会翻红那条断言。
    """
    return json.dumps(_with_library(
        {"status": "error", "error": message}, library), ensure_ascii=False)


def _task_directory_register(entry: dict[str, Any]) -> dict[str, Any]:
    """把条目登记进用户级任务目录（``<user 库>/directory.json``）。

    这是任务目录的**受管通道**之一（登记），也是唯一自动
    挂点：``library_create`` 建库即调用。语义三条：

    * **目标库按角色定**：约定名 ``user`` 的用户级库——不是
      「当前寻址落点」，也不是调用方的家库。用户库没建/失联 ⇒ 返回
      ``registered: false`` + ``user_library_not_found``（**不建**用户库：
      建库是显式动作，不能被一次登记悄悄触发）。
    * **失败不抛**：登记是赠品，抛出去会让一次成功的建库变成失败（库已经在
      磁盘上，agent 却收到错误）。原因原样返回，由用例层转成
      ``directory_registered: false`` + warning。
    * **不静默**：坏 sidecar / 写盘失败都带 reason，绝不假装登记成功。

    读路径（``brain/session.py::_task_directory_brief``）不经过这里——它走
    轻量路径直读 sidecar，符号也不在本函数里出（读不需要权限）。
    """
    try:
        info = _library.resolve_library("user", libs=libraries())
    except LibraryResolutionError as exc:
        return {"registered": False, "reason": "user_library_not_found",
                "detail": str(exc)}
    if info is None:  # "user" 非空必走解析；None = 真失联（防御同上）
        return {"registered": False, "reason": "user_library_not_found"}
    try:
        return _task_directory.upsert_entry(info.path, entry)
    except Exception as exc:  # noqa: BLE001 —— 坏文件/写盘失败都要自曝
        logger.warning("task-directory write failed (%s): %r", _task_directory.directory_path(
            info.path), exc)
        return {"registered": False, "reason": "task_directory_write_failed",
                "detail": str(exc)}


def _library_admin_service() -> LibraryAdminService:
    """跨库管理 5 工具的用例服务（list/info/whoami/use/create）。

    **刻意不传 brain**：这 5 个工具不为某一个库构造 brain。唯一需要 brain
    的 ``library_create`` 经端口的 ``brain_for`` 在建库后现取（且只对**新建
    的那个库**取），其余四个全程与注册表无接触。

    注：本工厂**留在 server.py**（其余五个在
    ``server_runtime/composition.py``）——test_m4_batch4_tool_face T4 以
    ``ast.get_source_segment`` 钉住本函数在 server.py 源码里的函数体
    （「不传 brain、不经注册表直取」的源码级契约），按「改代码适配测试、
    不改测试」纪律（map_set json.loads 同例）留置；端口对象经
    再导出现取同一份。
    """
    return LibraryAdminService(ports=_LIB_PORTS)


_brain: HippoBrain | None = None


def get_brain() -> HippoBrain:
    """获取 HippoBrain 单例入口。

    **保留此函数是为了兼容性**（预热、旧脚本、测试都在用它）：它等价于
    ``_brain_for(None)``，即**默认库**。多库请用 :func:`_brain_for`。
    ``_brain`` 这个模块级变量现在只作**默认库的快捷缓存**，
    真正的持有者是 :class:`BrainRegistry`（key = 归一化的库路径）。
    """
    global _brain
    if _brain is not None:
        return _brain
    brain = _brain_for(None)
    _brain = brain
    return brain


@server.tool(structured_output=False)
async def ingest(
    text: str,
    speaker: str = "user",
    importance: float = -1.0,
    cognitive_type: str = "",
    entities: str = "",
    source: str = "",
    tags: str = "",
    auto_tags: bool = False, library: str = "",
    idem_key: str = "",
) -> str:
    """Recommended write entry for conversation or document text.
Stores the original text in L0, deterministically extracts L1 facts, links
entities, and reports merge/dedup/truncation details. No LLM API call is required
on the write path.

Args:
    text: Original conversation/document text.
    speaker: Speaker label, such as ``user`` or a role name.
    importance: 0 to 1; -1 lets the system infer importance.
    cognitive_type: experience / world / opinion; empty lets the system infer.
    entities: Comma-separated extra entity names; automatic extraction also runs.
    source: Source label such as a file path or session; returned on recall hits.
    tags: Structured tags as a JSON object string, e.g. ``{"chapter": 43}``.
    auto_tags: Enable automatic tag extraction; default false.
    idem_key: Idempotency key. Retrying the same request does not create a
        duplicate L0/L1 write and returns the existing memory IDs.
    """
    await _await_warmup()
    svc = _write_service(library)
    # tags 解析（错误体带 library 首键）、entities/importance/cognitive_type
    # 哨兵归一、save_warning 注入与日志都在 WriteService.ingest。
    return json.dumps(svc.ingest(
        text=text, speaker=speaker, importance=importance,
        cognitive_type=cognitive_type, entities=entities, source=source,
        tags=tags, auto_tags=auto_tags,
        origin=build_origin(session=session_key(), task=_origin_task(), surface="ingest"),
        idem_key=idem_key), ensure_ascii=False)


@server.tool(structured_output=False)
async def ingest_batch(items: str, source: str = "", tags: str = "",
                       granularity: str = "sentence",
                       detect_supersede: bool = False, save: bool = False,
                       auto_tags: bool = False, library: str = "") -> str:
    """Ingest multiple texts in one call.
Each item is processed independently; one invalid item does not block the rest.
The response reports per-item success, new L1 IDs, merges, truncations, and
errors. Prefer this over repeated single ``ingest`` calls for bulk imports.
    """
    await _await_warmup()
    svc = _write_service(library)
    # items/tags 解析（错误体均带 library 首键）、逐项隔离与整批落盘在
    # WriteService.ingest_batch；本工具只剩「调用 → 序列化」。
    return json.dumps(svc.ingest_batch(
        items=items, source=source, tags=tags, granularity=granularity,
        detect_supersede=detect_supersede, save=save, auto_tags=auto_tags,
        # P0-1：批量只接写入凭据，不接幂等键（批量导入的重放判据是
        # 源文件签名 source_drift，不是请求键）。
        origin=build_origin(session=session_key(), task=_origin_task(), surface="ingest_batch")),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def get_memory_context(query: str = "", top_k: int = 5, token_budget: int = 1200,
                             item_max_chars: int = 200, filter_tags: str = "", library: str = "",
                             scope: str = "project") -> str:
    """Recommended read entry for assembling Agent context.
Runs state filtering, hybrid retrieval, MMR, entity expansion, conflict
annotation, and budgeted five-slot assembly. The returned ``text`` is ready to
prepend to a system prompt; ``blocks`` and ``hits`` remain available for
programmatic use.

Args:
    query: Current task/question. Empty uses pinned-memory briefing.
    top_k: Hits requested for the current-hits slot; default 5.
    token_budget: Overall context budget; default 1200.
    item_max_chars: Per-item clipping in the rendered text; default 200.
        Pass 0 to disable clipping. ``hits`` always contain full content.
    filter_tags: JSON object string narrowing retrieval, e.g. ``{"chapter": 43}``.
    scope / libraries: Library selection semantics shared with ``recall``.
    """
    await _await_warmup()
    # filter_tags 解析（错误体不带信封）与 J2 回显在 RecallService.
    # get_memory_context；解析排在库解析之后，保序。
    svc = _recall_service(library, scope)
    return json.dumps(svc.get_memory_context(
        query=query, top_k=top_k, token_budget=token_budget,
        item_max_chars=item_max_chars, filter_tags=filter_tags, scope=scope,
        client=session_key()),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def session_start(task: str = "", recent_limit: int = 10,
                        working_on: str = "", task_top_k: int = 5,
                        pin_limit: int = 0, filter_tags: str = "",
                        client: str = "", library: str = "",
                        user_scope: int = 0, task_directory: int = 0) -> str:
    """Call at session start, after compaction, or when resuming work.
Returns a pre-activation package: recent memories, task recall, pinned items,
knowledge map, overview, status, task profile, map index, active sessions, and a
receipt.

For compaction recovery, keep only the receipt identity (``session_id``,
``assembled_at``, ``blocks``, ``memory_active``, ``knowledge_epoch``), do not copy
prose, then call this tool again with that ``session_id``. Treat the new response
as authoritative.

Args:
    task: Current task description; used for pre-recall and write origin.
    recent_limit: Number of recent memories; default 10, 0 disables.
    working_on: File/module being changed. Check ``active_sessions`` before editing.
    task_top_k: Task-recall result count; default 5, 0 disables.
    pin_limit: Pinned-slot count. 0 uses env (default 3); negative disables.
    filter_tags: JSON object string narrowing task pre-recall.
    client: Caller identity. Required on shared connections; otherwise the server
        creates a synthetic ``sess_*`` identity.
    library: Optional library override for this call/session.
    user_scope: User-level inheritance slot count. 0 uses env (default 3);
        negative disables the slot.
    task_directory: User-level task-directory slot count. 0 uses env (default 8);
        negative disables the slot.
    """
    await _await_warmup()
    return await _session_start_payload(
        task=task, recent_limit=recent_limit, working_on=working_on,
        task_top_k=task_top_k, pin_limit=pin_limit,
        filter_tags=filter_tags, client=client, library=library,
        user_scope=user_scope, task_directory=task_directory)


async def _session_start_payload(*, task: str, recent_limit: int,
                                 working_on: str, task_top_k: int,
                                 pin_limit: int, filter_tags: str,
                                 client: str, library: str,
                                 user_scope: int,
                                 task_directory: int) -> str:
    """Assemble the session_start payload after tool-face parsing/gating."""
    # Register the task declared for this connection; later writes use it for
    # origin.task. An empty task does not clear it; session_end does.
    bind_task(task)
    # Library resolution stays first: a bad library + bad JSON must report the
    # library error first, and that error body has no library envelope.
    svc = _bootstrap_service(library)
    parsed_tags, tag_err = _parse_tags_json(filter_tags)
    if tag_err:
        return json.dumps({"error": f"filter_tags parse failed: {tag_err}",
                           "input": filter_tags[:200]}, ensure_ascii=False)
    payload = json.dumps(svc.session_start(
        task=task, recent_limit=recent_limit, working_on=working_on,
        task_top_k=task_top_k, pin_limit=pin_limit, filter_tags=parsed_tags,
        client=client, user_scope=user_scope,
        task_directory=task_directory,
        user_access=_user_access_tier(svc.library_self, client)),
        ensure_ascii=False)
    return _inject_session_disclosures(payload, client)


@server.tool(structured_output=False)
async def export_memories(since_ts: float = 0.0, source: str = "", state: str = "",
                          limit: int = 0, library: str = "") -> str:
    """Export memories without vectors for backup or migration.
Supports incremental export: pass the largest ``timestamp`` from the previous
export. The boundary is inclusive, so deduplicate by ID after export.

Args:
    since_ts: Only records with timestamp >= this Unix time; 0 means all.
    source: Exact source filter.
    state: ``active``, ``superseded``, ``archived``, or empty for all.
    limit: Maximum records; 0 means unlimited.
    library: Optional library override.
    """
    await _await_warmup()
    svc = _maintain_service(library)
    # 注意：返回体**不带** library 首键（与现行工具逐位一致）。
    return json.dumps(svc.export_memories(since_ts=since_ts, source=source,
                                          state=state, limit=limit,
                                          client=session_key()),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def session_end(summary: str = "", open_topics: str | list[str] = "",
                      close_topics: str | list[str] = "",
                      session_id: str = "",
                      client: str = "", library: str = "") -> str:
    """Call at session end or when the user stops work.
Writes an L2 summary, records open topics, updates profile, triggers
consolidation, and closes the session lifecycle.

Args:
    summary: One-paragraph session summary.
    open_topics: Topics intentionally left unfinished; JSON array or comma list.
    close_topics: Topics now finished; JSON array or comma list. Close topics
        that have actually been resolved so the global backlog does not grow stale.
    session_id: Optional explicit session to finalize. Only sessions owned by
        this client may be finalized.
    client: Caller identity. Required on shared connections.
    library: Optional library override.
    """
    await _await_warmup()
    # P1：收尾清掉任务登记，避免下一会话误继承上一次的 origin.task。
    clear_task()
    svc = _bootstrap_service(library)

    # open_topics / close_topics 两种形态（JSON 数组 / 逗号串）的归一留在适配层。
    def _norm_list(v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return [str(t).strip() for t in v if str(t).strip()]

    return json.dumps(svc.session_end(
        summary=summary, open_topics=_norm_list(open_topics),
        close_topics=_norm_list(close_topics), session_id=session_id,
        client=client), ensure_ascii=False)


@server.tool(structured_output=False)
async def remember_intent(text: str, due: str = "", library: str = "") -> str:
    """Register a todo/intent for future sessions.
The intent is returned by later ``session_start`` calls and can trigger proactive
recall when a related topic appears. Use for user requests that span sessions.

Args:
    text: What should be done.
    due: Optional due date/deadline.
    """
    await _await_warmup()
    svc = _bootstrap_service(library)
    return json.dumps(svc.remember_intent(text=text, due=due), ensure_ascii=False)


@server.tool(structured_output=False)
async def complete_intent(keyword: str, library: str = "") -> str:
    """Mark a todo/intent complete by keyword.
Completion is keyword-based and reports whether a matching open intent was
found. Completed intents no longer participate in proactive recall.
    """
    await _await_warmup()
    svc = _bootstrap_service(library)
    return json.dumps(svc.complete_intent(keyword=keyword), ensure_ascii=False)


@server.tool(structured_output=False)
async def report_feedback(memory_id: str, useful: bool = True, library: str = "") -> str:
    """Report whether a recalled memory was useful.
Useful feedback increases future ranking priority; not-useful feedback lowers
visibility without deleting the memory. Use ``forget_memory`` for genuine
correction/deletion.
    """
    await _await_warmup()
    svc = _bootstrap_service(library)
    return json.dumps(svc.report_feedback(memory_id=memory_id, useful=useful),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def consolidate(filter_tags: str = "", max_items: int = 0,
                      dry_run: bool = False, library: str = "") -> str:
    """Run maintenance/consolidation over the library.
Can mark superseded facts, archive dormant records, and list contradiction
candidates. Dry-run is supported and recommended before a real write.

Args:
    filter_tags: JSON object string narrowing the consolidation scope.
    max_items: Maximum items to process; 0 uses the configured limit.
    dry_run: Report the plan without writing; default false.
    """
    await _await_warmup()
    # filter_tags 解析（错误体不带信封）、键名校验与 J2 软降级回显都在
    # OrganizeService.consolidate；解析排在库解析之后，保序。
    svc = _organize_service(library)
    return json.dumps(svc.consolidate(
        filter_tags=filter_tags, max_items=max_items, dry_run=dry_run),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def contradictions(limit: int = 10, source: str = "",
                         filter_tags: str = "", library: str = "") -> str:
    """List candidate pairs where one fact may have two versions.
This is annotation, not adjudication. Review candidates and resolve through
``update_memory`` (usually supersession) or ``forget_memory``.

Args:
    limit: Maximum candidate pairs; default 10.
    source: Optional source filter.
    filter_tags: JSON object string narrowing the candidate pool.
    """
    await _await_warmup()
    # filter_tags 解析（错误体不带信封）与 candidate_pool 回显在
    # RecallService.contradictions；解析排在库解析之后，保序。
    svc = _recall_service(library)
    return json.dumps(svc.contradictions(
        limit=limit, source=source, filter_tags=filter_tags),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def drift_audit(source: str = "", limit: int = 50,
                      include_ok: bool = False, library: str = "") -> str:
    """Audit memories whose source files may have drifted.
For file-backed sources, compares recorded signatures (size/mtime/hash) with the
current files and reports missing, changed, or unbaselined sources. Also reports
user-library quarantine and access-log summaries when applicable.

Args:
    source: Specific source to audit; empty audits known sources.
    limit: Maximum detailed rows; default 50.
    filter_tags: JSON object string narrowing the audited records.
    """
    # 零参数写法：层级按函数名在 STORE_ONLY_TOOLS 查表推断（drift_audit 已入表，
    # 只读库存 + 源签名层，不需要嵌入模型）。派发器那两处显式传参是结构性例外。
    await _await_warmup()
    svc = _recall_service(library)
    return json.dumps(svc.drift_audit(
        source=source, limit=limit, include_ok=include_ok,
        client=session_key()),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def entity_profile(entity: str, limit: int = 20, order: str = "asc",
                         include_superseded: bool = False, library: str = "") -> str:
    """Return the memory profile for one entity.
Includes linked memories, aliases when known, and mention statistics. Useful for
“what do we know about X?” before writing or updating project facts.

Args:
    entity: Entity name or alias.
    limit: Maximum linked memories; default 20.
    order: ``asc`` or ``desc`` chronological order.
    library: Optional library override.
    """
    await _await_warmup()
    svc = _recall_service(library)
    return json.dumps(svc.entity_profile(
        entity=entity, limit=limit, order=order,
        include_superseded=include_superseded), ensure_ascii=False)


@server.tool(structured_output=False)
async def entity_list(prefix: str = "", limit: int = 50, min_mentions: int = 1,
                      trusted_only: bool = False, library: str = "") -> str:
    """List known entities with optional prefix/count filters.
Use this to discover project vocabulary and inspect entity coverage.

Args:
    prefix: Optional case-insensitive prefix.
    limit: Maximum entities; default 50.
    min_mentions: Minimum linked-memory count; default 1.
    library: Optional library override.
    """
    await _await_warmup()
    svc = _recall_service(library)
    return json.dumps(svc.entity_list(
        prefix=prefix, limit=limit, min_mentions=min_mentions,
        trusted_only=trusted_only), ensure_ascii=False)


# ------------------------------------------------------------------
# 信息地图层：地图答「现在是什么」，记忆答「发生过
# 什么」。状态只走 map_set 一个入口（按键更新，同键再写整体替换）；
# 精确查询走 map_get（语义 top-k 查状态 = 把精确换成猜中）。
# 四个工具归独立 profile 组 "map" ⇒ common 面零 token 增加，full 与
# hippo_extra 可触达（drift_audit 归 admin 组的同一先例）。
# ------------------------------------------------------------------

@server.tool(structured_output=False)
async def map_set(name: str, key: str, status: str = "", fields: str = "{}",
                  desc: str = "", truth_source: str = "", note: str = "",
                  updated_by: str = "", library: str = "") -> str:
    """Create/update one key in a named information map.
Information maps store current state (e.g. “where is X deployed?”) separate from
historical memories. Writing the same key replaces the prior entry; provenance
and audit fields are retained.

Args:
    name: Map name.
    key: Entry key.
    status: Current status/value.
    fields: JSON object of additional structured fields.
    desc: Human-readable description; used for relevance matching.
    truth_source: File/path/system that is authoritative for this entry.
    note: Optional note.
    updated_by: Who/what made this update.
    library: Optional library override.
    """
    await _await_warmup()
    # fields 的 JSON 协议解析留在适配层：现行它排在 _brain_for 之前（先报
    # 语法错，不碰库），错误体带 library 首键（两保序工具的唯一差异点）。
    # json.loads 字面刻意保留在本函数——用例以它钉保序，见 _map_fields_error。
    fdict: dict | None = None
    if (fields or "").strip():
        try:
            fdict = json.loads(fields)
        except json.JSONDecodeError as exc:
            return _map_fields_error("fields is not valid JSON: %s" % exc, library)
        if not isinstance(fdict, dict):
            return _map_fields_error("fields must be a JSON object", library)
    svc = _organize_service(library)
    # brain.map_set 的 ValueError（非法 name/key）错误体在服务内组装，同样带 library 首键。
    return json.dumps(svc.map_set(
        name=name, key=key, status=status, fields=fdict, desc=desc,
        truth_source=truth_source, note=note, updated_by=updated_by),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def map_get(name: str, key: str, library: str = "") -> str:
    """Fetch one information-map entry by exact map/key.
Returns ``found=false`` with a reason when the key is absent. Prefer this over
semantic recall when you need current state, not historical discussion.
    """
    await _await_warmup()
    svc = _organize_service(library)
    # 注意：返回体**不带** library 首键（与现行工具逐位一致）。
    return json.dumps(svc.map_get(name=name, key=key), ensure_ascii=False)


@server.tool(structured_output=False)
async def map_list(name: str = "", library: str = "") -> str:
    """List information maps or all entries in one map.
Without ``name``, returns map names, counts, freshness, and drift summaries.
With ``name``, returns the entries in that map.
    """
    await _await_warmup()
    svc = _organize_service(library)
    # 注意：返回体**不带** library 首键（与现行工具逐位一致）。
    return json.dumps(svc.map_list(name=name), ensure_ascii=False)


@server.tool(structured_output=False)
async def map_verify(name: str = "", library: str = "") -> str:
    """Verify information-map entries against their truth sources.
For entries with a ``truth_source``, re-reads and compares the recorded source
signature, reporting stale or unverifiable entries. This is the map-layer
counterpart to source-drift auditing.
    """
    await _await_warmup()
    svc = _organize_service(library)
    # 信封（library 首键）在服务内组装——与现行 _with_library 逐位一致。
    return json.dumps(svc.map_verify(name=name), ensure_ascii=False)


@server.tool(structured_output=False)
async def knowledge_map(limit: int = 20, include_archived: bool = False,
                        library: str = "") -> str:
    """List derived knowledge topics in the library.
Topics are clusters of related L1 records with representative excerpts. Use this
to answer “what does this library know about?” and to locate a topic before
reading or writing its summary.

Args:
    limit: Maximum topics; default 20.
    include_archived: Include archived topics; default false.
    library: Optional library override.
    """
    await _await_warmup()
    svc = _recall_service(library)
    # 信封（library 首键）在服务内组装——本工具是读侧少数带 F5 自曝的。
    return json.dumps(svc.knowledge_map(
        limit=limit, include_archived=include_archived), ensure_ascii=False)


@server.tool(structured_output=False)
async def knowledge_topic(topic_id: str = "", action: str = "brief",
                          summary: str = "", evidence: str = "",
                          limit: int = 30, topics: str = "",
                          items: str = "", batch: int = 10,
                          per_topic: int = 5, fras: str = "",
                          library: str = "") -> str:
    """Read material for, or write/undo, one knowledge topic.
``action=brief`` returns cluster material. ``action=summarize`` writes a
host-generated summary back. ``action=revert`` undoes a host summary and returns
to extractive representative sentences. Evidence must be members of the topic.

Args:
    topic_id: Topic ID from ``knowledge_map``.
    action: ``brief`` / ``summarize`` / ``revert``.
    summary: Host-written summary for ``summarize``.
    evidence: JSON array of member memory IDs supporting the summary.
    fras: Optional four-field structured summary as a JSON object.
    library: Optional library override.
    """
    await _await_warmup()
    # items / fras 的 JSON 解析（失败显式报错、错误体不带信封）与
    # evidence/topics 的 lenient 解析都在 RecallService.knowledge_topic；
    # 解析排在库解析之后，与现行代码同序。
    svc = _recall_service(library)
    return json.dumps(svc.knowledge_topic(
        topic_id=topic_id, action=action, summary=summary,
        evidence=evidence, limit=limit, topics=topics, items=items,
        batch=batch, per_topic=per_topic, fras=fras), ensure_ascii=False)


@server.tool(structured_output=False)
async def lexicon_add(terms: str, relink: bool = True, library: str = "") -> str:
    """Add trusted entity names/aliases to the library lexicon.
The lexicon improves extraction, entity linking, and recall for project jargon,
model hashes, ticket numbers, and other names generic embedders miss. By default
relinks existing memories afterward.

Args:
    terms: JSON array or comma-separated list. Use ``alias => canonical`` for aliases.
    relink: Re-run entity linking after adding terms; default true.
    library: Optional library override.
    """
    await _await_warmup()
    # terms 的 lenient 解析（JSON 数组/逗号串都收，永不报错）与信封都在
    # OrganizeService.lexicon_add；解析排在库解析之后，保序。
    svc = _organize_service(library)
    return json.dumps(svc.lexicon_add(terms=terms, relink=relink),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def lexicon_suggest(top_n: int = 30, min_docs: int = 4, min_files: int = 2,
                          min_len: int = 3, max_len: int = 4, exclude: str = "", library: str = "") -> str:
    """Suggest high-frequency unregistered names for the lexicon.
Read-only candidate generation. Review suggestions before adding them; do not
automatically accept every frequent token.
    """
    await _await_warmup()
    # exclude 的 lenient 解析在服务内（现行排在库解析之后，保序）；
    # 注意：返回体**不带** library 首键（与现行工具逐位一致）。
    svc = _organize_service(library)
    return json.dumps(svc.lexicon_suggest(
        top_n=top_n, min_docs=min_docs, min_files=min_files,
        min_len=min_len, max_len=max_len, exclude=exclude),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def lexicon_alias_suggest(top_n: int = 30, min_confidence: float = 0.35,
                                min_support: int = 2, min_lift: float = 2.0,
                                min_containment: float = 0.5,
                                allow_stat_only: bool = False,
                                exclude: str = "", cue_words: str = "",
                                with_paren: bool = True, library: str = "") -> str:
    """Suggest possible aliases for the same entity.
Returns candidate groups with evidence; it does not merge automatically. Review
evidence, then register confirmed aliases through ``lexicon_add``.
    """
    await _await_warmup()
    # exclude / cue_words 的 lenient 解析在服务内（现行排在库解析之后，
    # 保序）；注意：返回体**不带** library 首键（与现行工具逐位一致）。
    svc = _organize_service(library)
    return json.dumps(svc.lexicon_alias_suggest(
        top_n=top_n, min_confidence=min_confidence, min_support=min_support,
        min_lift=min_lift, min_containment=min_containment,
        allow_stat_only=allow_stat_only, exclude=exclude,
        cue_words=cue_words, with_paren=with_paren),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def relink_entities(library: str = "") -> str:
    """Re-run entity extraction/linking across the library.
Use after a lexicon changes or when entity linking may be stale. The response
reports how many memories were relinked and any errors.
    """
    await _await_warmup()
    svc = _organize_service(library)
    # 信封（library 首键）在服务内组装——与现行 _with_library 逐位一致。
    return json.dumps(svc.relink_entities(), ensure_ascii=False)


@server.tool(structured_output=False)
async def miss_log(limit: int = 20, clear: bool = False, library: str = "") -> str:
    """Inspect recent retrieval misses.
A miss is a query whose top result did not pass the relevance floor. This is a
diagnostic signal for vocabulary gaps, retrieval failures, or missing project
knowledge—not an audit log of all queries.

Args:
    limit: Maximum recent misses; default 20.
    clear: Clear the miss log after reporting; default false.
    library: Optional library override.
    """
    await _await_warmup()
    svc = _maintain_service(library)
    # 本工具与 save_brain 一样**不带** library 首键（与现行工具逐位一致）。
    return json.dumps(svc.miss_log(limit=limit, clear=clear),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def brain_stats(daily_days: int = 0, library: str = "") -> str:
    """Return detailed library/runtime self-report.
Includes layer counts, entity stats, model/backend identity, save ledger,
knowledge coverage, and other health signals. Prefer this over ``get_stats``
for diagnosis.
    """
    await _await_warmup()
    svc = _maintain_service(library)
    # warmup 一节由注入探针产出（MaintainService.brain_stats），工具只序列化。
    return json.dumps(svc.brain_stats(daily_days=daily_days),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def save_brain(library: str = "") -> str:
    """Force a save of the current brain/library.
Useful before process exit, model swaps, or risky maintenance. Returns a save
report; failures are not silently swallowed.
    """
    await _await_warmup()
    svc = _maintain_service(library)
    # 信封（含 ConcurrentWriteError 的 conflict 体）在服务内组装，工具只序列化。
    return json.dumps(svc.save_brain(), ensure_ascii=False)


# ------------------------------------------------------------------
# F4' · 多库：library_* 工具（36 → 41）
# ------------------------------------------------------------------
@server.tool(structured_output=False)
async def library_list(include_backups: bool = False) -> str:
    """List discoverable memory libraries.
By default backup/snapshot directories are omitted. Use this before
``library_use`` or ``library_create`` to avoid guessing names or crossing
projects.

Args:
    include_backups: Include backup/snapshot directories; default false.
    """
    await _await_warmup()
    svc = _library_admin_service()
    # 扫描 / 零库自动建默认目录 / 行组装都在服务内（端口把连接态与磁盘读写
    # 留在适配层），工具只序列化。返回体**不带** library 首键（逐位一致）。
    return json.dumps(svc.list_libraries(include_backups=include_backups),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def library_info(library: str = "") -> str:
    """Describe one memory library.
Returns identity, path, layer files, sizes, model binding, and related metadata.
Empty library means the default/current library.
    """
    await _await_warmup()
    svc = _library_admin_service()
    # 刻意不为它构造 brain（只读目录元信息）；层文件大小经端口读盘。
    return json.dumps(svc.describe(library=library), ensure_ascii=False)


@server.tool(structured_output=False)
async def library_whoami() -> str:
    """Report the library this session/call currently targets.
Call this when the active library is uncertain. Write responses also self-report
``library`` as their first key, so a wrong destination can be checked there too.
    """
    await _await_warmup()
    svc = _library_admin_service()
    return json.dumps(svc.whoami(), ensure_ascii=False)


@server.tool(structured_output=False)
async def library_use(library: str, client: str = "") -> str:
    """Switch this session's sticky library.
After a successful switch, subsequent tools use this library without needing
``library=`` on every call. An unresolvable name fails loudly; it never silently
falls back to another library.

Args:
    library: Library name, alias, ``lib_*`` ID, or path.
    client: Caller identity; required on shared connections.
    """
    await _await_warmup()
    svc = _library_admin_service()
    # 解析不明/目标是备份 ⇒ 服务抛 LibraryResolutionError（不吞、不回落默认库）。
    return json.dumps(svc.use(library=library, client=client),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def library_create(name: str, path: str = "",
                         aliases: str = "", use: bool = False,
                         client: str = "", workspace_root: str = "") -> str:
    """Create a new memory library and optionally switch to it.
Creation is explicit: only this operation writes a manifest/initial storage.
Duplicate names and ambiguous paths fail rather than guessing.

Args:
    name: Library name.
    path: Optional explicit path; omitted uses the configured library root.
    aliases: Optional comma-separated aliases.
    use: Switch the current session to the new library after creation; default true.
    """
    await _await_warmup()
    svc = _library_admin_service()
    # 重名/空名/目标已存在且非目录 ⇒ 服务抛 LibraryResolutionError。
    return json.dumps(svc.create(name=name, path=path, aliases=aliases,
                                 use=use, client=client,
                                     workspace_root=workspace_root), ensure_ascii=False)


# ------------------------------------------------------------------
# 派发器（渐进式披露）：实现主体在 dispatch.py，注册块留在本模块
# ------------------------------------------------------------------
# _dispatch_map（经 vars(server) 现取工具函数）/ _hippo_help / _hippo_extra
# 见 server_runtime/dispatch.py（上方 import 再导出）；名单现取
# _TOOL_GROUPS（原 _ALL_TOOL_NAMES 快照随迁为现取派生），patch 后即刻生效。
# 仅在 _TOOL_PROFILE != "full" 时注册，两个理由（原文随迁 dispatch.py）：
# full 下派发器是多余负担；add_tool 程序化注册保持"已声明工具面 = 46"。

if _TOOL_PROFILE != "full":
    server._tool_manager.add_tool(
        _hippo_help,
        name="hippo_help",
        description=(
            "List extension actions hidden in this session but callable via hippo_extra, with their arguments "
            "(export / lexicon / entity graph / ops / state access, etc.). "
            "When unsure whether a memory capability exists or you cannot find the tool you need, call this first."
        ),
        structured_output=False,
    )
    server._tool_manager.add_tool(
        _hippo_extra,
        name="hippo_extra",
        description=(
            "Call a tool not directly exposed in this session by action name (export / lexicon / entity graph / ops, etc.). "
            'Call hippo_help first for the action list and argument names, then invoke via hippo_extra(action="...", '
            'args=\'{"argument": value}\')'
        ),
        structured_output=False,
    )


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------
def main() -> None:
    """启动 MCP stdio 服务端（启动序列在 entry.run_main）。"""
    _entry.run_main()


def __getattr__(name: str) -> Any:
    """按需解析重型符号（PEP 562）。

    本模块刻意不在顶层导入 ``brain`` / ``manager``（那会把 torch 的 ~5.7s 顶到
    MCP 握手路径上，见文件头的 import 说明）。为了让 ``server.HippoBrain``
    这类既有引用继续可用，这里在**首次读取该属性时**才真正导入 ——
    不读就没有任何代价，``get_brain()`` 走的是 ``_brain_class()``。
    """
    if name == "HippoBrain":
        from hippocampus_mcp.brain import HippoBrain

        return HippoBrain
    if name == "MemoryManager":
        from hippocampus_mcp.engine.manager import MemoryManager

        return MemoryManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
