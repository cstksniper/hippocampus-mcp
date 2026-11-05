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
    logger.warning("未知的 HIPPOCAMPUS_WARMUP=%r，回退 auto", _WARMUP_MODE)
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
    logger.warning("未知的 HIPPOCAMPUS_WARMUP_WAIT=%r，回退 raise", _WARMUP_WAIT_POLICY)
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
    "本服务是 AI Agent 的记忆底座。"
    "写用 ingest、读用 get_memory_context、裸检索用 recall、"
    "开会话用 session_start / 收尾用 session_end；"
    "remember_intent / consolidate 等工具若不在本次会话的工具列表里，"
    "先 hippo_help 拿目录、再 hippo_extra(action=\"名字\", args=\"{...}\") 调用，能力一个不少。"
    "宿主上下文若发生过压缩（compaction）：不要抄记忆正文，只保留 session_start "
    "返回体 receipt 里的身份信息（session_id / assembled_at / blocks / memory_active / "
    "knowledge_epoch），凭 session_id 重新调一次 session_start 即可重建开场包；"
    "memory_active 或 knowledge_epoch 与离线前不一致 ⇒ 库已被别的会话改过，以本次递送为准。"
)


def _create_server() -> Any:
    """兼容不同版本的 mcp SDK 构造参数。"""
    try:
        return MCPServer(
            name="hippocampus-mcp",
            version="0.1.0",
            description="智能体的记忆底座：四层记忆 + 实体索引 + 会话生命周期",
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
    """    「记住 X / 记一下 X」→ 本工具（单条事实；长文走 `ingest`）。裸写入：无 L0 溯源、不建实体。

    **返回体含截断自曝**：嵌入 512 token 硬上限，超出对向量零贡献——写入成功却检索不见。
    `truncated=true` 时仅前 `embedded_chars` 字入向量，余下拆成多张 <500 字卡另写。

    Args:
        content: 记忆的文本内容（必填）。
        importance: 0.0~1.0，默认 0.5。
        memory_type: episodic / semantic / procedural / general，默认 general。
        idem_key: 幂等键（P0-2）：同键重放**不新建、不重嵌入**，直接回既有
            `memory_id`（返回体 `status=duplicate`）。键应按**请求**唯一
            （重试沿用同一个），不要按内容哈希——内容哈希会把"两次真的写了
            同一句话"也当成重放。

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
    """「之前说的 X 是什么 / 有没提到 Y」→ 本工具（语义检索）。

    向量 + BM25(RRF) + MMR 去重 + 实体加成 + 硬门槛。**默认过滤
    superseded/archived**；看待作废条目走 `list_memories(state="superseded")`。
    返回 `{count, memories[]}`（键名 `memories` 非 `hits`）；装配式上下文用
    `get_memory_context`。单条 content 过长被截时该条目自曝 `content_truncated`。

    Args:
        query: 查询文本；留空则仅用 LNN 当前时序状态做隐式查询。
        top_k: 返回条数，默认 5。
        use_lnn: 融合 LNN 时序状态，默认 True。
        sample: 距离加权抽样，默认 False 用贪心排序。
        filter_tags: 结构化标签过滤（JSON 对象字符串），如 `{"章": 43}`，**检索前**收窄候选集。
        scope: 检索哪个库（U2）：`project`（默认，当前库）/ `user`（用户级库，约定名
            `user`）/ `all`（项目库结果全前、用户级殿后，`top_k` 是**总数**；两库独立
            嵌入模型分数不可比，只做硬分段不融合）。`scope=user/all` 时返回体自曝
            `scope` + `libraries`；跨库内容完全相同的用户级条目带 `duplicate_of` 标注。
            用户库未建会显式报错（不静默回落）。非法值报错。
        libraries: 显式多库扇出（B1，拍板②a）：逗号分隔库名（≤4 个），家库/任务库
            段全前、外围各流 RRF 融合竞争剩余名额（排名融合，跨库比分数没有意义）。
            每条带 `library` 来源标记；返回体自曝 `mode=multi_library` + 实际打的
            `libraries`；不在任务目录的值带 `unregistered`（可查，但看得见）。与
            `library=` / `scope=user|all` **互斥**（组合显式报错）。
        occurred_after / occurred_before: 时间窗（`YYYY-MM-DD`，UTC 基准，含边界日）。
            用户说「上周五」「上个月」「三年前」时用它——**把相对时间换算成绝对
            日期是调用方的事**（模型知道"今天"是哪天，库不知道），库只按时间戳裁
            候选。默认**没有**按时间淘汰的硬过滤（老记忆照常按相似度参与排序），
            所以时间窗是"精确圈定范围"而不是"防止旧记忆被丢"的必需品；只有部署
            方显式开了过时窗口时才需要它来绕开。传空 = 不限；格式错显式报错。
            返回体带 `time_window` 自曝生效区间——`count=0` 时要分清"窗口内没有"
            还是"检索没找到"。注意时间窗**不豁免 pin**：钉住是"多给一次机会"，
            不是"绕过用户的筛选条件"。
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
    """根据 ID 获取单条记忆的完整信息。

    Args:
        memory_id: 记忆的唯一标识符。
        library: 多库时指定目标库（留空走会话粘性/默认库）。
        include_history: True 时附 ``history`` 键——
            该记录的改写前值流水（update/merge 的 before，最新在前，
            cap 20）；默认 False 不加键（返回面零变化）。
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
    """
    「那条记错了 / 改成 X」→ 本工具；只更新提供的字段，其余保持不变。

    Args:
        memory_id: 待更新记忆的 ID。
        content: 新内容（留空则不更新）；会重新生成嵌入。
        importance: 新的重要程度（传 -1 或不传则不更新）。
        memory_type: 新的记忆类型（留空则不更新）。
        state: 只接受 `active` / `superseded` / `archived`，**其它值显式报错**。
            `superseded` = 作废旧事实但**保留记录**（退出默认召回）——「新事实取代旧结论」
            走此通道；不要改 content 覆盖旧事实（会留真假难辨的记录）。
        superseded_by: 取代本条的新记忆 ID（配合 `state="superseded"`），不存在则报错。
        clear_quarantine: 受管转正（D2a）：把 user 库待审队列里的一条转正
            （恢复进召回池与 user_scope 槽）。**只对家库会话开放**——非家
            会话请求显式报错（提权键不在 agent 手里）；转正动作记 access_log。
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
    """
    批量给已有记忆补标签 / 改标签 / 删标签——**存量回填的唯一通路**。

    `update_memory` 不接受 `tags`，所以入库时没打标签的记录事后就改不动了。
    「库里 58% 的条目没有任何标签，按域查不到东西」只能靠本工具补。

    **默认只出计划不写入**（`dry_run=True`）。影响超过 100 条还须显式
    `confirm=True`。幂等：值未变的条目不计入 `to_change`，同一条命令重跑
    第二次 `to_change=0`。

    **可撤销**：执行即记全量前后影像账本，返回体带 `op_id`；
    `undo=tagops_N`（或 `"last"`）回滚——**fail-closed**：只改回"当前值
    仍等于记录的 after"的条目，被改过的进 `conflicts` 自曝，不盖后来的
    修改；撤销本身也记账，可再撤一次。

    Args:
        tags: 要写入的标签，JSON 对象字符串，如 `{"域": "F"}`。
            `mode="merge"`（默认）逐键覆盖，`mode="replace"` 整体替换。
        clear_keys: 要删掉的键，JSON 数组字符串，如 `["旧键"]`。
        source: 只改来源**精确等于**该值的条目。
        source_contains: 只改来源**包含**该子串的条目。
        contains: 只改**正文包含**该关键词的条目。真库实测 52% 的无标签条目
            来自同一个文件，靠 source 无法细分，只有这一路能圈出「这一批」。
        state: 只改指定状态（`active` / `superseded` / `archived`）；留空 = 全部。
        untagged_only: 只改**完全没有标签**的条目（补缺场景）。
        missing_keys: JSON 数组，只改**至少缺其中一个键**的条目。
        memory_ids: JSON 数组，精确点名要改的 id。
        mode: `merge`（默认）／`replace`。
        dry_run: 默认 True —— 只出计划，不写。
        limit: 最多处理多少条（0 = 不限），先小批试用。
        confirm: 影响 >100 条时的显式确认。
        undo: 传上次执行的 `op_id`（或 `"last"`）；给了它就忽略
            tags/clear_keys 转为回滚计划（同样默认 dry_run、超阈值要
            confirm）。报"没有这个 op"时带 recent_ops 供选。

    Returns:
        `matched`（过滤命中数）／`to_change`（真会变的条数）／`skipped_unchanged`
        ／`sample`（前 10 条的改前改后）／`coverage_after`（执行后的标签覆盖率）。
        执行后另有 `op_id`（撤销凭据）。`matched=0` 时先怀疑过滤条件写错——
        它和「真没有符合条件的记录」长得一样。
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
    """    「这条别记了 / 删掉这条」→ 本工具，级联清理全部派生痕迹（也推荐用它纠正错误记忆）。

    除删 L1 外还清实体倒排引用、摘 L3 画像派生条目（否则错误信息仍进上下文）、
    剔他条 `entity_ids`、恢复被它取代的记忆为 active、标记 L0 未引用来源、写墓碑。

    Args:
        memory_id: 待删除记忆的 ID。
        reason: 删除原因（建议填，如「用户纠正：上线时间记错了」）。

    """
    await _await_warmup()
    svc = _write_service(library)
    return json.dumps(svc.forget_memory(memory_id=memory_id, reason=reason),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def forget_by_query(query: str, top_k: int = 3, dry_run: bool = True,
                          min_similarity: float = -1.0,
                          allow_multiple: bool = False, library: str = "") -> str:
    """按自然语言定位疑似错误记忆并删除（不知道 ID 时用这个）。

    默认只预览（`dry_run=True`）返回候选及内容，确认后再以 `dry_run=False` 执行。

    **真正执行时默认单次只删 1 条**：先剔除没有向量背书的候选（BM25 词面命中、
    similarity=0.0），再只删排序第一的那条，其余进 `skipped`（带 skip_reason）。
    「删掉那条记错的」在语义上几乎总是单数，多出来的槽位正是误删来源。

    Args:
            query: 自然语言描述，如「星轨项目三月上线」。
            top_k: **预览**多少候选，默认 3。它不等于删除条数（见上）。
            dry_run: True 只预览候选；False 才真正执行删除。
            min_similarity: 门槛，负值＝用默认值（0.30，与召回的相似度硬门槛同值）。
                刻意调高并不能有效防误删——相关性随查询长度整体漂移。
            allow_multiple: 确认要批量删时才传 True。届时仍会先剔除无向量背书的候选。

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
    """    「我库里都有些什么 / 最近存了啥」→ 本工具（仅元数据，不含嵌入向量）。

    按来源审计：先 `list_sources` 看各文件入库多少条，再 `source="卷五.md"` 列该文件条目。

    Args:
        limit: 返回的最大数量，默认 50。
        source: 只列出来自该来源（文件名/相对路径）的条目；留空不过滤。
        state: 只看指定状态 active/superseded/archived；留空不过滤。
    """
    await _await_warmup()
    svc = _recall_service(library)
    return json.dumps(svc.list_memories(
        limit=limit, source=source, state=state, client=session_key()),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def list_sources(library: str = "") -> str:
    """列出记忆库里的来源标签及各自条数（按文件审计用）。

    判断"哪些 md 文件已经导过、各导了多少条"，也是导入幂等的基础。
    批量导入的记录会显示文件相对路径；手工 ingest 且未传 source 的归入 "(无来源)"。
    """
    await _await_warmup()
    svc = _recall_service(library)
    return json.dumps(svc.list_sources(client=session_key()),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def reset_context(library: str = "") -> str:
    """重置 LNN 的时序上下文。

    清空 LNN 的隐藏状态和滑动窗口，适用于开始全新的对话或话题切换。
    已存储的记忆不会被删除，只是 LNN 不再记得之前的时序上下文。
    """
    await _await_warmup()
    svc = _organize_service(library)
    # 注意：返回体**不带** library 首键（与现行工具逐位一致）。
    return json.dumps(svc.reset_context(), ensure_ascii=False)


@server.tool(structured_output=False)
async def get_stats(library: str = "") -> str:
    """获取记忆系统的当前状态统计。"""
    await _await_warmup()
    svc = _bootstrap_service(library)
    # 注意：stats 返回体**不带** library 首键（与现行工具逐位一致）。
    return json.dumps(svc.get_stats(), ensure_ascii=False)


@server.tool(structured_output=False)
async def save_state(library: str = "") -> str:
    """将所有状态持久化到磁盘（向量库 + LNN + L0~L3 四层 + 实体索引）。

    与 `save_brain` 等价（两者现在都保存完整状态）。建议在重要操作后
    或定期调用，防止数据丢失。
    """
    await _await_warmup()
    svc = _write_service(library)
    # ConcurrentWriteError 的 conflict 信封在 WriteService.save_state 内组装
    # （hint 指向 load_state / retry_save），工具只序列化。
    return json.dumps(svc.save_state(), ensure_ascii=False)


@server.tool(structured_output=False)
async def retry_save(library: str = "") -> str:
    """并发冲突后重试保存：取磁盘最新版本，只补「磁盘上没有的增量」。

    **何时用**：`ingest` / `save_state` 返回 `conflict`，或 ingest 结果里出现
    `save_warning`——新记忆还在内存里，只是没落盘。

    L1 与 L0 一起存（只存 L1 会让 `source_ids` 指不到原文）；不绕过 `disk_stamp`
    保护；**不做字段级 merge**（双方都有的保留磁盘版本，本地改动进
    `dropped_local_edits`）。

    Returns:
        added / dropped_local_edits / disk_only（磁盘独有：别人新增的，或本进程删过又「复活」的）。
    """
    await _await_warmup()
    svc = _write_service(library)
    return json.dumps(svc.retry_save(), ensure_ascii=False)


@server.tool(structured_output=False)
async def load_state(library: str = "") -> str:
    """从磁盘恢复所有状态（向量库 + LNN + 四层记忆），覆盖当前内存中的数据。

    换嵌入模型后无需重新导入：按当前模型缺失的向量会自动补算。
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
        logger.warning("工作区绑定反查失败（回落声明库）：%r", exc)
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
        logger.info("工作区绑定生效：cwd=%s → 家库=%s（任务=%s，覆盖声明库 %s）",
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
    raw = fallback.get("declared_raw") or "(空)"
    tail = os.path.basename((declared or "").rstrip("\\/")) or declared or "(未知)"
    return (f"工作区声明库「{raw}」未注册/解析失败（{fallback.get('reason')}），"
            f"已回落默认库「{tail}」：读将落该库，写入将进待审队列（不是全量直写）。"
            f"请 library_create 注册该库，或核对工作区 .hippocampus.json 的 "
            f"library 字段。")


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
        logger.warning("任务目录写失败（%s）：%r", _task_directory.directory_path(
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
    """    「把这段存进去 / 帮我整理这段对话笔记」→ 本工具——**推荐写入口**。

    原文一字不改进 L0、抽取短句进 L1、维护实体索引；重要性与类型自动判定。批量走 `store_memory`。

    Args:
        text: 原始文本。
        speaker: 说话人（user / assistant / 角色名）。
        importance: 0~1；传 -1 由系统自动判定。
        cognitive_type: experience / world / opinion；留空自动判定。
        entities: 逗号分隔的额外实体名（系统也会自动抽取）。
        source: 来源标签（文件 / 路径 / 会话）；填了则 recall 命中结果带上它。
        tags: 结构化标签（JSON 对象字符串），如 `{"章": 43}`，供 recall /
            get_memory_context 的 filter_tags 过滤。
        auto_tags: 自动抽标签（默认关）。结构由 `<记忆库>/tag_schema.json` 定；生效键名用
            `brain_stats` 的 tag_schema 核对——**键名不一致 = filter_tags 静默失效**。（显式 tags 优先）
        idem_key: 幂等键（P0-2）：同键重放**不写 L0、不写 L1**，直接回既有
            `memory_id`（返回体 `status=duplicate`）。给编排层的重试 / resume
            用，键按请求唯一。

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
    """批量写入多段文本（一次调用写完一批，收尾只落盘一次）。

    逐条语义与 `ingest` 完全一致。两个批量特性：**逐项错误隔离**（某项坏只进
    `errors`，已成功的照常保留）与**整批只在收尾落盘一次**。单批上限 2000 条，超了请分批。

    Args:
            items: JSON 数组字符串，每项字段同 `ingest`（`text` 必填）。
                **项级字段覆盖批次级默认值**。
                例：`[{"text":"甲","importance":0.8},{"text":"乙","source":"x.md"}]`
            source: 批次默认来源标签（项里写了以项为准）。
            tags: 批次默认结构化标签（JSON 对象字符串），如 `{"章": 43}`。
            granularity: `sentence`（按句拆成多条 L1，默认）/ `chunk`（整段一条）。
            detect_supersede: 是否做事实更新检测；**批量导入建议保持 false**
                （md 本身已是去重后的当前真相，沿革叙述会被大面积误判）。
            save: 收尾是否落盘一次（开着自动保存的实例即使传 false 也会收尾存一次）。
            auto_tags: 逐项从正文自动抽标签（默认关，语义同 `ingest`）。

        返回：`count` / `failed` / `total_input` / `memory_ids` / `entities` /
        `merged_into`（每项带 index）/ `superseded` / `items` / `errors` /
        `saved` / `save_warning`。

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
    """
    「接着上次的 / 现在该想起什么」→ 本工具：5 槽位装配，可直接拼进 system prompt。

    5 槽位：当前命中 / 待办未决 / 时间线 / 相关实体状态 / 上次未聊完。

    Args:
        query: 当前任务或问题（用于检索命中与实体定位）。
        top_k: 命中记忆数量，默认 5。
        token_budget: 上下文字符预算，默认 1200。
        item_max_chars: 单条截断长度，默认 200。**传 0 = 不截断**（账目/条文类长句的
            数字尾巴常被切掉）。只影响 text。
        filter_tags: 结构化标签过滤（JSON 对象字符串），如 `{"章": 43}`，**检索前**收窄候选集。
        scope: 装配哪个库（U2）：`project`（默认，当前库）/ `user`（用户级库整体替换
            查询库，返回体自曝 `scope` + `libraries`）。**不支持 `all`**——5 槽位装配没有
            「项目排前用户殿后」的落点，要跨库请用 `recall(scope="all")` 或分别调用。
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
    """
    **每次会话第一步**（开新任务 / compaction 后）→ 本工具拿预激活包。

    **compaction 恢复**：只保留返回体 `receipt` 的身份信息（session_id /
    assembled_at / blocks / memory_active / knowledge_epoch），**不要抄正文**；
    凭 session_id 再调一次本工具重建，knowledge_epoch 对不上以本次为准。
    包内槽位各答一类问题：最近记忆 / task 预召回 / 钉住 / 知识地图（这库装了什么）/
    全貌卡（库整体是什么）/ 状态卡（现在什么状况）/ 任务轮廓（这任务触及什么）/
    地图索引（有哪几地图、哪张可能旧了）。**槽位开关与条数上限是运维侧 env**
    （进程启动时读定，调用方改不了；不占工具面参数是因为加参数就是长期 token 成本）。

    Args:
        task: 任务描述，可留空；非空则用于预召回。
        recent_limit: 带最近多少条记忆，默认 10；传 0 关闭。
        task_top_k: 按 task 预召回多少条，默认 5；传 0 关闭。
        working_on: 声明在改什么（文件/模块）。**改文件前先看返回的 `active_sessions`**，
            避免两会话同改一处；只登记，不阻止写入。
        pin_limit: 钉住槽条数。0 = 用 env（都没设则 3）；负 = 不带。
        user_scope: 用户级继承槽条数（U1）：带几条**用户级库**（约定名 `user`）的
            偏好类记忆——pinned 优先。0 = 用 env（都没设则 3）；负 = 不带该槽（返回体
            零 user_scope 键、零用户库 I/O）。用户库未建/失联不报错：`user_scope`
            键自曝 `ready=false` + `reason`。
        task_directory: 用户级任务目录槽条数（U5）：带几条 `任务 → 库` 映射，最近更新的
            在前；0 = 用 env（都没设则 8）；负 = 不带该槽（零键零 sidecar I/O）。
            超出的老任务不在包里：看 `task_directory.total`。未建/失联/文件坏都不报错：
            `task_directory` 键自曝 `ready=false` + `reason`。
        filter_tags: 结构化标签过滤（JSON 对象字符串），如 `{"章": 43}`。把 task
            预召回（`task_hits`）收窄到某个标签域，语义与 `recall` 的 `filter_tags`
            完全一致（全键 AND、单键多值 OR）。留空 = 不限域；只作用于 task 预召回，
            recent / pinned / knowledge 槽不受影响。
        client: 调用方身份。**共享连接下必填**：服务端自造 `sess_xxx` 标识进程不是你。
    """
    await _await_warmup()
    # P1：登记本次会话声明的任务，供后续写入的 origin.task 取用。
    # 空 task 不清（compaction 恢复那次多半不带），清除只由 session_end 做。
    bind_task(task)
    # 保序：库解析先行（现行代码 brain 解析在 filter_tags 解析之前——
    # 「坏库名 + 坏 JSON」必须先报库错），解析错误体不带 library 首键。
    svc = _bootstrap_service(library)
    parsed_tags, tag_err = _parse_tags_json(filter_tags)
    if tag_err:
        return json.dumps({"error": f"filter_tags 解析失败：{tag_err}",
                           "input": filter_tags[:200]}, ensure_ascii=False)
    # 键名白名单校验与 degraded 回显、预激活包组装在
    # BootstrapService.session_start。
    payload = json.dumps(svc.session_start(
        task=task, recent_limit=recent_limit, working_on=working_on,
        task_top_k=task_top_k, pin_limit=pin_limit, filter_tags=parsed_tags,
        client=client, user_scope=user_scope,
        task_directory=task_directory,
        user_access=_user_access_tier(svc.library_self, client)),
        ensure_ascii=False)
    # C1 工作区绑定自曝 + 实测反馈 2026-10-31 的声明库回落顶层 ⚠
    # （注入逻辑都在 _inject_session_disclosures，工具体薄纪律）。
    return _inject_session_disclosures(payload, client)


@server.tool(structured_output=False)
async def export_memories(since_ts: float = 0.0, source: str = "", state: str = "",
                          limit: int = 0, library: str = "") -> str:
    """
    按时间戳 / 来源 / 状态增量导出记忆（不含向量，可直接落盘另存）。

    Args:
        since_ts: 只导出 timestamp **>=** 该值的记忆（Unix 秒），默认 0 = 全量。
            **增量导出**：把上次导出结果里最大的 timestamp 原样传进来即可。边界是
            「含等于」，所以那批最后一条会重复一次——**按 id 去重**即可。**不要**传
            `max_ts + ε` 去躲这个重复：同一时间戳可能有多条（显式指定 timestamp
            的批量写入），加 ε 会把它们整批漏掉，那是丢数据。
        source: 只导出来自该来源的（精确匹配，如 `docs/decisions.md`）。
        state: 只导出该状态的（active/superseded/archived）；留空不限。
        limit: 最多导出多少条，默认 0 = 不限。
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
    """
    **每次会话收尾**（「今天到这 / 先这样」）→ 本工具：写摘要、登记未完结话题、更新画像、触发巩固。

    Args:
        summary: 本次会话的一句话摘要。
        open_topics: 没聊完的话题；JSON 数组或逗号分隔字符串都收。
            登记前做精确 + 相似度去重（默认开，env 可关）——相似命中会
            跳过并在 `dedup_hits` 里回显撞上的旧条，供你决定是否
            `close_topics` 收掉旧条；已达上限时返回 `topics_cap_reached`，
            优先 close 而不是硬塞。
        close_topics: 要**关闭**（从全局待办移除）的话题；JSON 数组或逗号
            分隔字符串都收，匹配同样是去空白精确匹配。话题收口了就关——
            全局待办纯追加会让收口话题永远残留。
        session_id: **可选**，精确收尾指定会话。只许收**本 client** 的
            会话（不许代收，错配报错且零副作用）；留空 = 收本路 current。
        client: 调用方身份。**共享连接下必填**——服务端自造 `sess_xxx` 标识进程不是你。
    返回体的 `maintenance_reminders` 是收尾维护清单（可能为空）：启用了文件建档
    的库会提醒「N 个文件磁盘更新未同步」，带保留标签键的地图超期会提醒同步——
    没问题时不出现，不占 token。`topics_closed` / `topics_deduped` 分别报
    本次关闭 / 去重跳过的话题数；`saved_layers` 是本次落盘的层级明细
    （written/skipped/failed，failed 必伴随 saved=false）。
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
    """登记一条待办/意图（"帮我记住我要做 X"），下次会话会提醒一次，不重复催。

    Args:
        text: 待办内容，要具体到可检查。
        due: 截止日期描述，可留空。
    """
    await _await_warmup()
    svc = _bootstrap_service(library)
    return json.dumps(svc.remember_intent(text=text, due=due), ensure_ascii=False)


@server.tool(structured_output=False)
async def complete_intent(keyword: str, library: str = "") -> str:
    """标记一条待办完成。

    Args:
        keyword: 待办内容中的关键词，用于匹配。
    """
    await _await_warmup()
    svc = _bootstrap_service(library)
    return json.dumps(svc.complete_intent(keyword=keyword), ensure_ascii=False)


@server.tool(structured_output=False)
async def report_feedback(memory_id: str, useful: bool = True, library: str = "") -> str:
    """反馈某条记忆是否真的用上了：有用则提升重要性，无用只降优先级不降权。

    Args:
        memory_id: 记忆 ID。
        useful: 是否有用。
    """
    await _await_warmup()
    svc = _bootstrap_service(library)
    return json.dumps(svc.report_feedback(memory_id=memory_id, useful=useful),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def consolidate(filter_tags: str = "", max_items: int = 0,
                      dry_run: bool = False, library: str = "") -> str:
    """执行记忆巩固（睡眠修剪）：标记被取代的事实、归档长期未用的记忆、列出矛盾候选。

    建议在会话结束或积累较多记忆后调用。矛盾只列出不自动裁决。

    Args:
        filter_tags: 结构化标签过滤（JSON 对象字符串），如 `{"域": "F"}`。
            把**沉寂归档与矛盾列举**的候选收窄到某个标签域，语义与 `recall`
            的 `filter_tags` 完全一致（全键 AND、单键多值 OR）。留空 = 全库，
            返回体 `scope: null` 自曝"这次没设域"。
        max_items: 本次沉寂归档最多处理多少条（0 = 不限）。大库分批次归档用。
        dry_run: 只干跑——列出**将会**归档的候选与矛盾候选，不做任何状态变更
            （不归档、不刷新画像/主题层、不落盘）。
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
    """
    按需列出库里「同一事实、两种口径」的候选对。只读，不改任何记忆。

    `consolidate` 只在收尾时顺带列 5 条；这个让你在**写之前**就能查整库有没有
    并存的口径。

    **只标注不裁决**：确认旧的该作废时由你调 `update_memory` 显式标，本工具
    绝不自动作废——按「向量相似 + 时间更新」自动裁决会把"两条在讲同一件事"
    误判成"后者推翻前者"，而作废后的条目会被 `recall` 默认过滤掉、**不报错**。

    Args:
        limit: 最多返回多少对（默认 10，按相似度降序）。
        source: 只查某个来源（文件名 / 会话标识）；留空 = 全库。
        filter_tags: 结构化标签过滤（JSON 对象字符串），如 `{"域": "F"}`。
            语义与 `recall` 的 `filter_tags` **完全一致**（全键 AND、单键多值
            OR）。用来把"整库矛盾"收窄成"**某个域内的**矛盾"——六代口径混在
            一个库里时，全局 top-N 会被无关域占满。
            ⚠️ 标签键名写错 ⇒ 候选池为 0，看返回体的 `candidate_pool`
            能当场发现（0 = 过滤没命中，不是"域内没矛盾"）。

    Returns:
        `candidates` 每对带 `a_id` / `b_id` / 双方正文与来源 / `sim` /
        `newer`（谁更新）；另有 `candidate_pool`（过滤后进比对的条数，即分母）。
        确认旧的该作废时，用 `update_memory(a_id, state="superseded",
        superseded_by=b_id)`——**标作废而不是删除**，删了就没有「曾经改过口径」
        的痕迹了。
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
    """源文件漂移审计："文件改了但记忆没更新"的服务端答案。只读，低频。

    拿 ingest 记的 {mtime, size, sha1} 签名与磁盘现状对比，六类回报：
    `stale`（文件已变该重导）/ `missing`（文件没了记忆还在）/ `unbaselined`
    （从没记过签名——**无法判断，不是没漂移**）/ `orphan`（签名无人引用）/
    `unresolvable`（非文件 source）/ `ok`（一致，默认不列）。每类带分母：
    `actionable=0` 且 `unbaselined=0` 才是"真没漂移"。`never_ingested`
    （从没导入过的文件）不在本范围，用 `export_memories(source=...)` 算。

    Args:
        source: 只审计这一个来源；留空 = 全库。
        limit: 每类最多回多少行（默认 50；0 = 只给计数）。
        include_ok: 是否把一致的源也列出来（默认只列有问题的）。
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
    """
    取某个实体的完整档案：把它参与过的全部记忆按时间拉一遍。每条带 memory_id /
    content / timestamp / source / state——能按时间核、能回指原文件、能看出哪条已被作废。

    Args:
        entity: 实体名（不区分大小写）。不知道有哪些实体先用 `entity_list`。
        limit: 最多返回条数，默认 20；传 0 不限。
        order: `asc`（默认，时间正序，拉时间线用）或 `desc`（最近在前）。
        include_superseded: 是否带上已作废条目，默认 False。
    """
    await _await_warmup()
    svc = _recall_service(library)
    return json.dumps(svc.entity_profile(
        entity=entity, limit=limit, order=order,
        include_superseded=include_superseded), ensure_ascii=False)


@server.tool(structured_output=False)
async def entity_list(prefix: str = "", limit: int = 50, min_mentions: int = 1,
                      trusted_only: bool = False, library: str = "") -> str:
    """列出记忆库里已有哪些实体（按提及条数降序）。

    Args:
        prefix: 只看名字以该前缀开头的实体（如 "沈"）。
        limit: 最多返回条数，默认 50。
        min_mentions: 至少被多少条记忆提到才列入，默认 1。
        trusted_only: 只看受信任来源（manual/lexicon/title/spacy）的实体。
            启发式产出的候选噪声很大，排查时可先勾上这个。
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
    """改地图上某个条目的当前状态：「把 CN2601 挪到验证集」「chr1 组装完成了」。

    基因组进度 / 合约分到哪个集合 / 编译开关当前值这类**现在是什么**的信息
    走本工具，不要写成记忆——记忆的旧状态会在 recall 里和新状态竞争，而
    精确键查被语义 top-k 挡着（设计文档 §2 的核心判断）。

    **同一键再写整体替换**（一键一条是数据模型保证的，不靠自觉）；首次写入
    自动建图，`desc` 一句话说明这图管什么（任务相关性匹配靠它）。
    `truth_source` 声明「这条目的真相在哪个文件」：set 时记录签名，之后
    `map_verify` 重读比对——文件变了这条目标 drift，不靠人记得核。

    Args:
        name: 图名，如「合约」「基因组」「代码开关」。
        key: 条目键，如 CN2601 / chr1 / --use-avx2。
        status: 状态值，如「验证集」「QC 通过」；留空保留原值。
        fields: 附加字段 JSON 字符串，如 `{"来源": "10月重划分"}`。
        desc: 图描述（建图时写；之后传非空可改）。
        truth_source: 真相源文件路径（相对建档项目根或绝对路径）。
        note: 备注（短文本）。
        updated_by: 谁改的（宿主/会话标识，可留空）。
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
            return _map_fields_error("fields 不是合法 JSON：%s" % exc, library)
        if not isinstance(fdict, dict):
            return _map_fields_error("fields 必须是 JSON 对象", library)
    svc = _organize_service(library)
    # brain.map_set 的 ValueError（非法 name/key）错误体在服务内组装，同样带 library 首键。
    return json.dumps(svc.map_set(
        name=name, key=key, status=status, fields=fdict, desc=desc,
        truth_source=truth_source, note=note, updated_by=updated_by),
        ensure_ascii=False)


@server.tool(structured_output=False)
async def map_get(name: str, key: str, library: str = "") -> str:
    """查地图上某个条目现在是什么：「CN2601 现在在哪个集合」。

    L2 精确查询——**不能省这个入口**：recall 是语义 top-k，用它查状态等于
    把「精确」换成「猜中」（花大力气堵的「静默返回空」会换个形态回来）。
    未命中**显式报** ``found=false`` + ``reason``（no_maps / no_map /
    no_key），不静默返回空。

    Args:
        name: 图名。
        key: 条目键。
    """
    await _await_warmup()
    svc = _organize_service(library)
    # 注意：返回体**不带** library 首键（与现行工具逐位一致）。
    return json.dumps(svc.map_get(name=name, key=key), ensure_ascii=False)


@server.tool(structured_output=False)
async def map_list(name: str = "", library: str = "") -> str:
    """列出本地图有哪些地图（名+条数+新鲜度+漂移数）；点名一张图看它全部条目。

    不知道图有哪些、或某张图都有什么条目时用本工具。`name` 留空 = 列地图，
    给了图名 = 列该图条目（按最近更新排序，cap 100，超出自曝）。
    """
    await _await_warmup()
    svc = _organize_service(library)
    # 注意：返回体**不带** library 首键（与现行工具逐位一致）。
    return json.dumps(svc.map_list(name=name), ensure_ascii=False)


@server.tool(structured_output=False)
async def map_verify(name: str = "", library: str = "") -> str:
    """核对地图条目与真相源文件还对不对得上（代码开关/合约分配的自动巡检）。

    对声明了 `truth_source` 的条目：重读文件比签名——没变 zero-read；
    变了读一次内容哈希仲裁（git checkout / touch 重置 mtime 不误报）；
    真变了标 ``drift``（reason=changed/missing），条目清单随返回体给出。
    没有声明真相源的条目跳过（纯人工状态退到索引行 ⚠️ / 收尾提醒）。
    `name` 留空 = 全库核对。

    Args:
        name: 只核对这一张图；留空 = 全部。
    """
    await _await_warmup()
    svc = _organize_service(library)
    # 信封（library 首键）在服务内组装——与现行 _with_library 逐位一致。
    return json.dumps(svc.map_verify(name=name), ensure_ascii=False)


@server.tool(structured_output=False)
async def knowledge_map(limit: int = 20, include_archived: bool = False,
                        library: str = "") -> str:
    """整理 / 列出这个记忆库的**知识结构**：有哪些知识点、每个知识点一句话、多大规模。

    **说「整理hippo库」「梳理知识点」就是调本工具**——返回体里的 `playbook`
    是一份可直接照着执行的菜谱（取素材 → 你归纳 → 写回），`pending_count`
    是还剩几个知识点没归纳。

    和 `recall` 的分工：那个回答「我记得什么跟这句话有关」，这个回答
    **「这个库里到底装了些什么」**——L1 是平铺的事实、L2 只有会话级摘要且
    写完即沉睡、`entity_list` 只能按实体看，此前没有任何工具能回答它。

    主题是 L1 的**派生层**：由 active 事实聚类而来（受信任实体锚点 + 余弦
    凝聚），成员失效自动重裁；**不进 `recall` 主路径**（V9f 教训：摘要类内容
    进检索把 F1 从 .311 打到 .200），出口只有本工具和开场上下文。

    ⚠️ **代表句是抽取式摘录**（离主题质心最近的几条原文），**不是归纳**。
    真正的上层结论要你（宿主）读完这几条之后总结再写回——本服务的生产
    路径不内嵌 LLM，这是刻意守的边界。

    Args:
        limit: 最多返回多少个主题，默认 20；传 0 不限。
        include_archived: 是否带上成员已被删空而归档的主题，默认 False。

    库里从没梳理过时，首次调用会**自动梳理一次**（全库聚类，万条约秒级），
    返回体的 `auto_built` 标 True；之后只在 `consolidate` 时增量刷新。
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
    """对一个「知识点」出素材 / 写回归纳 / 撤销归纳（另含批量与重建 action）。

    `knowledge_map` 回答「这个库有哪些知识点」，本工具是**单个知识点**上的
    读写通道。三件事刻意合成一个工具而不是三个：工具面每会话随 tools/list
    注入宿主上下文，加一个工具就是长期 token 成本（本项目 42 个工具已经踩过
    这条线）。

    **action=brief（默认）** 出归纳素材：按「离主题质心最近」排序取 N 条原文
    ——不是按时间序，最早那批往往正是后来被推翻的。

    **action=summarize** 把你的归纳写回。三条硬校验，任一不过**整条退回、
    一个字段都不改**：summary 非空、≤2000 字、evidence 必须是本主题的存活
    成员 id（凭空引用会被拒）。

    **可选四字段契约**（`fras`）：`{"f":"范畴（直接当标题）","r":["关联
    实体/主题"],"a":"可引用的结论","s":"推不出来但搞错就出事的约束"}`。
    四键必须齐全，`s` 没有也要写"无"；只传 fras 不传 summary 时 `a`
    就是那句话。

    **action=revert** 撤销归纳、退回抽取式代表句。写错了要能撤，否则一条坏
    归纳会永久占位（自动流程不敢动它，人工又看不见它）。宿主命名过的标题
    （fras.f）跟着归零。

    **action=rebuild** 重画簇边界（整层世代 +1）。只在**换了聚类参数**（
    `HIPPOCAMPUS_TOPIC_MAX` / `MAX_MEMBERS`）或旧结构已经长歪时用——平时
    的增量刷新（`action=refresh`，或 consolidate 自动跑）不会动边界。
    重建会按成员重叠迁移旧的 host 归纳（搬过去的标 ⚠️ 待复核），迁移不了
    的条数在返回值 `lost_host` 里自曝——它不会静默消失。
    ⚠️ 只在**单写者**场景用：多进程同时重建，最后保存的那一代覆盖前一代。

    🔑 **host 归纳是稀缺资产**：写回后 `refresh`（巩固）只填抽取式、绝不覆盖
    它；多进程层合并时 **host 优先于时间戳**；成员集合变了只标 `stale`、
    不改写内容——本服务判断不了归纳对不对，只负责把「依据变了」摆出来。
    要改就再写一次，要退回就 revert。

    Args:
        topic_id: 主题 id（从 `knowledge_map` 拿）。
        action: brief / summarize / revert / rebuild / refresh。
        summary: 归纳文本（summarize 必填；传了 fras 可省，用 fras.a）。
        evidence: 出处，L1 记忆 id 列表（JSON 数组或逗号分隔）。
        limit: 素材条数，默认 30（1-200）。
        fras: 可选四字段结构（JSON 对象字符串），见上；summarize_batch
            的每条 item 里也可各带一个 fras 键。

    出素材挑的是「最能代表这个主题」的几条，读完总结完写回，这个库才真正
    有了上层知识——抽取式摘录永远只是摘录。
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
    """向项目专名词库追加专名并**立即生效**（热加载）。

    登记在册的专名会被稳定抽出、参与检索加成与画像。还能**登记别名**（同一实体的
    表字 / 简称 / 全称 / 旧称 / 代号）：别名行写 `子安 => 顾言`（`->` / `→` 等价），
    抽取与查询两侧都会归一。

    Args:
            terms: JSON 数组 `["同安","顾父"]` 或逗号分隔串；已存在的自动跳过。
                语法不完整的行**不写盘**，原样出现在返回的 `invalid` 里。
            relink: 追加后对全库重挂实体，**默认 True 且建议别关**——否则新词只对
                之后写入的记忆生效，存量记忆仍然查不到。大库重挂约几十秒。写了别名
                会额外把历史上按别名拆开的档案并回规范名。

        `alias_count` 是当前生效的别名总数（0 表示一条都没生效）；查单个别名是否
        命中，看 `entity_profile("子安")` 返回的 `resolved_from`。

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
    """列出「高频出现但还没登记进词库」的候选专名，供人工过筛后追加。
    **只做候选展示，不自动写词库**；过筛后用 `lexicon_add` 落盘。

    做法是中文 n-gram 频次统计 + 碎片抑制；按「出现过的记忆条数」降序并给出跨文件数
    （只在同一文件里反复出现的多半是该文件的术语或标题）。通用行话与序数短语已内置
    过滤；项目**特有**的结构标签写进 `<词库同目录>/entity_lexicon_stop.txt`。

    Args:
            top_n: 最多返回多少个候选，默认 30。
            min_docs: 至少在多少条记忆里出现过才列出，默认 4。
            min_files: 至少跨几个源文件，默认 2。
            min_len / max_len: 候选长度，默认 3~4（2 字候选几乎全是通用词）。
            exclude: 临时排除词，逗号分隔或 JSON 数组。

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
    """挖「同一实体的多种称法」候选。**只出候选 + 证据，不自动写盘**
    （按 `paste` 复核后用 `lexicon_add` 落盘）。与 `lexicon_suggest` 分工：那个问
    「这个词要不要登记成专名」，这个问「这两个名字是不是指同一个东西」。

    四条信号：`cue`（称谓提示词）、`paren`（括号同位语）、`cooc`（共现）、
    `abbrev`（包含关系）。**后两条不能单独成立**（要就得 `allow_stat_only=True`）：
    实测 `台湾外记 => 台湾` 对包含度 100% 却不是同一事物。

    返回按置信度降序，带 `paste` / `direction_basis` / `evidence`。**不自动归并**——
    合并即改变实体身份，猜错难拆。

    Args:
            top_n: 最多返回多少个候选，默认 30。
            min_confidence: 置信度门槛，默认 0.35。括号信号（0.30）单独过不了，
                需要多条证据或另一个信号——真实语料里「（作者）」「（一手）」这类
                注释性括号太多，让它单独成立会拉低整表可用性。
            min_support: 共现类候选的最少共同出现条数，默认 2。
            min_lift: 共现的提升度门槛，默认 2.0（独立时约 1）。
            min_containment: 共现的包含度门槛，默认 0.5（与提升度**满足其一即可**；
                小语料里提升度天然趋近 1，只设提升度会漏掉真候选）。
            allow_stat_only: 是否允许只靠统计信号成候选，默认 False。
            exclude: 本次临时排除词，逗号分隔或 JSON 数组。
            cue_words: 本次额外补充的称谓提示词（不落盘）。项目专有的写法建议写
                `<词库同目录>/alias_cue_words.txt`（一行一个，# 注释）长期生效。
            with_paren: 是否扫括号同位语，默认 True。

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
    """对全库已有记忆重跑实体抽取并补挂（词库扩充后的收尾动作）。

    一般不需要手动调用——`lexicon_add` 默认已经做了。只有在你手动编辑了
    entity_lexicon.txt 之后，才需要跑它让改动对存量记忆生效。
    """
    await _await_warmup()
    svc = _organize_service(library)
    # 信封（library 首键）在服务内组装——与现行 _with_library 逐位一致。
    return json.dumps(svc.relink_entities(), ensure_ascii=False)


@server.tool(structured_output=False)
async def miss_log(limit: int = 20, clear: bool = False, library: str = "") -> str:
    """查看（或清空）「召回不到」的查询留痕。

    显式查询召回不到时自动记一条（带查询、当时的最高相似度、入口、会话 id），平行期
    不用手记「没命中的 query」，可直接喂归因分析与门槛评测。阈值默认是召回硬门槛 0.30。
    **不要轻易上调**：实测真命中的最高相似度只有 0.32~0.53，调到 0.50 会把真命中记成未命中。

    Args:
            limit: 返回最近多少条，默认 20（按时间倒序）。
            clear: 传 True 清空留痕。

    """
    await _await_warmup()
    svc = _maintain_service(library)
    # 本工具与 save_brain 一样**不带** library 首键（与现行工具逐位一致）。
    return json.dumps(svc.miss_log(limit=limit, clear=clear),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def brain_stats(daily_days: int = 0, library: str = "") -> str:
    """查看四层记忆 + 索引层的规模统计（L0/L1/L2/L3、实体数、待办数等）。

    返回里带 `warmup` 一节——本工具只依赖**库存**层级（几秒应答），所以它同时是
    「后端准备好没有」的**探针**：`ready=false` 表示正在预热，此时其余带向量的
    工具会返回「稍后重试」，重试即可，不必重启会话。

    Args:
            daily_days: `daily_activity.recent` 回显最近多少天。**0 = 用 env
                `HIPPOCAMPUS_DAILY_ACTIVITY_DAYS`，都没设则 14**。注意这与**存储**上限
                无关（库里最多留 60 天），只影响这里回显几条。

    """
    await _await_warmup()
    svc = _maintain_service(library)
    # warmup 一节由注入探针产出（MaintainService.brain_stats），工具只序列化。
    return json.dumps(svc.brain_stats(daily_days=daily_days),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def save_brain(library: str = "") -> str:
    """持久化整个记忆底座（向量库 + LNN + 四层数据 + 索引层）。"""
    await _await_warmup()
    svc = _maintain_service(library)
    # 信封（含 ConcurrentWriteError 的 conflict 体）在服务内组装，工具只序列化。
    return json.dumps(svc.save_brain(), ensure_ascii=False)


# ------------------------------------------------------------------
# F4' · 多库：library_* 工具（36 → 41）
# ------------------------------------------------------------------
@server.tool(structured_output=False)
async def library_list(include_backups: bool = False) -> str:
    """列出已发现的记忆库。**只读**，不创建也不修改文件——**唯一例外**：一个库都
    没有时（全新部署首启）会建出默认库目录，并在返回里给 `auto_created`。

    扫描根：env `HIPPOCAMPUS_LIBRARY_ROOTS`（`os.pathsep` 分隔），默认
    `~/.hippocampus_mcp` + `<仓库>/data`。

    备份目录（`.bak-*` / `.snap-*` / `corrupt`）默认不列出——它们是只读快照，
    **不能**作为写入目标；盘点磁盘占用时再开 `include_backups`。

    Args:
            include_backups: 是否连同备份 / 快照目录一起列出。

    """
    await _await_warmup()
    svc = _library_admin_service()
    # 扫描 / 零库自动建默认目录 / 行组装都在服务内（端口把连接态与磁盘读写
    # 留在适配层），工具只序列化。返回体**不带** library 首键（逐位一致）。
    return json.dumps(svc.list_libraries(include_backups=include_backups),
                      ensure_ascii=False)


@server.tool(structured_output=False)
async def library_info(library: str = "") -> str:
    """某个记忆库的详情（路径 / id / 名字 / 别名 / 层文件）。

    **刻意不构造 HippoBrain**：本工具只读目录元信息，所以不必等嵌入模型
    （那要 25–33 秒）。要看条数/统计请用 ``brain_stats``。

    Args:
        library: 库名 / 别名 / ``lib_`` id / 路径。留空＝本次调用的目标库。
    """
    await _await_warmup()
    svc = _library_admin_service()
    # 刻意不为它构造 brain（只读目录元信息）；层文件大小经端口读盘。
    return json.dumps(svc.describe(library=library), ensure_ascii=False)


@server.tool(structured_output=False)
async def library_whoami() -> str:
    """**本次调用会落到哪个库、能写谁** —— 防串库的第一道自曝。

    `library_use()` 是**纪律不是强制**（已知洞），漏调就静默落到默认库；会话首句
    问一句即可确认，不必等到写完才发现写错库。

    返回里看三样：`home`（可写家库，写永远钉在这）、`current`（本次寻址落点，读落这）、
    `permission`（`read-write` 家库 / `managed-write` user 受管通道 / `read-only`
    会被守卫拦下）。`source` 说明 current 怎么定的：`binding`（会话级粘性）/ `default`。
    """
    await _await_warmup()
    svc = _library_admin_service()
    return json.dumps(svc.whoami(), ensure_ascii=False)


@server.tool(structured_output=False)
async def library_use(library: str, client: str = "") -> str:
    """会话级切换：此后本会话**读类**工具自动落到该库（**不必逐个传 library**）；
    **写永远钉在家库**（家库写边界会拦下跨库写，报错文案里有说明；env 逃生门
    ``HIPPOCAMPUS_CROSS_LIBRARY_WRITE=1`` 需改宿主配置后重连才生效）。
    优先级：**显式 `library` 参数 > 会话粘性 > 默认库**。
    写侧归属用 ``library_whoami`` 确认（``home`` = 可写家库）。

    **不自动建库**：库不存在就报错并给出候选——自动建库会让拼错的库名变成一个
    新库，写进去的记忆从此找不到（静默且不可逆）。

    切换后**后台预取**该库（首次调用要加载全库 + 嵌入模型，25–33 秒），避免这段
    落到第一次工具调用上被判超时。

    Args:
            library: 库名 / 别名 / `lib_` id / 路径。**必填**。
            client: 会话署名，多会话共享进程时用它区分绑定。

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
    """
    新建一个记忆库（写 `library.json` 清单 + 初始化存储）。

    建库是**显式动作**：只有这里写清单，日常扫描（`library_list`）绝不写盘。
    重名直接报错——两个同名库会让 `library_use(名字)` 变歧义，而歧义的代价是
    写错库。

    Args:
        name: 库名（必须唯一）。
        path: 库目录；**留空**则放在默认扫描根下（`<第一个根>/hippo_<name>`）。
        aliases: 逗号分隔的别名。
        use: 建完是否立即切到该库。
        client: 会话署名。
        workspace_root: C2 受管绑定——把该工作区路径登记为本库的绑定根
            （HELLO 反查命中即家库）。这是「绑库而非装机」onboarding 的
            第三步：问一次 → 建库 → 绑定，一步到位。留空 = 不绑定。
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
            "列出本会话被隐藏、但可通过 hippo_extra 调用的扩展功能及其参数"
            "（导出/词库/实体图/运维/状态存取等）。"
            "不确定某个记忆功能是否存在、或找不到要用的工具时，先调本工具查目录。"
        ),
        structured_output=False,
    )
    server._tool_manager.add_tool(
        _hippo_extra,
        name="hippo_extra",
        description=(
            "按 action 名调用本会话未直接暴露的工具（导出 / 词库 / 实体图 / 运维等）。"
            '先调 hippo_help 看有哪些 action 及参数，再用 hippo_extra(action="...", '
            'args=\'{"参数名": 值}\') 调用。'
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
