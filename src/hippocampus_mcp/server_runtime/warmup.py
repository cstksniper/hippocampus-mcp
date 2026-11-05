# -*- coding: utf-8 -*-
"""MCP 服务端预热区（server.py 预热区块的实现本体）。

这里实现：后台预热线程体（``_warmup_body`` / ``start_background_warmup`` /
``_warmup_library`` / ``maybe_warmup_library``）、工具侧等待
（``_await_warmup``）、零等待就绪探针（``warmup_status``）、诊断看门狗
（``_start_warmup_watchdog``）、导入链预载（``_preload_import_chain`` /
``_preload_list``）、``STORE_ONLY_TOOLS`` 层级表、两个预热异常类、
torch 单线程兜底（``_prepare_torch``）。

🔴 状态与常量**刻意不搬**——它们留在 server.py，因为那是测试的
monkeypatch 面（``_WARMUP_MODE`` / ``_warmup_ready`` / ``_warmup_started``
/ ``_warmup_error`` / ``_warmup_fatal`` / ``_warmup_phase`` / ``_store_ready``
/ ``_WARMUP_WAIT_TIMEOUT`` / ``_WARMUP_WAIT_POLICY`` / ``_PRELOAD*`` /
``_warmup_libs`` / ``_PERSIST_DIR`` …，见 tests/test_warmup.py 与
tests/test_i3_warmup_per_library.py）。本模块对这些符号的读写一律经
:func:`_srv` **现取**（教训：快照进新模块会让 monkeypatch 静默失效）；
对 server 侧函数（``get_brain`` / ``_registry_get`` /
``default_library_path`` …）的调用同样现取——测试替换
``server.HippoBrain`` / ``server._registry_get`` 才能一路生效到预热线程。

server.py 侧按 import 名原样再导出本模块公共符号（``_await_warmup`` /
``warmup_status`` / ``STORE_ONLY_TOOLS`` / ``WarmupPendingError`` /
``WarmupFailedError`` / ``start_background_warmup`` / ``maybe_warmup_library``
/ ``_prepare_torch`` / ``_preload_import_chain`` / ``_start_warmup_watchdog``），
既有调用点（46 个工具体的 ``await _await_warmup()`` 字面）与测试引用
（``srv.STORE_ONLY_TOOLS`` / ``isinstance(e, srv.WarmupPendingError)`` …
）零改动。测试对 ``srv._await_warmup = <替身>`` 的替换同样生效——工具
调用点与 server 模块同名，名字解析走 server 模块命名空间。
"""

# env 直读豁免：本文件的 env 直读是第三方库线程 env 与 warm 早读
# （Config 消费前），封闭清单见 config.py「根层豁免清单」。
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from typing import Any

from hippocampus_mcp import library as _library

logger = logging.getLogger("hippocampus_mcp")


def _srv() -> Any:
    """server 模块本体（现取入口，见模块 docstring 的 monkeypatch 面说明）。"""
    from hippocampus_mcp import server as _s

    return _s


class WarmupPendingError(RuntimeError):
    """预热未在预算内完成。

    抛给 MCP 层后变成一次**明确的工具错误**（而不是客户端超时）：调用方据此
    可以重试，也可以提示用户；把"还没准备好"与"坏了"区分开。
    """


class WarmupFailedError(RuntimeError):
    """预热过程中抛了异常。消息里带原始原因，避免把配置问题误判成性能问题。"""


# torch 不在顶层导入（见 server.py 头注），但「单线程」设置必须在 torch 被
# 真正使用之前生效。用本函数做统一兜底：无论从哪条路径首次碰 torch（auto 的
# 预热线程 / off 的首次工具调用 / sync 的主线程）都已经设好。
# 本标志归本模块自有（测试无 patch 面，与 server 命名空间解耦）。
_torch_ready = False


def _prepare_torch() -> None:
    """首次导入 torch 并限制为单线程（**仅当确有 torch 消费者时**）。

    Windows 下 torch/OpenMP 在 anyio 工作线程里推理偶发 access violation，
    限制单线程可规避（嵌入吞吐对本服务不是瓶颈）。

    ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` 必须在 ``import torch`` **之前**
    设进环境，所以放在本函数里而不是 main() 里。

    默认配置（LNN=numpy + 嵌入 auto/onnx）的生产进程
    **零 torch**，这里不再无条件导入——只有显式回退路
    （``HIPPOCAMPUS_LNN_BACKEND=torch`` / ``HIPPOCAMPUS_EMB_BACKEND=torch``）
    才付出这份内存。ST 回退路（auto 无产物）由 ``embeddings._load_model``
    在真正加载 ST 时补设线程限制。
    """
    global _torch_ready
    if _torch_ready:
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    needs_torch = (
        (os.environ.get("HIPPOCAMPUS_LNN_BACKEND") or "").strip().lower() == "torch"
        or (os.environ.get("HIPPOCAMPUS_EMB_BACKEND") or "").strip().lower() == "torch"
    )
    if needs_torch:
        try:
            import torch

            torch.set_num_threads(1)
        except Exception:  # noqa: BLE001 导入失败不在这里抛，后面构造时会给出更明确的错
            pass
    _torch_ready = True


def _warmup_body() -> None:
    """后台线程执行体：把重型初始化（记忆库 / 嵌入模型 / spaCy）挪出握手路径。

    分阶段计时日志是**运维必需**的：预热一旦卡住（例如模型路径不可达、试图联网
    下载 spaCy 模型），宿主侧表现为"工具调用了但一直没响应"，没有阶段日志就只能
    靠猜。
    """
    t_all = time.time()
    s = _srv()

    def _phase(name: str, fn: Any) -> Any:
        t0 = time.time()
        s._warmup_phase = name          # 超时时消息里要能指明"卡在哪一步"
        logger.info("预热阶段开始: %s", name)
        try:
            return fn()
        finally:
            logger.info("预热阶段完成: %s（%.1fs）", name, time.time() - t0)

    try:
        try:
            brain = _phase("加载记忆库/构造 HippoBrain", s.get_brain)
        except BaseException:
            # 核心层没建起来：后续所有工具都不可能成功，标记为致命。
            s._warmup_fatal = True
            raise
        # 库存就绪：读条数 / 按 id 取 / 列来源 / 看统计这类操作只需要它（约 5 秒）
        s._store_ready.set()
        # 🔴 **不预热 spaCy**。spaCy 的 thinc.compat 会
        # `import torch`（实测 +160MB 级常驻），而 NER 路径本就是惰性的
        # （`_get_nlp` 首次英文抽取才真正加载、失败退启发式）。预热强行加载
        # = 每个会话进程永久多付 ~240MB（spaCy 78 + torch 160），中文库更是
        # 纯浪费。代价：首次英文实体抽取多付一次 ~2-5s 的加载延迟。
        emb = brain.manager.embedder
        # 再拆一层：模型构造函数 vs 真正算一个向量。两者卡住的原因完全不同
        # （前者多为模型路径/缺件或 DLL 加载慢，后者多为推理线程环境问题）。
        _phase("构造嵌入模型", emb._ensure_model)
        _phase("首次推理", lambda: emb.embed("warmup"))
        # 后端自述必须在**启动时**就能看见。spaCy 装不上、
        # 嵌入器静默回落 MiniLM 这两类事故的共同点就是"结果变了而没人知道"，
        # 而日志是宿主侧唯一会去看的地方（brain_stats 要等人来调）。
        # ⚠️ 这里**不能**调 stats()：它要遍历全库算 entity_origins 等自曝项，
        # 一旦抛异常会被外层 try 记成「预热失败」——为打一行日志毁掉预热不值。
        # 账簿那格只取内存里的末条，代价可忽略。
        try:
            _last_save = (brain.save_ledger_view().get("last_save_at")
                          or "从未落盘")
        except Exception:  # noqa: BLE001 —— 日志字段缺失不影响预热成功
            _last_save = "unknown"
        logger.info(
            "后台预热完成: L1=%d 实体=%d（总 %.1fs）"
            " | NER=%s | 嵌入后端=%s(%s 维, 模型来源=%s)"
            " | 账簿: 上次保存=%s",
            brain.manager.store.count(), len(brain.entities), time.time() - t_all,
            brain.ner_backend_lazy,  # 无副作用版：别在预热日志里把 spaCy/torch 拉进进程
            brain.manager.embedder_backend,
            getattr(brain.manager, "embedding_dim", "?"),
            getattr(brain.manager.embedder, "model_source", "n/a"),
            _last_save,
        )
    except BaseException as e:  # noqa: BLE001 预热失败必须也不能拖垮服务
        s._warmup_error = f"{type(e).__name__}: {e}（阶段：{s._warmup_phase}）"
        logger.warning("后台预热失败（首次调用将自行初始化）: %s", s._warmup_error)
    finally:
        # 失败也要放行，否则所有工具会一直卡到超时。
        # _store_ready 同样要置位：若 get_brain() 本身就抛错，它不会走到 set()，
        # store 级工具会白等满超时才转惰性初始化（那时同样会失败，但至少不装死）。
        s._store_ready.set()
        s._warmup_ready.set()


def start_background_warmup() -> threading.Thread | None:
    """启动后台预热线程（仅 auto 模式）。立即返回，不阻塞握手。"""
    s = _srv()
    if s._WARMUP_MODE != "auto":
        return None
    # main() 这次预热覆盖的是进程级默认库，登记进集合，
    # 免得第一条连接再为同一个库起第二个预热线程（重复加载模型）。
    with s._warmup_libs_lock:
        s._warmup_libs.add(_library.normpath(s.default_library_path()))
    t = threading.Thread(target=_warmup_body, name="hippo-warmup", daemon=True)
    t.start()
    s._warmup_started = True
    logger.info("预热已转入后台线程（mode=auto，超时上限 %.0fs）", s._WARMUP_WAIT_TIMEOUT)
    return t


def _warmup_library(path: str) -> None:
    """后台预热**指定库**（构造 brain + 装模型 + 跑一次推理）。

    与 main() 那次 ``_warmup_body`` 的区别：那次只能覆盖进程级默认库，
    这一条是**连接声明了别的库**时补的。全程吞异常——预热失败只意味着
    首次调用要自付加载时间，不该让连接建立或任何工具调用失败。
    """
    try:
        brain = _srv()._registry_get().get(path)
        emb = brain.manager.embedder
        emb._ensure_model()
        emb.embed("warmup")
        logger.info("库预热完成: %s（L1=%d 条）", path,
                    brain.manager.store.count())
    except BaseException as e:  # noqa: BLE001 —— 预热失败绝不能拖垮服务
        logger.warning("库预热失败（首次调用将自行初始化）: %s —— %s",
                       type(e).__name__, e)


def maybe_warmup_library(path: str) -> None:
    """连接声明了自己的默认库之后，为该库补一次后台预热（按库去重）。

    **只在 auto 模式**下触发（off / sync 是用户显式选择，不替他改）；
    同一库只起一次线程；失败不抛。
    """
    s = _srv()
    if s._WARMUP_MODE != "auto" or not path:
        return
    with s._warmup_libs_lock:
        if path in s._warmup_libs:
            return
        s._warmup_libs.add(path)
    threading.Thread(target=_warmup_library, args=(path,),
                     name="hippo-lib-warmup", daemon=True).start()


# 预载档位（HIPPOCAMPUS_PRELOAD）：
#   off / 0   不做预载 —— **会卡死**。实测 PRELOAD=off 时 150s 仍未完成。
#   1（默认）  scipy.special + scipy.interpolate（热态 0.9s / 冷态 6.6s）
#   其他逗号分隔的模块名亦可。
#
# 为什么这一集**不能再收窄**（独立进程实测）：
#   · torch 已经改成按需导入（见 server.py 头注）——它在事件循环启动之后导入
#     **正常**，实测：预热 28.6s 完成 L1=9359、recall 命中 3 条。
#     所以 torch 不再占用握手时间。
#   · 但 scipy 不行：只预载 scipy.special 仍卡（卡点转移到 _fitpack）；
#     完全不预载必卡。所以优化只能从别处找，不要再压缩这个集合。
#   · 附注：单独预载 scipy.interpolate 实测也能通过（冷启动省约 2s），但
#     "只 special 卡、只 interpolate 不卡"这种不对称说明卡点对加载顺序敏感，
#     默认档仍保留两者以求保险。
#
# 实测数据（热态，probe_handshake auto）：
#              默认档                      预载 0.9s → 工具可见 5.6s / recall 首次 29.2s ✅
#              sklearn                     3.6s → 工具 10.3s / recall 22.4s
#              full = sentence_transformers 13.7s → 工具 18.7s ⚠️ / recall 9.3s
#              `full` 能把握手推过注册窗口，只在"本机确认宿主会补注册"时才用。
#
# 档位值本体（``_PRELOAD``）与预算（``_PRELOAD_BUDGET`` / ``_HANDSHAKE_BASE_COLD``）
# 留在 server.py（无 patch 面但属头部配置区），本模块经 _srv() 现取。
# 实测会卡死在 LoadLibrary 上的模块。只在**事件循环启动之前**加载才安全。
# 规律是"事件循环起来后加载 scipy 的任何 C 扩展都会卡"，所以这一集要覆盖
# 后续导入会碰到的 scipy C 扩展；scipy.interpolate 会带出 linalg/sparse/special。
_PRELOAD_DEFAULT = ("scipy.special", "scipy.interpolate")
_PRELOAD_FULL = ("sentence_transformers",)


def _preload_list() -> tuple[str, ...]:
    """把 ``HIPPOCAMPUS_PRELOAD`` 解析成要预载的模块元组。"""
    preload = _srv()._PRELOAD
    low = preload.lower()
    if low in ("0", "false", "no", "off", ""):
        return ()
    if low == "full":
        return _PRELOAD_FULL
    if low in ("1", "true", "yes", "on"):
        return _PRELOAD_DEFAULT
    return tuple(m.strip() for m in preload.split(",") if m.strip())


def _preload_import_chain() -> None:
    """在事件循环启动前预载嵌入模型的导入链（best-effort，失败不阻断启动）。

    **为什么必须在循环启动前做**（全部为独立进程实测）：
    同一句 `import sentence_transformers`（会拉起 torch / transformers / sklearn / scipy）：
      · 在事件循环启动**之前**做 → 正常完成，之后构造模型只要 **0.7s**
      · 在 `server.run()` **起来之后**做 → 卡在 scipy 的 C 扩展（`_ufuncs.pyd`；
        清掉它之后换 `_fitpack.pyd`…）的 `LoadLibrary` 上，30/60/120s 三次线程栈
        dump 停在**同一行**、永不返回（不是慢，是不动）
    同一份加载在普通脚本里 12–16s 正常完成，所以这是本机环境的进程级现象。

    代价与取舍见 ``_PRELOAD`` 的说明（server.py 头部配置区）。
    """
    s = _srv()
    mods = _preload_list()
    if not mods:
        return
    t_all = time.time()
    for name in mods:
        t0 = time.time()
        try:
            __import__(name)
            logger.info("导入链预载: %s（%.2fs）", name, time.time() - t0)
        except Exception as e:  # noqa: BLE001
            logger.warning("导入链预载失败: %s（%s）—— 不影响启动", name, e)
    used = time.time() - t_all
    logger.info("导入链预载总耗时 %.2fs（档位=%s）", used, s._PRELOAD)
    if used > s._PRELOAD_BUDGET:
        logger.warning(
            "预载耗时 %.1fs 超出预算 %.1fs：握手会推迟到 %.1fs 左右，"
            "有错过宿主工具注册窗口（约 15s）的风险。想缩短请调小 "
            "HIPPOCAMPUS_PRELOAD 的预载集。",
            used, s._PRELOAD_BUDGET, used + s._HANDSHAKE_BASE_COLD,
        )


def _start_warmup_watchdog() -> None:
    """按 30/60/120s 逐档检查预热是否完成，未完成就 dump 全部线程栈。

    与预热模式无关：卡死可以发生在后台线程（auto），也可以发生在主线程
    （off 模式的惰性初始化），只看 auto 会漏掉一半现场。
    """
    s = _srv()

    def _dump() -> None:
        import traceback

        deadline = 30.0
        while not s._warmup_ready.is_set():
            if not s._warmup_ready.wait(min(deadline, s._WARMUP_WAIT_TIMEOUT)):
                if s._warmup_ready.is_set():
                    break
                frames = sys._current_frames()
                for tid, frame in frames.items():
                    tname = next(
                        (x.name for x in threading.enumerate() if x.ident == tid),
                        str(tid),
                    )
                    logger.warning(
                        "DIAG[%.0fs] 预热未完成，线程 %s 栈:\n%s",
                        deadline, tname,
                        "".join(traceback.format_stack(frame)[-14:]),
                    )
                deadline = min(deadline * 2, s._WARMUP_WAIT_TIMEOUT)
                if deadline >= s._WARMUP_WAIT_TIMEOUT:
                    break

    threading.Thread(target=_dump, name="hippo-diag", daemon=True).start()


# 只需要「读库存」的工具（不需要嵌入模型）：等 _store_ready 就能干活。
# 实测：加载 9359 条库存约 5 秒，而加载 BGE 模型要 25-100 秒不等——
# 让这类工具陪着等模型，等于把秒级操作拖成一分钟级（且首调最容易被宿主判超时）。
# 未列出的工具一律按 embed 级处理（保守＝安全，只影响延迟不影响正确性）。
STORE_ONLY_TOOLS = frozenset({
    "get_memory", "forget_memory",
    "list_memories", "list_sources", "get_stats",
    "save_state", "load_state", "save_brain", "reset_context",
    "complete_intent", "report_feedback",
    "entity_profile", "entity_list", "lexicon_suggest",
    "lexicon_alias_suggest", "contradictions",
    "brain_stats",
    # 主题层只读派生表（空表时会自动梳理一次，用的是已加载的向量，
    # 不必等嵌入模型）
    "knowledge_map",
    # 源漂移审计只读库存 + 源签名层（文件 mtime/sha1），
    # 完全不碰向量 ⇒ store 级就够，不必为它等 25-33 秒模型预热。
    "drift_audit",
    # 地图层四工具只读写 maps.json 侧车 + 真相源文件，
    # 不碰库存不碰向量 ⇒ store 级（真实延迟只花在等库存加载完）。
    "map_set", "map_get", "map_list", "map_verify",
    # ⚠️ knowledge_topic **刻意不在本表**：它查出主题 id 依赖主题层已被
    # 建立（空库首调会 refresh 一次），store 级放行会表现成「没有这个
    # 主题」——那比多等几十秒难查得多。写回归纳本身不用向量，但为的是
    # 不出现这种静默失败，宁可按 embed 级保守等待。
    # 库管理类工具只需要目录元信息，不必等嵌入模型（那要 25-33 秒）
    "library_list", "library_info", "library_whoami", "library_use",
})


async def _await_warmup(tier: str | None = None) -> None:
    """MCP 工具入口的第一句：等后端就绪到所需层级再往下走。

    Args:
        tier: ``"store"`` 只用库存（读/列/统计/按 id 删），几秒就能开始；
              ``"embed"`` 还要用到嵌入模型（检索 / 写入 / 巩固），必须等模型到位。
              **留空则按调用者自动判定**（见下），工具一律写 `await _await_warmup()`，
              新增工具不必记住该选哪一级。

    自动判定用调用函数名在 ``STORE_ONLY_TOOLS`` 里查表：工具本身决定是否向量相关，
    调用点不用重复表达。之所以不在每个调用点显式传层级：30 处调用点改动面太大，
    且外部验收脚本按字面统计 `await _await_warmup()`
    的覆盖数——保持零参数写法才能让两边一致。

    - mode=off / 未曾启动后台预热（例如被脚本直接 import 调用）→ 立即返回；
    - 目标层级已就绪 → 立即返回；
    - 未就绪 → 在线程池等待（**不占事件循环**，其他 MCP 消息照常处理）。
      等待有**预算**（``_WARMUP_WAIT_TIMEOUT``，默认 90s，小于宿主 120s 超时），
      超预算按 ``_WARMUP_WAIT_POLICY`` 处置：raise（默认）抛
      :class:`WarmupPendingError`，release 则放行交给惰性初始化。

    为什么超时默认**抛错而不是放行**：放行会让调用方在"后端仍未就绪"的状态下
    继续跑（多数路径接着就去碰嵌入模型），于是再等几十秒——宿主先超时，用户
    看到的是"调用超时"而不是"还在预热"。抛错则是一条**可解析、可重试**的状态。
    另外预热**失败**（而非慢）时立即抛 :class:`WarmupFailedError` 并带原始原因，
    不让"配置错了"伪装成"性能问题"。

    状态与模式常量经 :func:`_srv` 现取（server 命名空间的
    monkeypatch 面）。
    """
    if tier is None:
        try:
            caller = sys._getframe(1).f_code.co_name
        except ValueError:  # pragma: no cover
            caller = ""
        tier = "store" if caller in STORE_ONLY_TOOLS else "embed"
    s = _srv()
    event = s._store_ready if tier == "store" else s._warmup_ready
    if s._WARMUP_MODE != "auto" or not s._warmup_started:
        return
    if event.is_set():
        # 就绪了也要看一眼是否"失败后放行"——致命失败必须当场说清楚，
        # 否则调用方会继续往下走，再撞一次更晦涩的错。
        if s._warmup_fatal and s._warmup_error:
            raise WarmupFailedError(
                f"后端初始化失败：{s._warmup_error}。这不是超时问题——请检查记忆库"
                f"目录、模型路径与依赖（HIPPOCAMPUS_WARMUP_DIAG=1 可拿线程栈）。"
            )
        return
    started = time.time()
    logger.info("后台预热进行中（层级=%s），本次调用等待其完成…", tier)
    await asyncio.to_thread(event.wait, s._WARMUP_WAIT_TIMEOUT)
    if event.is_set():
        return
    waited = time.time() - started
    if s._warmup_error:
        raise WarmupFailedError(
            f"后端预热失败：{s._warmup_error}。这不是超时问题——请检查模型路径/"
            f"依赖是否就绪（HIPPOCAMPUS_WARMUP_DIAG=1 可拿到各阶段耗时与线程栈）。"
        )
    msg = (
        f"后端仍在预热（已等待 {waited:.0f}s，当前阶段：{s._warmup_phase or '启动中'}）——"
        f"本次调用未执行，**稍后重试即可**，不是数据或参数问题。"
        f"若持续如此，用 HIPPOCAMPUS_WARMUP_DIAG=1 重启获取现场。"
    )
    if s._WARMUP_WAIT_POLICY == "raise":
        logger.warning("等待后台预热超预算（层级=%s，%.0fs）：%s", tier, waited, msg)
        raise WarmupPendingError(msg)
    logger.warning(
        "等待后台预热超时（层级=%s，%.0fs），按 policy=release 转由惰性初始化处理",
        tier, waited,
    )


def warmup_status() -> dict[str, Any]:
    """后端就绪状态自述（**零等待**，立即返回）。

    存在的理由：预热期的"慢"与"坏"在调用方看来是一样的（都是没响应），
    所以需要一个能被**主动查询**的 ready 探针——先问一句，再决定要不要调用，
    比反复试错调用省事得多。（懒加载服务给个 ready 端点是通用做法；
    这里挂在 ``brain_stats`` 的返回里，不新增工具。）

    单进程内多线程共享同一份状态，所以任何工具调用期间查到的都是同一事实：
    ``ready=false`` 且 ``phase`` 有值 = 正在预热；``error`` 非空 = 预热已失败
    （此时带向量的工具会立刻抛错，不必等满预算）。

    全部字段经 :func:`_srv` 现取——测试对 server 命名空间的
    赋值（``srv._warmup_error = ...`` 等）在本探针里立即可见。
    """
    s = _srv()
    return {
        "mode": s._WARMUP_MODE,
        "ready": s._warmup_ready.is_set(),
        "store_ready": s._store_ready.is_set(),
        "started": s._warmup_started,
        "phase": s._warmup_phase,
        "error": s._warmup_error,
        # fatal=True → 核心层没建起来，带向量的工具会当场报错而不是放行；
        # fatal=False 且 error 非空 → 降级可用（例如仅仅 spaCy 没装上）
        "fatal": s._warmup_fatal,
        "wait_timeout": s._WARMUP_WAIT_TIMEOUT,
        "wait_policy": s._WARMUP_WAIT_POLICY,
        "persist_dir": s._PERSIST_DIR,
    }
