# -*- coding: utf-8 -*-
"""多库寻址区（server.py 多库寻址区块的实现本体）。

这里实现：工作区引导文件（``GUIDE_FILENAME`` / ``_find_workspace_guide``）、
默认库解析（``_resolve_default_lib_path``）、会话 key（``session_key``）、
库清单 TTL 缓存扫描（``libraries``）、三级优先库解析
（``resolve_library_path``：**显式 > 粘性 > 默认**）、会话级粘性绑定
（表 + 锁 + ``set/get/clear_library_binding`` 三件）、库自曝
（``_library_self``）、信封装配（``_with_library``）、跨库端口 I/O 助手
（``_ensure_default_dir`` / ``_layer_files`` / ``_prefetch_library``）。

🔴 状态与常量**刻意不搬**——留在 server.py（测试的 monkeypatch 面）：
``_DEFAULT_LIB_PATH``（赋值 patch）、``_conn_default_lib`` /
``_conn_lib_lock``（clear）、``_library_cache`` /
``_library_cache_at`` / ``_LIBRARY_CACHE_TTL``（置 None 后
重扫）、``_registry`` / ``_registry_lock`` / ``_registry_get``（patch）、
``_PERSIST_DIR``。本模块对这些符号的读写一律经 :func:`_srv`
**现取**（教训：快照进新模块 = monkeypatch 静默失效）；对 server 侧
函数（``default_library_path`` / ``_registry_get``）的调用同样现取——
测试替换 ``server._brain_for`` / ``server._registry_get`` 才能一路生效到
寻址与预取。粘性绑定表（``_library_binding`` / ``_library_binding_lock``）
随本模块走：全仓无直接 patch 面（外部只经 set/get/clear API 操作）。

server.py 侧按 import 名原样再导出本模块全部公共符号，既有调用点
（工具体的三级解析 / 库自曝首键 / 信封装配）与测试直调
（``S.resolve_library_path`` / ``S._find_workspace_guide`` /
``S.GUIDE_FILENAME`` …）零改动。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from hippocampus_mcp import library as _library
from hippocampus_mcp.common import session_ctx as _session_ctx
from hippocampus_mcp.library import LibraryResolutionError  # noqa: F401（resolve 签名要抛）

logger = logging.getLogger("hippocampus_mcp")


def _srv() -> Any:
    """server 模块本体（现取入口，见模块 docstring 的 monkeypatch 面说明）。"""
    from hippocampus_mcp import server as _s

    return _s


# 工作区引导文件名：主目录下放这一个文件即可把默认库指到别处。
GUIDE_FILENAME = ".hippocampus.json"


def _find_workspace_guide(start: str | None = None) -> str | None:
    """从 ``start``（默认 cwd）向上找工作区引导文件（最多 5 层）。

    引导文件机制的**零配置便利项**：文件不存在就完全不生效，
    不改变任何既有行为 —— 主通道始终是调用侧显式寻址。
    """
    d = os.path.abspath(start or os.getcwd())
    for _ in range(5):
        cand = os.path.join(d, GUIDE_FILENAME)
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def resolve_default_lib_ex(
    persist_dir: str | None = None,
    start: str | None = None,
) -> tuple[str, dict[str, str] | None]:
    """默认库路径的**完整版**：``(路径, 回落详情)``。

    默认库路径：引导文件 > env ``HIPPOCAMPUS_PERSIST_DIR``。

    ``persist_dir`` / ``start`` 是**连接级**入口：daemon 一个
    进程服务 N 条连接，每条连接可能属于不同工作区，不能共用进程启动时算出的
    那一份。不传就走模块级 env / cwd，行为与改造前**逐位一致**。

    🔴 引导文件**只接受能明确解析成库的**值。解析不了就打一条 warning 并
    回退 env 默认——这是"便利项"该有的行为（它不该有能力让服务起不来），
    而且**出声**，不是静默。

    回落详情只在「声明了但没用上」时非 ``None``：``declared_raw``（原始
    声明名）/ ``reason``（``unresolvable`` 解析不明 | ``read_error`` 文件
    坏了或不是 JSON）/ ``detail``（异常或说明文本）。服务端日志 agent
    看不见（实测反馈 2026-10-31），这仨键是 session_start 顶层告警块的
    事实源——见 ``server._inject_fallback_warning``。

    ``_PERSIST_DIR`` 经 :func:`_srv` 现取（server 命名空间）。
    """
    s = _srv()
    base = _library.normpath(os.path.expanduser(
        persist_dir if persist_dir is not None else s._PERSIST_DIR))
    guide = _find_workspace_guide(start)
    if not guide:
        return base, None
    want = ""
    try:
        with open(guide, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        want = str((cfg or {}).get("library") or "").strip()
        if not want:
            return base, None
        info = _library.resolve_library(want, default_path=base)
        if info is None:
            logger.warning(
                "工作区引导文件 %s 的 library「%s」解析为空，回退 env 默认库",
                guide, want)
            return base, {"declared_raw": want, "reason": "unresolvable",
                          "detail": "resolve_library 返回 None"}
        logger.info("工作区引导文件 %s 指定默认库：%s（%s）", guide, info.name, info.path)
        return info.path, None
    except _library.LibraryResolutionError as e:
        # 指定了但解析不明：出声，且**不**拿它当默认库（串库比回退危险得多）
        logger.warning("工作区引导文件 %s 的 library 解析不明，回退 env 默认库：%s", guide, e)
        return base, {"declared_raw": want, "reason": "unresolvable",
                      "detail": str(e)}
    except Exception as e:      # 文件坏了 / 不是 JSON：同样出声回退
        logger.warning("工作区引导文件 %s 读取失败，回退 env 默认库：%s", guide, e)
        return base, {"declared_raw": want, "reason": "read_error",
                      "detail": str(e)}


def _resolve_default_lib_path(persist_dir: str | None = None,
                              start: str | None = None) -> str:
    """默认库路径（薄壳）：引导文件 > env。回落详情丢弃，签名/返回值
    与拆分前**逐位一致**——既有测试桩按旧签名 patch，不能破；要回落
    详情走 :func:`resolve_default_lib_ex`。
    """
    path, _fallback = resolve_default_lib_ex(persist_dir, start)
    return path


# 会话级粘性绑定：session key -> 库路径。
# key 空间与全仓会话身份约定完全一致：**显式 client > ctx.session > __default__**。
# 直连 stdio 下只有 __default__ 一路（一个进程＝一个会话）；
# daemon 接入 ctx 会话之后自动按连接分路，这里不用改。
# 表与锁随寻址区走（全仓无直接 patch 面，外部只走 set/get/clear API）。
_library_binding: dict[str, str] = {}
_library_binding_lock = threading.RLock()


def session_key(client: str | None = None) -> str:
    """本次调用的会话 key（与全仓其他会话身份消费方共用同一命名空间）。

    显式 ``client`` 署名优先级最高（调用方自己报身份）；
    没署名时读 :mod:`session_ctx` 的上下文 key——**直连 stdio 下恒为
    ``__default__``**（一进程只服务一个会话，行为与改造前完全一致），
    daemon 化之后由每条连接在拉起会话前绑定，于是同一个 daemon 进程里的
    N 个会话各自落在独立 key 上，``library_use()`` 的粘性不会串。

    session_ctx 在本模块直接 import，不经 server。
    """
    c = (client or "").strip()
    if c:
        return c
    return _session_ctx.current_session()


def libraries(force: bool = False) -> list[Any]:
    """已发现的库清单（带 TTL 缓存）。**只读**，不创建也不修改任何文件。

    缓存三态（``_library_cache`` / ``_library_cache_at`` /
    ``_LIBRARY_CACHE_TTL``）留在 server.py（monkeypatch 面），经
    :func:`_srv` 现取——测试置 ``None`` 后本函数立即重扫。
    """
    s = _srv()
    now = time.time()
    if (not force and s._library_cache is not None
            and now - s._library_cache_at < s._LIBRARY_CACHE_TTL):
        return s._library_cache
    try:
        found = _library.scan_libraries()
    except Exception as exc:  # noqa: BLE001 —— 扫描失败不该让工具挂掉
        logger.warning("库扫描失败: %s", exc)
        found = []
    s._library_cache = found
    s._library_cache_at = now
    return found


def resolve_library_path(library: str | None = None,
                         client: str | None = None) -> str:
    """决定本次调用落到哪个库（返回归一后的绝对路径）。

    优先级：**显式 library > 会话级粘性 > 默认库**。

    Raises:
        LibraryResolutionError: 显式指定了但解析不明（含重名歧义）。
            **绝不静默回落到默认库** —— 那是串库的入口。

    库清单/默认库经 :func:`_srv` 现取（``s.libraries()``
    / ``s.default_library_path()``——server 命名空间的 patch 面一路生效）；
    粘性绑定表随本模块（无直接 patch 面，外部只走 set/get/clear API）。
    """
    s = _srv()
    q = (library or "").strip()
    if q:
        info = _library.resolve_library(
            q, libs=s.libraries(), default_path=s.default_library_path(client))
        if info is None:  # 理论上不会（q 非空时要么命中要么抛）
            raise LibraryResolutionError(f"库 {q} 解析失败")
        if info.backup:
            raise LibraryResolutionError(
                f"{info.name} 是备份/快照目录，只读，不能作为写入目标。"
                f"请用真库名或 library_id 指定。")
        return info.path
    with _library_binding_lock:
        bound = _library_binding.get(session_key(client))
    if bound:
        return bound
    return s.default_library_path(client)


def set_library_binding(path: str, client: str | None = None) -> str:
    """会话级粘性绑定（``library_use`` 的落点）。返回生效的 key。"""
    key = session_key(client)
    with _library_binding_lock:
        _library_binding[key] = _library.normpath(path)
    return key


def get_library_binding(client: str | None = None) -> str:
    """当前会话绑定的库路径（未绑定则返回默认库路径）。"""
    with _library_binding_lock:
        return _library_binding.get(session_key(client)) or _srv().default_library_path(client)


def clear_library_binding(client: str | None = None) -> bool:
    """解除绑定（回到默认库）。返回是否真的解除了。"""
    key = session_key(client)
    with _library_binding_lock:
        return _library_binding.pop(key, None) is not None


def _library_self(library: str | None = None) -> dict[str, str]:
    """库自曝：本次调用的目标库（``{id, name, path}``）。

    写类返回体的**首键**就是它。串库之所以可怕，是因为写错库与写对库
    在返回体里长得一模一样；这一行把差别摆到最前面。

    解析与清单经 :func:`_srv` 现取（``s.resolve_library_path``
    / ``s.libraries()``）。
    """
    s = _srv()
    path = s.resolve_library_path(library)
    name = ""
    lid = ""
    for i in s.libraries():
        if i.path == path:
            name, lid = i.name, i.library_id
            break
    if not lid:
        lid = _library.library_id_for(path)
        name = _library._guess_name(path)
    return {"id": lid, "name": name, "path": path}


def _with_library(payload: Any, library: str | None = None) -> Any:
    """给写类返回体插上 ``library`` 首键。**只有 dict 才插**。

    非 dict 返回（错误分支的字符串、列表等）原样返回 —— 不加键，
    免得把错误路径的结构改坏。

    走 application 服务编排的工具不再走本函数——信封由服务层自产
    （``BaseService._envelope``，与本函数同一逐位规则）。信封装配与错误体
    规则的三条硬约束（只对 dict 插键、错误分支非 dict 原样、两侧用同一
    逐位序列化参数）在本函数与 ``BaseService._envelope`` 两个装配点都必须
    逐位一致；``_map_fields_error``（map_set fields 解析错误的信封装配）
    是本函数的现役使用者之一。

    库自曝经 :func:`_srv` 现取（``s._library_self``）。
    """
    if isinstance(payload, dict):
        return {"library": _srv()._library_self(library), **payload}
    return payload


# ------------------------------------------------------------------
# 跨库用例端口实现（连接态与 I/O 留在适配层，用例只拿能力）
# （三个 I/O 助手随寻址区走；注册表经 _srv() 现取）
# ------------------------------------------------------------------
def _ensure_default_dir(path: str) -> bool:
    """``os.makedirs(path, exist_ok=True)`` 的布尔版：失败**出声**并返回 False。

    零库首启这条路径是 ``library_list`` 唯一会创建东西的分支；建目录失败不该
    让列表工具挂掉，但不能静默——「自动建了默认库」是会被人当真的自曝。
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError as e:
        logger.warning("零库首启自动建 default 失败（不影响使用）: %s", e)
        return False


def _layer_files(path: str) -> dict[str, Any]:
    """库目录里的层文件名 → 字节大小（``brain_*`` / ``memories.json`` / 清单）。

    目录不可读时把原因放进 ``_error``：自曝优先于可用性——宁可返回带错误的
    结构，也不要让工具因为 stat 失败整个挂掉。
    """
    layers: dict[str, Any] = {}
    try:
        for fn in sorted(os.listdir(path)):
            if fn.startswith("brain_") or fn in ("memories.json",
                                                 _library.MANIFEST_NAME):
                fp = os.path.join(path, fn)
                if os.path.isfile(fp):
                    layers[fn] = os.path.getsize(fp)
    except OSError as e:
        layers["_error"] = str(e)
    return layers


def _prefetch_library(path: str) -> None:
    """后台线程预取某库的 brain（切库后避免首次调用自付 25–33 秒加载）。

    注册表经 :func:`_srv` 现取——测试替换
    ``server._registry_get`` 后预取拿到的是替身。
    """
    def _prefetch() -> None:
        try:
            _srv()._registry_get().get(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("库预取失败（不影响后续惰性加载）: %s", e)

    threading.Thread(target=_prefetch, name="hippo-lib-prefetch",
                     daemon=True).start()
