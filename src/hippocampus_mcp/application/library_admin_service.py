# -*- coding: utf-8 -*-
"""LibraryAdminService：跨库管理用例（5 工具）。

工具面（原「Maintain 11」的另一半）：library_list / library_info /
library_whoami / library_use / library_create。

**为什么不并进 MaintainService / 为什么不继承 ``BaseService``**：这 5
个工具的共同点是**不绑定单一 brain**——

* ``library_info`` 刻意不构造 HippoBrain（只读目录元信息，不必等嵌入模型
  25–33 秒），而 ``BaseService`` 的构造契约是「一个已解析的目标库 + 它的
  brain」。若让它们走同一个工厂，服务一构造就把 brain 造出来，这恰恰是其
  设计意图要避开的东西；
* ``library_use`` / ``library_create`` / ``library_list`` 操作的对象是**库集
  合与会话绑定**，不是某一个库的内容。

故本类自己一个，依赖全部经 :class:`LibraryPorts` **构造注入**：storage
（注册表/绑定表/连接级默认库）留在适配层（server.py），用例只声明要什么。
这与 base.py 已写的那条边界同向——「服务不碰库发现/会话粘性那套连接态
状态」，这里把它落到接口上而不是只写在 docstring 里。

编排逻辑与 server.py 工具体逐字对应（去 ``json.dumps``），只挪位置不改行为。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

from ..library import (LibraryInfo, LibraryResolutionError, default_roots,
                       library_id_for, normpath, resolve_library,
                       write_manifest)
# 私有借用说明：库名回退推断在 library 包里只有这一个实现（server.py 当初
# 也是以 ``_library._guess_name`` 用它）。与其在服务侧重抄一份命名逻辑（两份
# 迟早漂移），不如显式留这行借用，将来升为公开 API 时一处改完。
from ..library import _guess_name
#: ``whoami`` 的 ``permission`` 键与写类守卫同一判定
#: 函数——自曝口径必须与拦截口径同源，否则又是「报白的和拦的不是一回事」。
from ..domain.write_boundary import classify_write_permission

logger = logging.getLogger("hippocampus_mcp")


@dataclass(frozen=True)
class LibraryPorts:
    """跨库用例的外部能力清单（组合根用 server.py 的连接态函数填空）。

    每一条都是**活在适配层的状态或 I/O**：库清单 TTL 缓存、连接级默认库、
    会话绑定表、brain 注册表、``os.makedirs`` / ``os.listdir``。用例拿到的是
    「能力」不是「状态本身」——试举反例：直接把 ``_library_binding`` 字典传
    进来，就等于让用例层持有连接态。
    """

    #: ``libraries(force)``：带 TTL 缓存的只读扫描（零库不建任何东西）
    libraries: Callable[[bool], list[Any]]
    #: 本次调用语境下的默认库路径（连接级 > 进程级）
    default_path: Callable[[], str]
    #: ``resolve_library_path(name_or_none)``：显式 > 粘性 > 默认，解析不明抛错
    resolve_path: Callable[[str | None], str]
    #: 当前会话绑定的库路径（未绑定返回默认库）
    binding_get: Callable[[], str]
    #: ``set_library_binding(path, client)``：返回生效的 session key
    binding_set: Callable[[str, str | None], str]
    #: ``_brain_for(path)``：**只**给 ``create`` 用（建库后初始化存储）
    brain_for: Callable[[str], Any]
    #: 常驻 brain 注册表统计（``whoami`` 的 ``resident`` 一节）
    registry_stats: Callable[[], dict[str, Any]]
    #: ``os.makedirs(path, exist_ok=True)`` 的布尔版：失败时适配层已出声
    ensure_dir: Callable[[str], bool]
    #: 目录里的层文件名 → 字节数（不可读时返回 ``{"_error": ...}``）
    layer_files: Callable[[str], dict[str, Any]]
    #: 后台线程预取某库的 brain（失败不影响绑定）
    prefetch: Callable[[str], None]
    #: 把新建的库登记进用户级任务目录
    #: （``<user 库>/directory.json``，建库流程的**唯一自动挂点**）。
    #: 入参是条目 dict（task/name/library_id/path/note），返回
    #: ``{"registered": bool, "reason"?: str, ...}``；失败由调用方自曝，
    #: 不抛给建库流程。
    task_directory_register: Callable[[dict[str, Any]], dict[str, Any]]


class LibraryAdminService:
    """库管理用例：列 / 查 / 自曝 / 切换 / 新建（发现与 manifest 写入在 library 包）。"""

    def __init__(self, ports: LibraryPorts) -> None:
        self.ports = ports

    # ---------------------------------------------------------------- 内部

    def _default_info(self) -> Any:
        """默认库的 LibraryInfo——**即便它不在任何扫描根下也要能自曝**。

        等价于 server 的 ``default_library_info()``（走同一个 default_path
        端口，故 daemon 下按连接算，行为不变）。
        """
        path = self.ports.default_path()
        for i in self.ports.libraries(False):
            if i.path == path:
                return i
        return LibraryInfo(path=path, library_id=library_id_for(path),
                           name=_guess_name(path), explicit=False)

    # ---------------------------------------------------------------- 列

    def list_libraries(self, *, include_backups: bool) -> dict[str, Any]:
        """列出已发现的库。**只读**——唯一例外是零库首启建出默认目录。

        备份目录默认不列（它们是只读快照，不能当写入目标，盘点磁盘占用才
        开 ``include_backups``）；``auto_created`` 是「这次真的建了目录」的
        自曝——此前零库时 list 返回空，用户看不出下一步该干什么。
        """
        ports = self.ports
        infos = ports.libraries(True)
        rows: list[dict[str, Any]] = []

        def _collect(infos_: object) -> None:
            for i in infos_:
                if i.backup and not include_backups:
                    continue
                d = i.to_dict()
                d["is_default"] = (i.path == ports.default_path())
                d["is_bound"] = (i.path == ports.binding_get())
                rows.append(d)

        _collect(infos)

        # 零库首启自动建 default。
        auto_created = None
        _cur_default = ports.default_path()
        if not rows and _cur_default:
            if ports.ensure_dir(_cur_default):
                auto_created = _cur_default
                rows.clear()
                _collect(ports.libraries(True))

        return {
            "libraries": rows,
            "count": len(rows),
            "default_path": _cur_default,
            "bound_path": ports.binding_get(),
            "roots": default_roots(),
            "auto_created": auto_created,
        }

    # ---------------------------------------------------------------- 查

    def describe(self, *, library: str) -> dict[str, Any]:
        """某个库的详情（路径/id/名字/别名/层文件）。**刻意不构造 brain**。

        ``resolve_path`` 抛错（库名不明 / 指定到备份目录）时不吞——「不确定
        你说的是哪个库」必须让调用方看见，**绝不静默回落到默认库**。
        """
        ports = self.ports
        path = ports.resolve_path(library or None)
        info = None
        for i in ports.libraries(False):
            if i.path == path:
                info = i
                break
        if info is None:
            info = self._default_info() if path == ports.default_path() else None
        if info is None:
            info = LibraryInfo(path=path, library_id=library_id_for(path),
                               name=_guess_name(path))
        out = info.to_dict()
        out["is_default"] = (info.path == ports.default_path())
        out["is_bound"] = (info.path == ports.binding_get())
        out["layer_files"] = ports.layer_files(path)
        return out

    # ---------------------------------------------------------------- 自曝

    def whoami(self) -> dict[str, Any]:
        """本次调用会落到哪个库 + 能写谁——防串库的第一道自曝（含 ``source`` 依据）。

        三键：

        * ``home``——本次连接的**可写家库**（连接级默认库，写永远钉在这）；
        * ``current``——**寻址落点**（``library_use`` 后可以是他库，读落这）；
        * ``permission``——现在往 ``current`` 写会怎样：``read-write``（家库）/
          ``managed-write``（user 受管通道）/ ``read-only``（会被守卫拦下）。

        口径纪律：``permission`` 直接调**守卫同款**纯函数
        （``domain.write_boundary.classify_write_permission``），不在这里
        另写一套判断——自曝与拦截必须同源，否则 agent 看到的和实际执行的是
        两回事。逃生门 env 是**进程级运维开关**，不在自曝里体现：开着时
        自曝仍报 ``read-only``（偏保守，不会误导 agent 去跨库写）。

        ``library`` 是历史旧键，与 ``current`` 同值保留（工具面兼容）。
        """
        ports = self.ports
        bound = ports.binding_get()
        home_path = ports.default_path()
        is_bound = bound != home_path
        info = None
        for i in ports.libraries(False):
            if i.path == bound:
                info = i
                break
        if info is None:
            info = self._default_info()
        home = self._default_info()
        current = {"id": info.library_id, "name": info.name, "path": info.path}
        return {
            "library": dict(current),
            "home": {"id": home.library_id, "name": home.name,
                     "path": home.path},
            "current": current,
            "permission": classify_write_permission(
                target_path=info.path, home_path=home_path,
                target_name=info.name),
            "source": "binding" if is_bound else "default",
            "default_path": home_path,
            "resident": ports.registry_stats(),
        }

    # ---------------------------------------------------------------- 切换

    def use(self, *, library: str, client: str) -> dict[str, Any]:
        """会话级切换：此后本会话工具自动落到该库。**不自动建库**（拼错即失败）。

        绑完再后台预取（首次调用不然要自付 25–33 秒加载）——顺序勿改：绑定先
        落，预取失败也不影响绑定本身。
        """
        ports = self.ports
        info = resolve_library(library, libs=ports.libraries(True),
                               default_path=ports.default_path())
        if info is None:
            raise LibraryResolutionError("library_use requires a library name")
        if info.backup:
            raise LibraryResolutionError(
                f"{info.name} is a backup/snapshot directory; it is read-only and cannot be switched to.")
        key = ports.binding_set(info.path, client or None)
        ports.prefetch(info.path)
        return {
            "status": "bound",
            "library": {"id": info.library_id, "name": info.name,
                        "path": info.path},
            "session_key": key,
            "note": "Read tools in this session now target this library automatically; writes remain pinned to the home library "
                    "(see library_whoami home). Pass library= for a one-call override.",
        }

    # ---------------------------------------------------------------- 新建

    def create(self, *, name: str, path: str, aliases: str, use: bool,
               client: str, workspace_root: str = "") -> dict[str, Any]:
        """新建一个记忆库（写 manifest + 初始化存储，可选立即切过去）。

        建库是**显式动作**：只有这里写清单，日常扫描绝不写盘。重名与「同名
        目录已存在且不是目录」都直接报错——歧义寻址的代价是写错库。

        建库成功后**顺手登记进用户级任务目录**（该 sidecar
        的唯一自动挂点，属受管通道）。登记失败**不阻断建库**，但必须
        出声——``directory_registered: false`` + ``directory_reason`` + 一条
        warning：静默漏登记会让新库「建成即隐身」（此后所有跨库寻址都找不到
        它，而用户以为登记过了）。
        """
        ports = self.ports
        nm = (name or "").strip()
        if not nm:
            raise LibraryResolutionError("Library name cannot be empty")
        for i in ports.libraries(True):
            if not i.backup and i.name.lower() == nm.lower():
                raise LibraryResolutionError(
                    f"Library name {nm} already exists ({i.path}). Duplicate names make addressing ambiguous; choose another.")
        target = (path or "").strip()
        if not target:
            target = os.path.join(default_roots()[0], "hippo_" + nm)
        target = normpath(os.path.expanduser(target))
        if os.path.exists(target) and not os.path.isdir(target):
            raise LibraryResolutionError(f"{target} already exists and is not a directory")
        alias_list = [a.strip() for a in aliases.split(",") if a.strip()]
        info = write_manifest(target, nm, alias_list)
        # 建完先把存储初始化出来：空目录不会在下次扫描前自动成为「已就绪的库」。
        ports.brain_for(target).save()
        out: dict[str, Any] = {
            "status": "created",
            "library": {"id": info.library_id, "name": info.name,
                        "path": info.path},
            "aliases": alias_list,
        }
        # 建库即登记进用户级任务目录。登记是赠品，绝不阻断
        # 建库；但失败必须自曝——「建成即隐身」比建库失败更难查。
        reg: dict[str, Any] = {"registered": False, "reason": "register_error"}
        try:
            entry = {"task": info.name, "name": info.name,
                     "library_id": info.library_id, "path": info.path,
                     "note": ""}
            if (workspace_root or "").strip():
                # C2 受管绑定（Q-B/Q-6 重审落地）：建库即绑定工作区——
                # onboarding「问一次、建一次、绑一次」的第三步由此一步到位。
                entry["workspace_root"] = workspace_root.strip()
            reg = ports.task_directory_register(entry)
        except Exception as exc:  # noqa: BLE001 —— 登记失败不许炸建库
            reg = {"registered": False, "reason": f"register_error: {exc}"}
        out["directory_registered"] = bool(reg.get("registered"))
        if not out["directory_registered"]:
            out["directory_reason"] = str(reg.get("reason") or "unknown")
            logger.warning(
                "Task directory registration failed (library %s was created but not added to the user-level task directory: %s). "
                "Cross-library lookup cannot find it by task name; add it to directory.json manually or check the user library.",
                info.name, out["directory_reason"])
        if use:
            out["session_key"] = ports.binding_set(info.path, client or None)
        # 实测反馈 2026-10-31：写边界钉在**连接级**家库上，建库当连接
        # ingest 写不进新库——没人提示的话 agent 要靠写入报错才知道，
        # 多付一轮试错。「建库 + 落首条」天然拆两次会话，这里把这件事
        # 在建库成功时就说清楚。提示是赠品：拿不到 home 名也不炸建库。
        try:
            out["note"] = (
                f"Write boundary note: writes on this connection remain pinned to the original home library"
                f" ({self._default_info().name}). To write the first memory into "
                f"{info.name}, start a new session (the new connection HELLO will bind by "
                "workspace_root to the new library). Read side can use library_use to switch immediately.")
        except Exception:  # noqa: BLE001 —— 提示是赠品，绝不炸建库
            out["note"] = (
                f"Write boundary note: writes on this connection remain pinned to the original home library; to write the first memory into "
                f"{info.name}, start a new session (the new connection HELLO will bind by "
                "workspace_root to the new library).")
        return out
