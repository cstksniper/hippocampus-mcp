"""侧车文件的短临界区锁。

**要解决的具体病**：``maps.json`` / ``file_registry.json`` 原来是
「读 → 改内存 → tmp+rename 写」。tmp+rename 只保证**不留半个文件**，
保证不了**不丢更新**：

    daemon 会话扫描:  load(v1) ──────────────── save(v1+扫描结果)
    CLI maps.py set:        load(v1) ── save(v1+新条目)
                                          ↑ 后写的把前一个的改动整个抹掉

两个都是合法写入，谁也没报错，结果一个进程的活白干。这比崩溃更阴——
崩溃会留痕迹，这叫**静默**丢更新。

**为什么不用 filelock**：主库的跨进程保护依赖 ``filelock``，而它是可选
依赖（本机 3.13 就没有，见 vector_store 的降级注释）。侧车是 opt-in 的
辅助数据，不该因为缺一个第三方包就整块不可用，也不该把「缺依赖」变成
「静默无锁」。所以这里用**标准库自足的原子发布 + 可接管**，与
``launcher.acquire_spawn_lock`` 同一范式，但独立实现（导 launcher 会把
daemon 启动链拉进只想写地图的 CLI）。

**三条设计约束**：

1. **非阻塞优先、不排队**。侧车写入是毫秒级临界区，抢不到就如实说
   「忙」，由调用方决定等还是跳过。绝不无限阻塞——持锁者可能是个
   已经被杀的会话，阻塞会把所有人一起堵死（launcher 曾因双 daemon
   事故吃过阻塞式互斥的亏）。
2. **崩溃可接管**。锁里带 ``{pid, ts, token}``：持有者进程没了、或超过
   ``ttl`` 秒，判定陈旧、清理后重试一次。TTL 是最后一道兜底（pid 可能
   被复用），不是主要判据。
3. **只删自己的锁**。释放时校验 token，不匹配就不动——否则慢进程醒来
   会把别人的锁删掉，回到双持锁。
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid

#: 侧车锁最长持有时间（秒）。实测临界区：maps.json 全量写几十 ms，
#: file_registry 全量扫描受双预算约束（默认 5s）⇒ 60s 有充足余量。
#: 超时视为持有者已死（pid 复用的兜底）。
SIDECAR_LOCK_TTL = 60.0

#: 锁文件「刚出现、内容还没写完」的宽限期。正常路径用 os.link 原子发布，
#: 内容一定完整；只有退化路径（文件系统不支持硬链接）才可能出现半吊子
#: 内容，那时靠这个宽限别急着抢。
SIDECAR_LOCK_GRACE = 2.0

#: 轮询间隔（``wait`` 模式）。
_POLL_S = 0.02

#: 重入哨兵：同一线程已持锁时再次 acquire 返回它，release 时什么都不做。
REENTRANT = object()


class SidecarBusy(RuntimeError):
    """侧车锁被别的进程/线程持有，且调用方不愿等。

    **这是显式的**：调用方必须决定「重试 / 跳过本轮 / 报给用户」，
    不允许把它当成写入成功（静默丢更新正是本模块要消灭的东西）。
    """

    def __init__(self, path: str, holder: dict | None = None):
        self.path = path
        self.holder = holder or {}
        who = ""
        if self.holder:
            who = "（pid=%s，持锁 %.1fs）" % (
                self.holder.get("pid", "?"),
                max(0.0, time.time() - float(self.holder.get("ts") or 0)))
        super().__init__("侧车正被另一个进程写入%s：%s" % (who, path))


class SidecarUnreadable(RuntimeError):
    """侧车文件**存在但读不出来**（JSON 损坏 / 被独占 / 权限）。

    🔴 必须与「不存在」区分开：``load`` 对两者都返回 None，若写路径据此
    起一个空侧车（``create=True``）再整量写回，一次瞬时读失败就会把
    **整个地图库清空**。丢更新还能靠下轮巡检重算，清库不可逆。
    所以这里显式报错，让人决定是修文件还是显式重建。
    """

    def __init__(self, path: str, reason: str = ""):
        self.path = path
        self.reason = reason
        super().__init__(
            "侧车文件存在但读不出来%s：%s（不会当作不存在而覆盖——"
            "确需重建请显式 force）" % (("（%s）" % reason) if reason else "",
                                       path))


_tls = threading.local()


def lock_path(target: str) -> str:
    """侧车文件对应的锁文件：``maps.json`` ⇒ ``.maps.json.lock``。

    放在同目录（同一卷）——``os.link`` 不能跨卷。
    """
    full = os.path.abspath(target)
    d, base = os.path.split(full)
    return os.path.join(d, "." + base + ".lock")


def _held() -> dict[str, str]:
    paths = getattr(_tls, "paths", None)
    if paths is None:
        _tls.paths = paths = {}
    return paths


def _read_lock(path: str) -> dict | None:
    """读锁内容。读不出 / 格式不对 ⇒ None（**不等于没人持锁**，见陈旧判定）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError):
        return None
    if isinstance(info, dict) and isinstance(info.get("ts"), (int, float)):
        return info
    return None


def pid_alive(pid: int) -> bool:
    """只读探活，与 ``launcher._pid_alive`` 同口径。

    🔴 Windows 上**不能**用 ``os.kill(pid, 0)``——那里的 os.kill 对任意
    非 CTRL_* 信号会走 TerminateProcess，探活变成杀人。所以走
    OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)。
    """
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(0x1000, False, pid)
            if h:
                k32.CloseHandle(h)
                return True
            return False
        except Exception:  # noqa: BLE001 —— 探不了就当活着，宁可等 TTL
            return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def is_stale(path: str, ttl: float = SIDECAR_LOCK_TTL,
             grace: float = SIDECAR_LOCK_GRACE) -> bool:
    """锁是否可接管。

    🔴 **内容读不出 ≠ 没人持锁**：对方刚建出文件、还没写完 JSON 的那个
    窗口里，判断成「可抢占」就会双持锁。文件存在但内容残缺 ⇒ 按 mtime
    给一个宽限期。
    """
    info = _read_lock(path)
    if info:
        pid = info.get("pid")
        # 🔴 pid == 自己 ⇒ **一律不判陈旧**，先于 TTL 判断：那只能是同
        # 进程另一线程持锁（崩掉的进程 pid 不可能等于当前进程）。若这里
        # 因超时判陈旧，对方线程还在写、这边就把锁删了 ⇒ 双持锁。
        # 代价是同进程某线程卡死时要等它自己结束，可接受。
        if isinstance(pid, int) and pid == os.getpid():
            return False
        ts = float(info.get("ts") or 0.0)
        if ts and time.time() - ts > ttl:
            return True
        if isinstance(pid, int) and pid > 0:
            return not pid_alive(pid)
        return False
    try:
        age = time.time() - os.stat(path).st_mtime
    except OSError:
        return True                       # 文件不存在 ⇒ 确实没人持锁
    return age > grace


def acquire(path: str, wait: float = 0.0,
            ttl: float = SIDECAR_LOCK_TTL,
            grace: float = SIDECAR_LOCK_GRACE):
    """抢锁。成功返回 token（或 :data:`REENTRANT`），抢不到返回 ``None``。

    ``wait`` > 0 时最多等这么多秒（短轮询）；默认不等待。
    """
    key = os.path.abspath(path)
    held = _held()
    if key in held:
        return REENTRANT                  # 同一线程重入：锁已经在自己手上

    try:
        os.makedirs(os.path.dirname(key) or ".", exist_ok=True)
    except OSError:
        return None

    token = "%d:%s" % (os.getpid(), uuid.uuid4().hex[:12])
    deadline = time.time() + max(0.0, float(wait))
    while True:
        # 陈旧判定 + 清理（确认陈旧之后才删，绝不无条件 rm）
        if is_stale(key, ttl=ttl, grace=grace):
            try:
                os.remove(key)
            except OSError:
                pass
        if _publish(key, token):
            held[key] = token
            return token
        if time.time() >= deadline:
            return None
        time.sleep(_POLL_S)


def _publish(lock: str, token: str) -> bool:
    """原子发布锁文件。

    🔴 锁文件必须「一出现就是完整的」：先写临时文件 + fsync，再用
    ``os.link`` 挂到锁路径——硬链接创建是原子的，目标已存在必然失败，
    互斥语义一点没丢，半吊子窗口彻底消失。
    """
    tmp = "%s.tmp.%d.%s" % (lock, os.getpid(), token.split(":")[-1])
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "ts": time.time(),
                       "token": token}, f)
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        _unlink(tmp)
        return False
    try:
        os.link(tmp, lock)
        return True
    except FileExistsError:
        return False                      # 正常竞争输给别人
    except OSError:
        return _publish_legacy(lock, token)   # 不支持硬链接 ⇒ 退化
    finally:
        _unlink(tmp)


def _publish_legacy(lock: str, token: str) -> bool:
    """``os.link`` 不可用时的退化（例如某些网络文件系统）。

    仍是 ``O_CREAT|O_EXCL``，**半吊子窗口还在**——由 :func:`is_stale`
    的 ``grace`` 兜着：对方有 grace 秒把内容写完。
    """
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError:
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "ts": time.time(),
                       "token": token}, f)
            f.flush()
            os.fsync(f.fileno())
        return True
    except OSError:
        _unlink(lock)
        return False


def release(path: str, token) -> bool:
    """释放锁。**只删自己的**：内容里的 token 对不上就不动。

    返回是否真的删掉了（``REENTRANT`` 返 False，表示外层还持着）。
    """
    if token is REENTRANT or token is None:
        return False
    key = os.path.abspath(path)
    _held().pop(key, None)
    info = _read_lock(key)
    if not info or info.get("token") != token:
        return False                      # 已经被接管 / 不是自己的
    return _unlink(key)


def _unlink(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def holder(path: str) -> dict | None:
    """当前持锁者信息（诊断用，不参与判定）。"""
    return _read_lock(path)


def quarantine(path: str) -> str:
    """把读不出来的侧车改名隔离（``.corrupt.<ts>``），返回新路径。

    **不直接删**：坏文件里可能还有几百条人工录的状态，删了就真没了。
    改名之后写路径可以安全重建，人也能拿它手工抢救。
    """
    dst = "%s.corrupt.%d" % (path, int(time.time()))
    try:
        os.replace(path, dst)
        return dst
    except OSError:
        return ""
