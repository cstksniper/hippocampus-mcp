# -*- coding: utf-8 -*-
"""launcher：极薄的启动器（**纯标准库**，不 import torch / 模型）。

为什么需要它：现在每个 MCP 子进程都要自己加载一份嵌入模型，
**单进程固定开销 755 MB**（与记忆条数无关）。N 个会话 = N 份。launcher 只做
三件事，加载完就变成一根管子：

    1. 算出「这个库 + 这个模型」的 daemon key
    2. 找 daemon：活着就接上；不在就拉起一个
    3. 把自己变成 stdio ↔ daemon socket 的**双向字节泵**

设计约束（改动前就定死，别在过程中漂移）：

- **纯标准库**。launcher 自己不能 import 重库，否则省下的内存又被它吃掉。
- **默认不启用**。``HIPPOCAMPUS_LAUNCHER=1`` 才走 daemon 路径；为 0 / 未设
  就退化成"直接跑 server"，行为与今天**逐位一致**。
- **daemon 拉不起来必须静默退化**，绝不能让宿主连不上 MCP —— 可用性优先于省内存。
- **多个会话共用一个 daemon**（只要 key 相同）。早期注释这里写的是"一条连接
  一个 daemon、不复用"，那是错的：``ensure_daemon`` 注册表命中就会复用，
  而复用正是"模型 N→1"的全部意义。会话分路靠的是 daemon 里**每条连接一个
  线程 + 独立的 ``asyncio.run``**（``session_ctx`` 的 ``conn-N``），
  不是靠"每个会话一个进程"。

  🔴 复用会带来串库（已修）：daemon 的默认库原本是**进程级**常量，
  后到的工作区会写进先到那个的库。现在 launcher 连上后先发一行带外握手
  （``session_ctx.HELLO_PREFIX``）声明自己的库，daemon 按连接记住它。
  改这条链路前先看 ``tests/test_f9_4e_conn_library.py``。
"""

# 豁免登记：本文件的 env 直读属启动拓扑 plumbing（launcher 开关 /
# 就绪超时 / announce 透传），封闭清单见 config.py「根层豁免清单」。
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from typing import Any

try:  # 纯标准库，零成本；但 launcher 绝不能因为 import 失败而起不来
    from hippocampus_mcp.common.session_ctx import HELLO_PREFIX
except Exception:  # noqa: BLE001 —— 拿不到就退回字面量（与 session_ctx 同值）
    HELLO_PREFIX = "HIPPO-HELLO "

# 启动器开关。默认关 —— 省内存是收益，连不上是事故，先保后者。
ENV_LAUNCHER = "HIPPOCAMPUS_LAUNCHER"
# daemon 注册表目录（默认系统临时目录下的 hippocampus_mcp）
ENV_RUNTIME_DIR = "HIPPOCAMPUS_RUNTIME_DIR"
# 记忆库目录。launcher 不解释它的语义，但**可写性**要在启动前兜一遍：
# 写不进的库等于所有记忆写入静默丢失，那不是"降级可用"，是"看起来可用"。
ENV_PERSIST = "HIPPOCAMPUS_PERSIST_DIR"
# 等 daemon 就绪的上限（秒）。模型加载是分钟级，给足但必须有上限。
DAEMON_READY_TIMEOUT = float(os.environ.get("HIPPOCAMPUS_DAEMON_READY_TIMEOUT", "180"))
# 字节泵单次读取的块大小
RELAY_CHUNK = 65536


def runtime_dir() -> str:
    """daemon 注册表落点。"""
    d = os.environ.get(ENV_RUNTIME_DIR, "").strip()
    if not d:
        import tempfile
        d = os.path.join(tempfile.gettempdir(), "hippocampus_mcp")
    return d


def launcher_enabled() -> bool:
    return os.environ.get(ENV_LAUNCHER, "0").lower() in ("1", "true", "yes", "on")


def daemon_key(library: str = "", model: str = "", version: str = "") -> str:
    """daemon 身份 = 库路径 + 模型 + 代码版本。三者任一不同就是不同的 daemon。

    模型必须进 key：库里存的向量是某个模型算出来的，换了模型就得换进程，
    否则 daemon 里那份权重与磁盘向量不是一回事，检索会静默变差。
    """
    try:
        lib = os.path.normcase(os.path.abspath(library)) if library else ""
    except Exception:  # noqa: BLE001 —— 路径异常不该让启动器起不来
        lib = str(library or "")
    raw = "|".join([lib, model or "", version or ""])
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:12]


def registry_path(key: str) -> str:
    return os.path.join(runtime_dir(), f"daemon-{key}.json")


def read_registry(key: str) -> dict[str, Any] | None:
    """读 daemon 登记。读不出 / 不是 dict / 没 port → None。"""
    try:
        with open(registry_path(key), "r", encoding="utf-8") as f:
            info = json.load(f)
    except Exception:  # noqa: BLE001 —— 读不到就是"没有 daemon"
        return None
    if not isinstance(info, dict):
        return None
    if not isinstance(info.get("port"), int) or not isinstance(info.get("pid"), int):
        return None
    return info


def write_registry(key: str, info: dict[str, Any]) -> bool:
    d = runtime_dir()
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return False
    path = registry_path(key)
    # tmp 名字带上 pid：两个 daemon 并发 announce 时（例如宿主先杀一个、
    # 第二个紧接着补位，或单飞锁被陈旧接管后放进来），共用一个 `.tmp` 会
    # 让其中一个在打开/替换阶段 OSError ⇒ announce 返回 False ⇒ daemon 直接
    # 退出且不打 banner —— 死得比"多起一个"更隐蔽（负向对照实测过）。
    tmp = path + f".{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def _pid_alive(pid: int) -> bool:
    """只读探测（与 brain.pid_alive 同口径，这里独立实现以保持纯标准库）。"""
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
        except Exception:  # noqa: BLE001
            return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def try_connect(port: int, timeout: float = 1.0) -> socket.socket | None:
    """探一下 daemon 端口通不通（连得上就算活着）。

    🔴 **连上之后必须把超时清掉**：``create_connection(timeout=...)`` 会把这个
    超时留在 socket 上，而这个 socket 随后要交给 :func:`relay` 当**长连接**用。
    留着超时的后果是：只要 daemon 处理一条消息的时间超过 ``timeout``（工具
    调用动辄几秒到几十秒，比如现场加载嵌入模型），``sock.recv()`` 就会抛
    ``socket.timeout``，被 ``_pump`` 的 ``except Exception: pass`` **静默吞掉**，
    整条 daemon→stdout 的回程泵当场死掉 —— 宿主看到的就是"请求发出去了、
    响应永远不回来"，**没有任何报错**。

    这是实测撞出来的第二个 bug（两个 launcher 子进程连同一 daemon）：
    握手与 ``tools/list`` 都在超时内返回所以正常，第一条 ``tools/call``
    因为要先加载模型而超出超时，回程就此断掉。
    """
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    except OSError:
        return None
    try:
        s.settimeout(None)          # 交给 relay 后是长连接，不再要超时
    except OSError:
        pass
    return s


def _daemon_creationflags() -> tuple[int, int]:
    """Windows 子进程创建标志：``(base_flags, breakaway_flag)``。

    ⚠️ **修订：最初写下的前提是错的，代码保留但不再当"治根"手段。**

    最初认为"宿主 Job 带 ``KILL_ON_JOB_CLOSE``，回收时连坐杀掉 daemon"，依据是把
    读到的 ``LimitFlags=0x2400`` 解读成 ``KILL_ON_JOB_CLOSE=True``。**位掩码用错了**：
    winnt.h 的真实值是 ``BREAKAWAY_OK=0x200`` / ``SILENT_BREAKAWAY_OK=0x400`` /
    ``KILL_ON_JOB_CLOSE=0x800`` —— 按正确值 ``0x2400 = JOB_MEMORY | SILENT_BREAKAWAY_OK``，
    含义近乎相反（行为实验验证过）。

    仍然保留这个 flag，有两个理由：
    1. 它是一枚**探针**：能否 breakaway 直接反映宿主 Job 策略，失败告警本身有价值；
    2. 若将来宿主放宽 Job 策略（例如在内层 Job 上放开 ``BREAKAWAY_OK``），这里自动生效。

    但**不要指望它治根**。真因：
    daemon 是被**外部终止**的（日志里 0 条孤儿回收、0 条崩溃痕迹），触发点是宿主
    空闲约 5 分钟后的 dispose；launcher 自身从不杀 daemon（无 atexit / terminate）。

    抽成独立函数是为了让测试能直接断言那一位 flag 还在
    （``tests/test_f9_6_process_tree.py`` 的 T1），防止后人把它删掉。
    """
    base = 0
    breakaway = 0
    if os.name == "nt":
        base = getattr(subprocess, "DETACHED_PROCESS", 0) | \
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        breakaway = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    return base, breakaway


def spawn_daemon(key: str, library: str = "", model: str = "",
                 extra_env: dict[str, str] | None = None,
                 cwd: str = "") -> dict[str, Any] | None:
    """拉起 daemon 并等它把登记写好。失败返回 None（调用方负责退化）。

    ``cwd``：daemon 的工作目录，空则取调用方的 cwd。
    🔴 **预热场景必须显式指定**：``library`` 为空时 daemon 是按**自己的 cwd**
    定位库的，cwd 错了会起来一个指向别处的 daemon —— 它照样能起、照样写登记、
    照样被 launcher 复用，只是**库是错的**，且不会报任何错。
    """
    env = dict(os.environ)
    run_cwd = cwd or os.getcwd()
    env["HIPPOCAMPUS_DAEMON"] = "1"
    # 🔴 输出必须**无缓冲**：daemon 是被宿主强杀、被总线杀、或 native 崩溃时死的，
    # 块缓冲里的东西一句都留不下 —— 表现在事故现场就是「daemon 无声消失、
    # 一行 traceback 都没有」。这条比"把 stderr 落盘"更靠前：落了盘但没 flush，
    # 等于写了封信塞进信封直接烧掉。
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["HIPPOCAMPUS_DAEMON_KEY"] = key
    env[ENV_RUNTIME_DIR] = runtime_dir()
    if library:
        env["HIPPOCAMPUS_DAEMON_LIBRARY"] = library
    if model:
        env["HIPPOCAMPUS_DAEMON_MODEL"] = model
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, "-m", "hippocampus_mcp.daemon"]
    base_flags, breakaway_flag = _daemon_creationflags()
    creationflags = base_flags | breakaway_flag
    # daemon 的 stdout/stderr 必须落盘，不能 DEVNULL：它是唯一常驻重进程，
    # 崩溃 traceback 一旦进黑洞就永远无法归因（真宿主首验时踩过）。
    log_path = os.path.join(runtime_dir(), f"daemon-{key}.log")
    try:
        if os.path.exists(log_path) and os.path.getsize(log_path) > 2 * 1024 * 1024:
            os.replace(log_path, log_path + ".1")
    except OSError:
        pass
    log_fh, log_used = _open_daemon_log(log_path)
    if log_fh is not None:
        try:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_fh.write(
                f"\n=== spawn {stamp} key={key} launcher_pid={os.getpid()} "
                f"cwd={run_cwd} log={os.path.basename(log_used)} ===\n".encode())
            log_fh.flush()
        except OSError:
            pass
    proc = None
    try:
        try:
            proc = subprocess.Popen(
                cmd, env=env, cwd=run_cwd,
                stdin=subprocess.DEVNULL,
                stdout=log_fh if log_fh is not None else subprocess.DEVNULL,
                stderr=log_fh if log_fh is not None else subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except OSError as exc:
            proc = _spawn_without_breakaway(
                cmd, env, log_fh, base_flags, breakaway_flag, exc, run_cwd)
    except Exception:  # noqa: BLE001 —— 起不来必须退化，不能让宿主连不上
        proc = None
    finally:
        # 子进程已持有自己的句柄副本，父进程这份用完即关
        if log_fh is not None:
            log_fh.close()
    if proc is None:
        return None
    deadline = time.time() + DAEMON_READY_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            return None                       # 起不来：别干等
        info = read_registry(key)
        if info and info.get("pid") == proc.pid:
            s = try_connect(int(info["port"]))
            if s is not None:
                s.close()
                return info
        time.sleep(0.2)
    return None


# ── spawn 单飞（singleflight）────────────────────────────────────────
# 由来：两个 launcher 相隔 5 秒各自发现「没有 daemon」，
# 于是各自 spawn ⇒ **双 daemon 竞写同一注册表**，前者没来得及
# 初始化 brain 就无声死亡。那一轮靠 idle 回收自愈成单 daemon，但代价是
# 两个 ~1GB 进程同框、注册表被后写者覆盖（谁服务谁是随机的）。
SPAWN_LOCK_TTL = 120.0      # 锁的最大持有时长（秒）；超时当作持有者已死
#: 「刚创建、内容残缺」的宽限期（秒）。见 :func:`_spawn_lock_stale` —— 这段时间内
#: 不把残缺的锁文件当成陈旧锁，否则会把正在创建锁的邻居一把拍死。
SPAWN_LOCK_CREATE_GRACE = 5.0
SPAWN_WAIT_TIMEOUT = 60.0   # 抢不到锁时，跟随别人 spawn 的最长等待（秒）


def _spawn_without_breakaway(cmd, env, log_fh, base_flags: int,
                             breakaway_flag: int, exc: OSError,
                             cwd: str = ""):
    """带 BREAKAWAY 的 spawn 失败时，退回旧 flags 重试。

    🔴 **退回必须吼出来**。静默退回等于把"daemon 仍在宿主进程树里"这个事实
    藏起来 —— 下一次宿主回收时会再死一次，而现场依旧是"15 个 daemon 一行
    traceback 都没留下"。可用性优先没错，但**诊断信息不许跟着一起降级**。
    """
    if not breakaway_flag:
        return None
    msg = (f"[launcher] ⚠ daemon spawn with CREATE_BREAKAWAY_FROM_JOB failed"
           f" ({type(exc).__name__}: {exc}) -- daemon remains in the host process tree; "
           f"it will be terminated with the host session. Retrying without this flag works, "
           f"but does not protect against host cleanup."
           f"To fully detach, use :mod:`hippocampus_mcp.prewarm` to change the creator to "
           f"a login-startup or scheduled task.")
    print(msg, file=sys.stderr, flush=True)
    try:
        return subprocess.Popen(
            cmd, env=env, cwd=cwd or os.getcwd(),
            stdin=subprocess.DEVNULL,
            stdout=log_fh if log_fh is not None else subprocess.DEVNULL,
            stderr=log_fh if log_fh is not None else subprocess.DEVNULL,
            creationflags=base_flags,
        )
    except OSError:
        return None


def spawn_lock_path(key: str) -> str:
    return os.path.join(runtime_dir(), f"daemon-{key}.spawnlock")


def _read_spawn_lock(key: str) -> dict[str, Any] | None:
    """读锁内容 ``{pid, ts}``。读不出 / 格式不对 ⇒ 当没人持锁。"""
    try:
        with open(spawn_lock_path(key), "r", encoding="utf-8") as f:
            info = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(info, dict) and isinstance(info.get("ts"), (int, float)):
        return info
    return None


def _spawn_lock_stale(key: str, info: dict[str, Any] | None) -> bool:
    """锁是否可以抢占。

    刻意**不用**阻塞式互斥：持有者是另一个独立进程，它可能连同宿主会话
    一起被杀（34280 就是一句话没留下死的）。任何阻塞锁都会在那之后把
    后面所有 launcher 一起堵死 ⇒ 一律「有时限 + 可接管」。

    🔴 **读不出内容 ≠ 没人持锁**。旧实现的洞就在这：别人刚用
    ``O_CREAT|O_EXCL`` 建出**空文件**、还没来得及写 JSON 时，这边
    ``json.load`` 失败 ⇒ 得到 None ⇒ 判"可抢占" ⇒ 把那把还没写完的锁 rm 掉，
    然后自己也去抢 —— 两个 launcher 同时认为自己持锁，两个 ~1GB daemon 同框
    （双 daemon 事故就是这个形态）。

    守住它：文件存在但内容残缺时，按 mtime 给它一个"写完的宽限期"。
    """
    if info:
        ts = float(info.get("ts", 0))
        if ts and time.time() - ts > SPAWN_LOCK_TTL:
            return True
        pid = info.get("pid")
        if (isinstance(pid, int) and pid > 0 and pid != os.getpid()
                and not _pid_alive(pid)):
            return True
        return False
    try:
        age = time.time() - os.stat(spawn_lock_path(key)).st_mtime
    except OSError:
        return True                       # 文件不存在 ⇒ 确实没人持锁
    return age > SPAWN_LOCK_CREATE_GRACE


# 进程内抢锁串行锁。
#
# 🔴 为什么还需要它（os.link 已经原子了）：原子的只是**发布**，不是
# 「判断陈旧 → 删除陈旧锁」这一步。同进程 8 个线程对齐起跑时，全都看到
# 「锁不存在」⇒ 都判可抢占；先到者 link 成功，慢一步的那个随后执行
# os.remove(lock)，**把刚发布好的新锁删掉**，再自己 link 成功 ⇒ 两个赢家。
# 这正是 test_f9_6 T6 在 Linux 上偶发 wins=[True,...,True] 的成因，也是
# 「两个 ~1GB daemon 同框」的同一种形态。
#
# 作用域说明：这把锁只在**本进程内**生效；跨进程互斥仍由 os.link 的原子性
# 保证（别人发布的是完整 JSON，后来的 stale 判定会读到新鲜内容而让行）。
# 跨进程那条更窄的「检查—删除」窗口仍在，若要彻底封死需改成原子认领协议。
_SPAWN_LOCK_MUTEX = threading.Lock()


def acquire_spawn_lock(key: str) -> bool:
    """抢占 spawn 锁。抢到返回 True。

    🔴 **锁文件必须"一出现就是完整的"**。旧实现用 ``O_CREAT|O_EXCL`` 先建出
    **空文件**再往里写 JSON，中间那个窗口正是 :func:`_spawn_lock_stale` 注释里
    说的双 daemon 成因（早年代码注释自己也承认了这个窗口，现在补掉）。

    修法：**先在临时文件里写完整 + fsync，再用 ``os.link`` 原子地挂到锁路径**。
    硬链接创建是原子的，且目标已存在时必然失败 —— 互斥语义一点没丢，
    半吊子窗口彻底消失。

    再加一层 :data:`_SPAWN_LOCK_MUTEX`：把「判断陈旧 → 清锁 → 写临时文件 →
    原子发布」整段串成进程内临界区，封住上面说的 TOCTOU 竞态。
    """
    try:
        os.makedirs(runtime_dir(), exist_ok=True)
    except OSError:
        return False
    lock = spawn_lock_path(key)
    with _SPAWN_LOCK_MUTEX:
        if not _spawn_lock_stale(key, _read_spawn_lock(key)):
            return False
        try:
            os.remove(lock)               # 确认陈旧之后才清
        except OSError:
            pass
        tmp = f"{lock}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "ts": time.time()}, f)
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass
            return False
        try:
            os.link(tmp, lock)            # 原子发布：目标已存在 ⇒ 失败 ⇒ 让行
            return True
        except FileExistsError:
            return False                  # 正常竞争输给别人
        except OSError:
            return _acquire_lock_legacy(lock)  # 文件系统不支持硬链接 ⇒ 退化
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass


def _acquire_lock_legacy(lock: str) -> bool:
    """``os.link`` 不可用时的退化路径（例如不支持硬链接的文件系统）。

    仍是旧的 ``O_CREAT|O_EXCL`` 语义，**半吊子窗口还在**，由
    :func:`_spawn_lock_stale` 的宽限期兜着 —— 对方有
    :data:`SPAWN_LOCK_CREATE_GRACE` 秒把内容写完。
    """
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError:
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "ts": time.time()}, f)
            f.flush()
            os.fsync(f.fileno())
        return True
    except OSError:
        try:
            os.close(fd)
        except OSError:
            pass
        return False


def _open_daemon_log(log_path: str):
    """打开 daemon 日志，返回 ``(句柄, 实际路径)``。

    🔴 **失败时绝不能悄悄退回 DEVNULL** —— 那等于把这一份 daemon 的所有输出
    （含崩溃 traceback）扔进黑洞，事后表现为"daemon 无声死亡、毫无线索"。
    此前两次"无声死亡"就有这个嫌疑。

    **Windows 实测**：两个进程并发 ``open(path, "ab")`` 同一个文件时，其中一个抛
    ``PermissionError [Errno 13]``（探针 ``_probe_out/_append_concurrency_probe_20260929.py``）。
    更要命的是 daemon 会**长期持有**继承来的句柄 ⇒ **后到**的 launcher 有很大
    概率撞上它，这正是当年双 daemon 竞写的同一时间窗。

    处置顺序：重试若干次（瞬时冲突）→ 仍失败就退到**带 pid 的备用文件**
    （这一份输出照样留得下，只是不在主文件里）→ 连备用都打不开才交给 DEVNULL，
    且两处降级都在启动器自己的 stderr 留一行，巡检时看得见。
    """
    last_err: BaseException | None = None
    for _ in range(3):
        try:
            return open(log_path, "ab"), log_path
        except OSError as exc:
            last_err = exc
            time.sleep(0.05)
    alt = log_path[:-4] + f"-{os.getpid()}.log" if log_path.endswith(".log") \
        else f"{log_path}.{os.getpid()}"
    try:
        fh = open(alt, "ab")
        print(f"[launcher] daemon main log is busy ({type(last_err).__name__}); "
              f"falling back to {alt}", file=sys.stderr)
        return fh, alt
    except OSError:
        print(f"[launcher] cannot open daemon log ({type(last_err).__name__}); "
              f"crash output will be lost", file=sys.stderr)
        return None, ""


def release_spawn_lock(key: str) -> None:
    """释放锁：**只删自己这把**。

    无差别 ``os.remove`` 会把别人刚刚抢到的新锁一起删掉 —— 那是双 daemon 的
    另一条路，而 S4 那条测试只看"用完无残留"，恰恰看不出这个区别。
    """
    path = spawn_lock_path(key)
    info = _read_spawn_lock(key)
    if isinstance(info, dict) and info.get("pid") == os.getpid():
        try:
            os.remove(path)
        except OSError:
            pass
        return
    print(f"[launcher] ⚠ spawn lock is no longer owned by this process (content={info}); not removing {path}",
          file=sys.stderr, flush=True)


def _registry_usable(info: dict[str, Any] | None) -> bool:
    """注册表条目能否连上（pid 活着 + 端口可握手）。"""
    if not info:
        return False
    pid = info.get("pid")
    if not isinstance(pid, int) or not _pid_alive(pid):
        return False
    s = try_connect(int(info["port"]), timeout=2.0)
    if s is None:
        return False
    s.close()
    return True


def ensure_daemon(key: str, library: str = "", model: str = "") -> dict[str, Any] | None:
    """找到或拉起 daemon —— **同一时刻只有一个 launcher 会真去 spawn**。

    三条路径：
      ① 注册表可用 → 直接返回（最常见，零等待）
      ② 拿到 spawn 锁 → double-check 后真 spawn（拿锁期间别人可能刚起好）
      ③ 抢不到锁 → 跟牌等待（持锁者死了或窗超时，回到 ② 自己尝试）
    """
    deadline = time.time() + DAEMON_READY_TIMEOUT
    while time.time() < deadline:
        info = read_registry(key)
        if _registry_usable(info):
            return info
        if acquire_spawn_lock(key):
            try:
                info = read_registry(key)          # double-check
                if _registry_usable(info):
                    return info
                return spawn_daemon(key, library=library, model=model)
            finally:
                release_spawn_lock(key)
        # ③ 别人在 spawn：跟随，盯着注册表也盯着锁（防止跟着死等）
        wait_until = min(time.time() + SPAWN_WAIT_TIMEOUT, deadline)
        while time.time() < wait_until:
            time.sleep(0.2)
            info = read_registry(key)
            if _registry_usable(info):
                return info
            if not read_registry(key) and _spawn_lock_stale(
                    key, _read_spawn_lock(key)):
                break                              # 持锁者没了且 daemon 没起 ⇒ 自己上
    return None


#: 转发字节流调试开关（``HIPPOCAMPUS_LAUNCHER_DEBUG=1``）。
#: 开着会在 stderr 上打每一段转发的方向与长度 —— 排查"请求发出去了但响应
#: 不回来"这类无报错的挂起时，这是唯一能看清双向流的东西。
_RELAY_DEBUG = os.environ.get("HIPPOCAMPUS_LAUNCHER_DEBUG", "") == "1"


def _name(obj: Any) -> str:
    if hasattr(obj, "recv"):
        return "sock"
    if obj is getattr(sys, "stdin", None):
        return "stdin"
    return "stdout"


def _pump(src, dst, chunk: int = RELAY_CHUNK) -> None:
    """单向字节泵。任一端关了就停，不抛。

    🔴 **必须用 ``read1`` 而不是 ``read``**：``BufferedReader.read(n)`` 的语义是
    "读满 n 字节或到 EOF 才返回"，对管道/stdio 意味着**宿主写一行 JSON-RPC
    之后会被永久阻塞**——MCP over stdio 的帧通常只有几百字节，远不到 64KB，
    于是宿主的 initialize 永远等不到响应，表现为"工具数为 0 / 服务不可用"，
    且**没有任何报错**。``read1(n)`` 只做一次底层读、有多少返回多少，才是
    字节泵该用的接口。

    这个坑是实测撞出来的（两个 launcher 子进程直连同一库）：daemon
    已经起好、注册表也写了，但会话 1 的握手 180 秒都等不到一个字节。
    """
    try:
        while True:
            if hasattr(src, "read1"):
                data = src.read1(chunk)      # 有多少给多少，绝不读满才返回
            elif hasattr(src, "read"):
                data = src.read(chunk)
            else:
                data = src.recv(chunk)
            if not data:
                break
            if _RELAY_DEBUG:
                print(f"[relay] {_name(src)}->{_name(dst)} {len(data)}B",
                      file=sys.stderr, flush=True)
            dst.write(data) if hasattr(dst, "write") else dst.sendall(data)
            if hasattr(dst, "flush"):
                dst.flush()
    except Exception:  # noqa: BLE001 —— 管道断开是正常的收尾方式
        pass
    try:
        if hasattr(dst, "close"):
            dst.close()
    except Exception:  # noqa: BLE001
        pass


def _send_hello(sock: socket.socket) -> bool:
    """转发任何 MCP 流量之前，先告诉 daemon「这条连接属于哪个库」。

    🔴 没有这一步就会串库：daemon 的默认库是**进程级**
    常量（server 模块导入时按当时那个 launcher 的 env/cwd 算出）。两个工作区
    复用同一个 daemon（key 相同 —— 这正是省内存的目的）时，后到的工作区会
    写进先到那个的库，且不报错。

    必须**先于**所有客户端字节发出，daemon 侧才能用一次 peek 干净地吃掉它。
    """
    payload = {
        "library": os.environ.get("HIPPOCAMPUS_PERSIST_DIR", ""),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
    }
    line = HELLO_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n"
    try:
        sock.sendall(line.encode("utf-8"))
        return True
    except OSError:
        # 发不出去不能让会话起不来：daemon 会回落到进程级默认库（旧行为）
        return False


# ══════════════════════════════════════════════════════════════════════
# stdio ↔ daemon 的中继：**行分帧 + 断线自愈**
# ══════════════════════════════════════════════════════════════════════
# 这一段是真宿主事故逼出来的重写。旧实现是两个
# **裸字节泵**：``_pump(stdin, sock)``（主线程）+ ``_pump(sock, stdout)``
# （另一线程）。daemon 被宿主回收时，这两个方向的行为**不对称**，事故
# 于是分两步发生：
#
#   ① 回程泵收到 EOF ⇒ 退出 ⇒ 走到 ``_pump`` 的 finally ⇒ **close(stdout)**。
#      宿主读到 EOF，判定 MCP server 已下线，打点 ``mcp.server.closed``。
#   ② 去程泵正阻塞在 ``stdin.read1()`` 上，对面死了它一无所知 ⇒ launcher
#      **进程存活**、攥着一条死管道 ⇒ 僵尸态。宿主既不重连也不重启，
#      因为它已经给这条连接判了死刑。
#
# 此后会话里每一次 hippo 调用都是 22~33 ms 的 ``Connection closed``。
# 注意：**没有一次走到了网络层** —— 连接是我们自己关的。
#
# 换成行分帧之后，三件事才第一次成立：
#   · **原子**：一整条 JSON-RPC 要么完整到达 daemon，要么没发；不会有半帧。
#   · **可计数**：按 JSON-RPC 语义记录 in-flight 的 request id。
#   · **可重连**：断在帧边界上，换一条新 socket 就能接着跑，
#     **宿主全程不需要知道发生过断连**。
#
# daemon 侧天然配合得了：``bridge.py`` 的 ``sock_reader`` 本来就按 ``\n``
# 切帧，且对非法帧是"跳过并吼一句"而不是杀连接（见 bridge.py 111-119 行）。
#
# 自检清单（改这里之前先看 ``tests/test_f9_5_relay_reconnect.py``）：
#   1. **回程线程绝不许 close(stdout)** —— 那等于替宿主判死刑；
#   2. 自愈不了就**明确失败**（关 stdout + 非 0 退出），不许停在僵尸态；
#   3. 重连后必须在**新 daemon 上重建 MCP 会话**（重放 initialize），
#      否则后续 tools/call 会因"未 initialize"被拒；
#   4. 未决请求必须拿到一个**明确交代**（合成 error），不许挂死。

#: 自愈开关。``HIPPOCAMPUS_LAUNCHER_RECONNECT=0`` 退回旧的单程字节泵行为，
#: 只用于 A/B 对照与测试里的**负对照**。别在生产关掉它。
ENV_RECONNECT = "HIPPOCAMPUS_LAUNCHER_RECONNECT"
#: 诊断输出开关（写 stderr，不碰 stdout）。
ENV_DIAG = "HIPPOCAMPUS_LAUNCHER_DIAG"
#: 一次断线的重连尝试次数上限。
RECONNECT_MAX_ATTEMPTS = 3
#: 一次断线的修复总预算（秒）。daemon 冷起要加载嵌入模型（实测 ~13-35s），
#: 给足，但要小于宿主常见的工具调用超时。
RECONNECT_DEADLINE = 90.0
#: 重连失败后的退避基数（秒），按 attempt 线性放大。
RECONNECT_BACKOFF = 0.5
#: 重连后重放 initialize 时，等待其响应的上限（秒）。
HANDSHAKE_REPLAY_TIMEOUT = 60.0

#: 未决请求的交代文案。**必须说清"结果未知"**——不许暗示成功或失败，
#: 调用方（尤其是写类操作的调用方）全靠这句话决定要不要重试。
ERR_DAEMON_LOST = (
    "HippoMCP daemon connection was interrupted and could not recover immediately: the result of this request is UNKNOWN"
    " (it may have executed, or may never have reached the daemon). Idempotent operations may be safely retried; "
    "for non-idempotent writes (ingest / update_memory / forget), read back before retrying."
)

#: main() 在开跑前填好 —— _Relay 靠它知道"该重连到哪个 key"。
_SESSION_PARAMS: dict[str, str] = {"key": "", "library": "", "model": ""}


def _diag(msg: str) -> None:
    """诊断只能走 stderr。

    stdout 是 **JSON-RPC 专用通道**，写脏一个字节整条 MCP stdio 会话就废了
    （``server.py`` 的 ``_ThreadAwareStdout`` 护栏是同一个原因）。
    """
    if os.environ.get(ENV_DIAG, "1") == "0":
        return
    try:
        print(f"[launcher] {msg}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 —— 诊断永远不许影响主链路
        pass


def reconnect_enabled() -> bool:
    return os.environ.get(ENV_RECONNECT, "1") != "0"


def _id_key(rid: Any) -> str:
    """把 JSON-RPC 的 id（str / int / None）压成稳定的 dict key。"""
    return json.dumps(rid, sort_keys=True, default=str)


def _request_key(line: bytes) -> str | None:
    """这一帧是不是一条**需要应答的 request**？是就返回它的 key。

    notification（有 method 无 id）不需要应答，因此也不进未决账
    —— 把它记进来会让 ``_fail_inflight`` 给它合成一个没人认领的 error。
    """
    try:
        obj = json.loads(line.decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(obj, dict):
        return None
    if "method" not in obj or "id" not in obj:
        return None
    return _id_key(obj["id"])


class _RelayLost(Exception):
    """daemon 连接不可恢复 —— 必须终止这条 stdio，绝不停在僵尸态。"""


class _Relay:
    """stdio ↔ daemon socket 的行分帧中继。

    线程模型（与旧的双泵不同）：
      · 主线程只做 stdin → daemon（**逐行**）
      · 每条 daemon 连接派生一个回程线程做 daemon → stdout（**逐行**）
      · 回程线程**不拥有** stdout 的生命期；stdout 只在宿主关掉 stdin
        或明确放弃自愈时才关。
    """

    def __init__(self, key: str, library: str = "", model: str = "") -> None:
        self.key = key
        self.library = library
        self.model = model
        self._sock: socket.socket | None = None
        self._gen = 0
        self._sock_lock = threading.Lock()
        self._stdout_lock = threading.Lock()
        self._reseat_lock = threading.Lock()
        self._inflight: dict[str, str] = {}     # id_key -> 原始请求帧
        self._inflight_lock = threading.Lock()
        self._handshake: list[bytes] = []       # 缓存的 initialize/initialized
        self._initialized = False
        self._sock_dead = threading.Event()
        self._stopped = threading.Event()
        self._stdout = None
        self._stdin = None

    # ── 主循环 ──────────────────────────────────────────────────────
    def run(self, sock: socket.socket) -> int:
        """返回 0＝宿主正常关闭；4＝自愈失败，已明确断开。"""
        try:
            self._stdout = sys.stdout.buffer
            self._stdin = sys.stdin.buffer
        except Exception:  # noqa: BLE001
            return 2
        # 🔴 必须在转发任何 MCP 字节**之前**声明这条连接属于哪个库，否则串库
        # （见 :func:`_send_hello` 的注释）。重连路径由 ``_connect_fresh``
        # 各自负责，这里只管首发这条。
        _send_hello(sock)
        if not self._install(sock):
            return 3
        try:
            self._pump_stdin()
        except _RelayLost:
            self._abandon()
            return 4
        finally:
            self._stopped.set()
            self._drop_sock()
        return 0

    def _pump_stdin(self) -> None:
        """主线程：宿主 stdio → daemon，一次一行。

        🔴 用 ``readline`` 而不是 ``read``/``read1`` 是有意的：这里要的正是
        「读到行尾就返回」的语义。``BufferedReader.read(n)`` 会等到读满 n 字节
        或 EOF（``read1`` 是为了治这个才引入的，见 :func:`_pump` 的注释），
        而 ``readline`` 天然按行返回，不存在这个问题。别把它"优化"回
        ``read1`` + 自己攒缓冲 —— 那是在重造 BufferedReader。
        """
        while True:
            if self._stopped.is_set():
                return
            line = self._stdin.readline()
            if not line:
                return                              # 宿主关了 stdin：正常收尾
            line = line.strip()
            if not line:
                continue
            self._deliver(line)

    # ── 宿主 → daemon ───────────────────────────────────────────────
    def _deliver(self, line: bytes) -> None:
        """投递一整帧。发不出去就自愈（重连 + 补发这一帧）。"""
        self._remember_handshake(line)
        # 登记必须**先于**发送：响应可能在 sendall 返回的那一刻就回来了，
        # 先发后记会把这条勾销漏过去。
        self._register_inflight(line)

        # 🔴 先查 ``_sock_dead`` 再发：回程线程通常比去程更早知道对面死了。
        # 少了这一步，sendall 会**成功写进内核缓冲**（对端已关闭时第一次写
        # 往往不报错），于是我们以为发出去了、宿主从此挂等到超时 ——
        # 这比立刻自愈糟得多。
        if not self._sock_dead.is_set() and self._send(line):
            return

        # 🔴 **这里故意不撤销登记**。这一帧接下来只有两种归宿：被 ``_reseat``
        # 补发成功，或补发失败时由 ``_replay_pending`` 撤销。中途撤销会让它
        # 从账上人间蒸发 —— ``_fail_inflight`` 看不见它，宿主永远等不到这条
        # 应答，只能挂到超时（自测撞到的记账 bug，第一版正是这么
        # 写的：日志里能看到「记入未决 id=13 在途=2」，最后却只交代了 1 条）。
        if self._reseat(pending=line):
            return
        n = self._fail_inflight(ERR_DAEMON_LOST)
        _diag(f"Recovery failed; returned error for {n} pending requests and terminated this stdio session")
        raise _RelayLost("daemon connection is unrecoverable")

    def _send(self, line: bytes) -> bool:
        with self._sock_lock:
            sock = self._sock
        if sock is None:
            return False
        try:
            sock.sendall(line + b"\n")
            return True
        except OSError:
            self._sock_dead.set()
            return False

    # ── daemon → 宿主 ───────────────────────────────────────────────
    def _pump_sock(self, sock: socket.socket, gen: int) -> None:
        """回程：按 ``\\n`` 切帧写一整行到 stdout。

        🔴 **这个函数绝对不许 close(stdout)**。旧实现的回程泵在 finally 里
        关了 stdout，宿主因此读到 EOF 并判定 server 下线 —— 事故
        的第一张多米诺骨牌就是它。只要宿主还没关 stdin，这条 MCP 连接就
        还有效，我们就得留着这条 stdout。
        """
        buf = b""
        try:
            while True:
                try:
                    data = sock.recv(RELAY_CHUNK)
                except OSError:
                    break
                if not data:
                    break
                with self._sock_lock:
                    stale = self._gen != gen or self._sock is not sock
                if stale:
                    return                       # 已被换代的旧连接：安静退出
                buf += data
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    self._settle_inflight(raw.strip())
                    self._write_stdout(raw + b"\n")
        finally:
            if buf.strip():
                # 断在半帧上：这一帧永远不完整，只能丢。要吼一句 —— 静默丢帧
                # 在以后的事故里会被当成"daemon 没回"。
                _diag(f"Discarded partial frame {len(buf)}B (connection broke at a non-frame boundary)")
            with self._sock_lock:
                current = self._gen == gen and self._sock is sock
            if current and not self._stopped.is_set():
                self._sock_dead.set()
                if reconnect_enabled():
                    # 主动预热：与其等下一条请求来时才拉 daemon（冷起 ~13-35s，
                    # 很可能顶穿宿主的调用超时），不如现在就在后台补。
                    threading.Thread(
                        target=self._reseat, kwargs={"pending": b""},
                        daemon=True).start()

    # ── 重连 ────────────────────────────────────────────────────────
    def _reseat(self, pending: bytes = b"") -> bool:
        """换一条 daemon 连接；成功时负责把 ``pending`` 补发出去。

        全程持 ``_reseat_lock``：回程线程的后台预热与主线程的请求投递可能
        同时到达这里，不串行就会出现两个 launcher 各拉一个 daemon（正是
        那个 ~1GB ×2 的双 daemon 事故）。
        """
        if not reconnect_enabled():
            return False
        if not self._reseat_lock.acquire(timeout=RECONNECT_DEADLINE):
            return False
        try:
            deadline = time.time() + RECONNECT_DEADLINE
            attempt = 0
            while time.time() < deadline:
                with self._sock_lock:
                    alive = not self._sock_dead.is_set()
                if alive:
                    return self._replay_pending(pending)   # 别人已经修好了
                attempt += 1
                if attempt > RECONNECT_MAX_ATTEMPTS:
                    break
                _diag(f"daemon connection dropped; reconnect attempt {attempt}")
                s = self._connect_fresh()
                if s is None:
                    time.sleep(RECONNECT_BACKOFF * attempt)
                    continue
                if not self._install(s, replay_handshake=True):
                    continue
                _diag("Reconnected; MCP session rebuilt on new daemon")
                return self._replay_pending(pending)
            return False
        finally:
            self._reseat_lock.release()

    def _replay_pending(self, pending: bytes) -> bool:
        """补发重连前没能发出去的那一帧。

        🔴 这一帧此前已经登记过，这里**不再重复登记**，只在确认发出去不通时
        才撤销 —— 记账的唯一入口是 ``_deliver``，否则同一个 id 会被两处
        互相覆盖/撤销，最后谁也说不清它到底发没发出去。
        """
        if not pending:
            return True
        if self._send(pending):
            return True
        self._rollback_inflight(pending)
        return False

    def _connect_fresh(self) -> socket.socket | None:
        info = ensure_daemon(self.key, library=self.library, model=self.model)
        if info is None:
            return None
        try:
            s = try_connect(int(info["port"]), timeout=5.0)
        except Exception:  # noqa: BLE001
            return None
        if s is None:
            return None
        if not _send_hello(s):
            try:
                s.close()
            except OSError:
                pass
            return None
        return s

    def _install(self, sock: socket.socket, replay_handshake: bool = False) -> bool:
        """接管一条新连接（并起它的回程线程）。"""
        if replay_handshake and not self._replay_handshake(sock):
            _diag("Failed to rebuild MCP session after reconnect; trying another connection")
            try:
                sock.close()
            except OSError:
                pass
            return False
        with self._sock_lock:
            old = self._sock
            self._sock = sock
            self._gen += 1
            gen = self._gen
        if old is not None:
            try:
                old.close()
            except OSError:
                pass
        self._sock_dead.clear()
        threading.Thread(
            target=self._pump_sock, args=(sock, gen), daemon=True).start()
        return True

    def _replay_handshake(self, sock: socket.socket) -> bool:
        """在新 daemon 上把宿主的握手重做一遍。

        🔴 **这一步不能省**：新 daemon 上是一条全新的 MCP 会话，而不重放的
        话，宿主后续发来的 ``tools/list`` / ``tools/call`` 会因为"未完成
        initialize"被拒。而宿主自己是认这段历史的（它早就 initialize 过了），
        绝不会重来一次 ⇒ 这段只能由 launcher 替它补。

        initialize 的响应必须**自己吃掉**，不能再转发给宿主 —— 否则同一个
        id 会出现第二个响应。
        """
        if not self._initialized or not self._handshake:
            return True
        for raw in self._handshake:
            if _RELAY_DEBUG:
                print(f"[relay] replaying handshake frame {raw[:70]!r}",
                      file=sys.stderr, flush=True)
            try:
                sock.sendall(raw + b"\n")
            except OSError:
                return False
            try:
                obj = json.loads(raw.decode("utf-8", "replace"))
            except Exception:  # noqa: BLE001
                continue
            if obj.get("method") == "initialize" and "id" in obj:
                if not self._drink(sock):
                    if _RELAY_DEBUG:
                        print("[relay] no response to replayed handshake",
                              file=sys.stderr, flush=True)
                    return False
        return True

    @staticmethod
    def _drink(sock: socket.socket) -> bool:
        """读走一整行并丢弃。"""
        try:
            sock.settimeout(HANDSHAKE_REPLAY_TIMEOUT)
            buf = b""
            while b"\n" not in buf:
                data = sock.recv(RELAY_CHUNK)
                if not data:
                    return False
                buf += data
            return True
        except OSError:
            return False
        finally:
            try:
                sock.settimeout(None)
            except OSError:
                pass

    def _drop_sock(self) -> None:
        with self._sock_lock:
            sock = self._sock
            self._sock = None
            self._gen += 1
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    # ── 握手历史 / in-flight 记账 ───────────────────────────────────
    def _remember_handshake(self, line: bytes) -> None:
        try:
            obj = json.loads(line.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return
        if not isinstance(obj, dict):
            return
        method = obj.get("method")
        if method == "initialize":
            # 只记最新一条 initialize：宿主重启会话时握手会重来。
            self._handshake = [line.strip()]
            self._initialized = False
        elif method == "notifications/initialized":
            if self._handshake:
                self._handshake.append(line.strip())
            self._initialized = True

    def _register_inflight(self, line: bytes) -> str | None:
        """登记未决请求。只有 **request**（有 method 且有 id）才需要应答。"""
        k = _request_key(line)
        if k is None:
            return None
        with self._inflight_lock:
            self._inflight[k] = line.decode("utf-8", "replace")
            if _RELAY_DEBUG:
                print(f"[relay] -> registered pending id={k} in-flight={len(self._inflight)}",
                      file=sys.stderr, flush=True)
        return k

    def _rollback_inflight(self, line: bytes) -> None:
        """**确认这一帧最终没能发出去**才撤销登记。"""
        k = _request_key(line)
        if k is None:
            return
        with self._inflight_lock:
            existed = self._inflight.pop(k, None)
        if _RELAY_DEBUG and existed is not None:
            print(f"[relay] xx canceled pending id={k}", file=sys.stderr, flush=True)

    def _settle_inflight(self, line: bytes) -> None:
        """应答到达，勾销未决。"""
        try:
            obj = json.loads(line.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return
        if not isinstance(obj, dict) or "id" not in obj:
            return
        if "method" in obj:
            # daemon 主动发来的 request（sampling / elicitation 之类），
            # 不是对宿主请求的应答 —— 不能拿来勾销。
            return
        with self._inflight_lock:
            removed = self._inflight.pop(_id_key(obj["id"]), None)
        if _RELAY_DEBUG and removed is not None:
            print(f"[relay] <- completed id={_id_key(obj['id'])}",
                  file=sys.stderr, flush=True)

    def _fail_inflight(self, reason: str) -> int:
        """给所有未决请求一个**明确交代**。

        为什么是 error 而不是留在那儿：留在那儿等于让宿主从今往后每一条
        请求都挂到超时，用户看到一个没有上下文的"工具调用失败"。
        为什么不自动重放：我们不知道这一帧是否已被 daemon 执行过 ——
        重放 ``ingest`` 就会静默写入两份。把"结果未知"如实告诉调用方，
        让**知道语义的上层**决定重不重试，是本分。
        """
        with self._inflight_lock:
            items = list(self._inflight.items())
            self._inflight.clear()
        for k, raw in items:
            try:
                rid = json.loads(k)
            except Exception:  # noqa: BLE001
                rid = raw
            resp = {
                "jsonrpc": "2.0",
                "id": rid,
                "error": {"code": -32000, "message": reason},
            }
            self._write_stdout(
                (json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
        return len(items)

    def _write_stdout(self, data: bytes) -> None:
        # 回程线程与主线程（合成 error）可能同时写 ⇒ 必须串行 + 整帧写完。
        with self._stdout_lock:
            try:
                self._stdout.write(data)
                self._stdout.flush()
            except Exception:  # noqa: BLE001
                self._stopped.set()

    def _abandon(self) -> None:
        """自愈不了就明确放弃 —— 关 stdout + 非 0 退出。

        旧行为最坏的地方不是死，是**不死**：launcher 进程活着、stdout 已关、
        宿主任为连接还开着，于是每一次调用都打进空气里，22~33ms 返回一个
        毫无信息量的 ``Connection closed``。要么能用，要么明确地断。
        """
        self._stopped.set()
        self._drop_sock()
        try:
            self._stdout.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._stdout.close()
        except Exception:  # noqa: BLE001
            pass


def relay(sock: socket.socket) -> int:
    """把这条 daemon 连接接入宿主的 stdio，直到会话结束。

    返回码：0＝宿主关了 stdin（正常）；2/3＝起不来；4＝自愈失败已断开。
    """
    return _Relay(
        _SESSION_PARAMS.get("key", ""),
        _SESSION_PARAMS.get("library", ""),
        _SESSION_PARAMS.get("model", ""),
    ).run(sock)


def run_server_inline(argv: list[str] | None = None) -> int:
    """退化路径：直接跑原 server（与今天完全一致）。

    server 入口零参（``entry.run_main`` 不读命令行），argv 无处转发：
    形参保留只为回退分支的调用现场不动。
    """
    from .server import main as _server_main   # 惰性：这里才 import 重库
    return _server_main() or 0


def preflight_persist_dir() -> int | None:
    """启动前对库目录做一次 fail-fast 探测。

    与"任何一步出问题都退回 inline"的总原则**刻意相反**的一处：退化哲学
    护的是「launcher 自己的机制坏了，服务仍要能起」；而库目录不可写是
    **配置本身坏了** —— 服务照样起得来，只是每次写入都无声丢失，用户要
    等到"记不住任何东西"才发现。与其挂到宿主超时、被静默标记 failed，
    不如现在带明确 stderr 退出，把修复路径直接写在报错里。

    返回 None = 继续；返回 int = 以该码退出。
    """
    lib = os.environ.get(ENV_PERSIST, "").strip()
    if not lib:
        # 未设不算错（人工裸跑 launcher 调试是合法场景），但要自曝：
        # 此刻的默认库很可能不是用户以为的那个库 —— 串库从不报错。
        print(f"[launcher] ⚠ {ENV_PERSIST} is not set; this run will use the default library. "
              f"If launched by a host, the host probably did not read its config. "
              f"Run `python -m hippocampus_mcp.cli doctor --host <host>` for a verdict.",
              file=sys.stderr, flush=True)
        return None
    try:
        os.makedirs(lib, exist_ok=True)
        probe = os.path.join(lib, ".hippo-write-probe")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
    except OSError as exc:
        print(f"[launcher] ✗ Library directory is not writable; refusing to start: {lib}\n"
              f"  Reason: {type(exc).__name__}: {exc}\n"
              f"  Why: rather than let the host time out and silently fail every later write, "
              f"disconnect now.\n"
              f"  Fix: change {ENV_PERSIST} in the host config env to a writable directory, "
              f"or check disk/permissions and restart the session.",
              file=sys.stderr, flush=True)
        return 2
    return None


def main(argv: list[str] | None = None) -> int:
    """launcher 入口。

    任何一步出问题都退回 ``run_server_inline``——可用性 > 省内存。
    例外是库目录不可写：那是配置坏了，fail-fast（见 preflight_persist_dir）。
    """
    fail = preflight_persist_dir()
    if fail:
        return fail
    if not launcher_enabled():
        return run_server_inline(argv)
    try:
        library = os.environ.get("HIPPOCAMPUS_DAEMON_LIBRARY", "")
        model = os.environ.get("HIPPOCAMPUS_EMB_MODEL", "")
        version = os.environ.get("HIPPOCAMPUS_DAEMON_VERSION", "")
        key = daemon_key(library, model, version)
        # 「慢」与「坏」必须可区分：这一段最长要等 DAEMON_READY_TIMEOUT（默认
        # 180s，首次要加载嵌入模型），期间 stderr 一个字都没有的话，宿主与
        # 用户都只能判定为"挂了"，然后按"坏了"处理（杀进程 / 改配置）。
        # 先 flush 一行状态，让等待可被识别为等待。
        _diag(f"Connecting to shared daemon (first start loads the embedding model; tens of seconds is normal; "
              f"after {DAEMON_READY_TIMEOUT:.0f}s without readiness, falls back to direct server)")
        info = ensure_daemon(key, library=library, model=model)
        if info is None:
            _diag("Daemon failed to start; falling back to direct server (this session gets its own model; "
                  f"logs under {runtime_dir()}/daemon-*.log)")
            return run_server_inline(argv)
        sock = try_connect(int(info["port"]), timeout=5.0)
        if sock is None:
            _diag("Daemon registry is ready but its port is unreachable; falling back to direct server")
            return run_server_inline(argv)
        # _Relay 靠它知道"断了该往哪个 key 重连"
        _SESSION_PARAMS["key"] = key
        _SESSION_PARAMS["library"] = library
        _SESSION_PARAMS["model"] = model
        rc = relay(sock)
        if rc == 4:
            _diag("stdio disconnected deliberately; surface an explicit failure to the host instead of leaving a zombie connection")
        return rc
    except Exception:  # noqa: BLE001 —— 启动器不能因为任何原因让宿主连不上
        return run_server_inline(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
