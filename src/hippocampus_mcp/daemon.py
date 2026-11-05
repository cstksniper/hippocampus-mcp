# -*- coding: utf-8 -*-
"""hippocampusd：常驻 daemon。**模型全局一份**的关键就在这里。

分工：
- :mod:`hippocampus_mcp.launcher` 是纯标准库的薄壳，只负责找/起 daemon + 转发字节
- 本模块是常驻进程，**持有全部 HippoBrain**，一只进程可能同时服务多个库

为什么能省内存：单进程固定开销 755 MB 里的大头是嵌入模型权重，与记忆条数无关。
N 个会话各自起进程 = N 份权重；daemon 化之后是 1 份。

连接语义：每条 TCP 连接是一个独立的 MCP ``ServerSession``，
所以 ``ctx.session`` 天然按连接分路 —— 这就是会话自动寻址的通道。
**绝不复用连接**，否则分路出来的隐状态又粘回一起。

用法::

    HIPPOCAMPUS_DAEMON=1 python -m hippocampus_mcp.daemon

状态：实现已落地并在本地回环验证；**默认仍是直连**（见 launcher 的开关）。
切换默认需要一次有人值守的验证（宿主 stdio 语义 / 热升级 / 多库），
不在无人值守的自动化里改默认。
"""

# 豁免登记：本文件的 env 直读属进程拓扑 plumbing（socket/key /
# 哨兵阈值 / 诊断），封闭清单见 config.py「根层豁免清单」。
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from typing import Any

from .launcher import (read_registry, registry_path, runtime_dir,
                       write_registry)
from .engine import model_lifecycle
from .library.registry import DEFAULT_KEY


#: 心跳间隔（秒）。30s 是权衡：太密 ⟹ 白写盘；太疏 ⟹ 死亡时刻定位不准。
#: ``0`` 关闭（测试里不想轮询时用它）。
HEARTBEAT_INTERVAL = float(
    os.environ.get("HIPPOCAMPUS_DAEMON_HEARTBEAT", "30") or 0)
#: 每多少轮心跳往日志里落一行（避免 30s 一行把日志刷爆）。
HEARTBEAT_LOG_EVERY = 10

#: 空闲多久就把嵌入模型卸掉（秒）。**默认 30 分钟**：够盖住一次起身接水，
#: 又不至于让连续对话被反复加载拖垮；实测重载只要 2.5–3.7s。
#: ``0`` 关闭（保留"进程活着就永远加载"的旧行为）。
DEFAULT_MODEL_UNLOAD = 1800.0
ENV_MODEL_UNLOAD = "HIPPOCAMPUS_MODEL_IDLE_UNLOAD"
#: 卸载哨兵的巡检间隔（秒）。30s 一次，误差对 30 分钟阈值毫无意义。
MODEL_UNLOAD_POLL = 30.0

# ----------------------------------------------------------------------
# 空闲「睡眠窗口」——离线重放式巩固的宿主侧旋钮。
# 与 model unload 同款模式：daemon 进程拓扑决定，env 直读不走 config
# （口径：根层入口属启动拓扑 plumbing）。脑侧的三个语义旋钮
# （重放窗口 / 限幅 / 成簇下限）在 config.py，brain.core.sleep_batch
# 调用时现取。
# ----------------------------------------------------------------------
#: 空闲多少秒才允许开睡。**默认 0 = 关**（检索强度接线前先不改变
#: 生产行为；接线完成后翻默认另行验证）。判据是「大脑真实空闲」：
#: 无活跃连接 **且** ``model_lifecycle.idle_seconds()`` ≥ 本值。
DEFAULT_SLEEP_IDLE = 0.0
ENV_SLEEP_IDLE = "HIPPOCAMPUS_SLEEP_IDLE"
#: 单次睡眠批的时间预算兜底（秒），传给 ``brain.sleep_batch``。
DEFAULT_SLEEP_BUDGET = 120.0
ENV_SLEEP_BUDGET = "HIPPOCAMPUS_SLEEP_BUDGET_S"
#: 睡眠哨兵的巡检间隔（秒）。同 MODEL_UNLOAD_POLL 的理由：30s 粒度
#: 对睡眠这种分钟级活动毫无意义。
SLEEP_POLL = 30.0
from .library.registry import BrainRegistry  # 多库注册表下沉为公共模块

# 每条连接一次读多少字节
CONN_CHUNK = 65536
# 待办队列上限：单连接压过来的消息不会无限堆积
CONN_QUEUE = 256


# ======================================================================
# 连接处理
# ======================================================================
# 库注册表已下沉到 .registry（server 侧直连 stdio 也要多库，共用同一份）
class Connection:
    """一条客户端连接。

    当前实现是**字节泵 + 会话登记**：把连接交给真正的 MCP server 之前，
    先在这里挂上「这条连接 = 一个会话」的身份，供 ``ctx.session``
    分路使用。真正的 MCP 会话跑在 ``handle()`` 里（由 server 侧接手）。
    """

    def __init__(self, sock: socket.socket, addr, conn_id: int) -> None:
        self.sock = sock
        self.addr = addr
        self.conn_id = conn_id
        self.started_at = time.time()
        self.closed = False

    def session_key(self) -> str:
        """自动通道：一条连接一个 key，天然按会话分路。"""
        return f"conn-{self.conn_id}"


def _pump(src, dst) -> None:
    """单向字节泵（echo 排障通道用）。

    与 ``launcher._pump`` 同一条纪律：**用 ``read1`` 不用 ``read``**——
    ``read(n)`` 要读满 n 字节或 EOF 才返回，对管道会永久阻塞。
    """
    try:
        while True:
            if hasattr(src, "recv"):
                data = src.recv(CONN_CHUNK)
            elif hasattr(src, "read1"):
                data = src.read1(CONN_CHUNK)
            else:
                data = src.read(CONN_CHUNK)
            if not data:
                break
            if hasattr(dst, "sendall"):
                dst.sendall(data)
            else:
                dst.write(data)
    except Exception:  # noqa: BLE001
        pass


def serve_connection(conn: Connection, handler, on_exit=None) -> None:
    """把一条连接交给 handler（bytes -> bytes）。

    handler 抛异常不能带走 daemon —— 记一行，然后正常关连接。
    ``on_exit``：连接收尾回调（孤儿回收用它递减活跃连接计数）。
    """
    try:
        handler(conn)
    except Exception as exc:  # noqa: BLE001
        print(f"[daemon] 连接 {conn.conn_id} 处理异常: {type(exc).__name__}: {exc}",
              file=sys.stderr)
    finally:
        conn.closed = True
        try:
            conn.sock.close()
        except OSError:
            pass
        if on_exit is not None:
            try:
                on_exit(conn)
            except Exception:  # noqa: BLE001 —— 计数失败不该拖垮收尾
                pass


class Daemon:
    """TCP daemon：监听 127.0.0.1，每条连接一个线程。

    孤儿回收：launchers 全部死掉后 daemon 不该永久空转
    （它手里攥着 ~1GB 的模型）。``idle_exit_seconds`` > 0 时，活跃连接数
    归零并持续超过该秒数 ⇒ 自动 save 全部 brain + 清注册表 + 退出。
    有新连接进来即重置空闲计时。``0``/负值关闭（保留旧行为，测试用）。
    """

    def __init__(self, registry: BrainRegistry, handler=None,
                 idle_exit_seconds: float = 0.0, key: str = "",
                 heartbeat_seconds: float = HEARTBEAT_INTERVAL) -> None:
        self.registry = registry
        self.handler = handler
        self.key = key
        self.idle_exit_seconds = float(idle_exit_seconds or 0)
        self.heartbeat_seconds = float(heartbeat_seconds or 0)
        # 空闲卸载模型（注意：**不是**空闲退出进程）。
        # 与 idle_exit_seconds 的区别必须讲清楚：前者只是把 ~400MB 的嵌入模型
        # 还给系统，进程与记忆库都还在，下次调用自动重载（实测 2.5–3.7s）
        # 且结果与卸载前完全一致（余弦相似度 1.000000）；后者是连进程一起走。
        # 默认 1800s（30 分钟）；0/负值关闭。
        self.model_unload_seconds = float(
            os.environ.get(ENV_MODEL_UNLOAD, str(DEFAULT_MODEL_UNLOAD))
            or 0)
        # O2 睡眠窗口（同 model unload 的读法：垃圾取值直接炸出来，
        # 不做静默兜底——睡眠打错 env 该在启动时暴露，不是悄悄不生效）。
        self.sleep_idle_seconds = float(
            os.environ.get(ENV_SLEEP_IDLE, str(DEFAULT_SLEEP_IDLE))
            or 0)
        self.sleep_budget_s = float(
            os.environ.get(ENV_SLEEP_BUDGET, str(DEFAULT_SLEEP_BUDGET))
            or 0)
        self.sock: socket.socket | None = None
        self.port = 0
        self._conn_seq = 0
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._active_conns = 0
        self._last_conn_at = time.time()
        self._stopped = False
        self._heartbeat_round = 0
        # 收尾完成信号 + 空闲回收线程句柄（见 join_idle 的说明：
        # 线程必须是**非 daemon**，否则收尾会被解释器关闭掐断）
        self._finished = threading.Event()
        self._idle_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._model_unload_thread: threading.Thread | None = None
        # O2 睡眠批：取消信号（新连接到来即置位）+ 单飞标志 + 线程句柄
        self._sleep_cancel = threading.Event()
        self._sleep_running = False
        self._sleep_thread: threading.Thread | None = None

    # -- 生命周期 ------------------------------------------------------
    def bind(self) -> int:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(64)
        self.port = self.sock.getsockname()[1]
        return self.port

    def announce(self, key: str, library: str = "", model: str = "") -> bool:
        """把「我在哪」写进注册表，launcher 才能找过来。"""
        self.key = key
        return write_registry(key, {
            "pid": os.getpid(),
            "port": self.port,
            "key": key,
            "library": library,
            "model": model,
            "started_at": time.time(),
            "alive_at": time.time(),
            "runtime_dir": runtime_dir(),
        })

    def _idle_watch(self) -> None:
        """孤儿回收哨兵：空闲超时 ⇒ 自愿退出（save 在 stop() 里做）。"""
        interval = min(5.0, max(0.2, self.idle_exit_seconds / 4 or 5.0))
        while not self._stop.wait(interval):
            if self._active_conns > 0:
                continue
            idle = time.time() - self._last_conn_at
            if idle < self.idle_exit_seconds:
                continue
            print(
                f"[daemon] 孤儿回收：无活跃连接已 {idle:.0f}s"
                f"（阈值 {self.idle_exit_seconds:.0f}s），save 后退出",
                file=sys.stderr,
            )
            self.stop()
            return

    # -- 心跳 ----------------------------------------------------------
    def _touch_registry(self) -> bool:
        """把「我还活着」刷进注册表。

        为什么需要这个：真宿主里出现过**单日 15 个 daemon 无一例
        干净退出**（日志里 0 条"孤儿回收"）。要区分"它自愿退了"和"它被人杀了"，
        唯一的办法是留一个**不在崩溃路径上**的、持续刷新的时间戳：

          · 最后一条 ``alive_at`` ≈ 被杀的时刻（真 daemon 的 dated sepulchre）
          · 拿它与宿主的 ``mcp.server.closed`` 对表，就能证明回收动作来自
            **宿主进程树**，而不是我们自己的空闲哨兵 —— 这一步坐实之后，
            给 ``spawn_daemon`` 加 ``CREATE_BREAKAWAY_FROM_JOB`` 才有依据。

        🔴 **注册表里已经不是我时就别再写**：那说明后来者把我盖掉了，我是
        孤儿（"看不见的孤儿"），继续写会把活着的 daemon 的数据搅乱。
        """
        if not self.key:
            return False
        try:
            info = read_registry(self.key)
        except Exception:  # noqa: BLE001
            return False
        if not info or info.get("pid") != os.getpid():
            return False
        info["alive_at"] = time.time()
        info["conns"] = self._active_conns
        try:
            return bool(write_registry(self.key, info))
        except Exception:  # noqa: BLE001
            return False

    def _heartbeat_watch(self) -> None:
        interval = max(1.0, self.heartbeat_seconds)
        while not self._stop.wait(interval):
            try:
                self._touch_registry()
            except Exception:  # noqa: BLE001 —— 心跳绝不能带动主流程出错
                pass
            self._heartbeat_round += 1
            if self._heartbeat_round % HEARTBEAT_LOG_EVERY == 0:
                print(
                    f"[daemon] heartbeat pid={os.getpid()} "
                    f"conns={self._active_conns} t={time.time():.3f}",
                    file=sys.stderr,
                )

    def _model_unload_watch(self) -> None:
        """空闲卸载哨兵：超阈值且无人在用 ⇒ 把嵌入模型卸掉。

        为什么看「最后一次用模型」而不是「活跃连接数」
        -------------------------------------------
        宿主侧的 launcher 会把 socket 一直攥到会话结束，僵尸 launcher 更是
        永不放手 ⇒ ``_active_conns`` 可能长期不为 0。这大概率就是"横幅写着
        ``idle_exit=600s`` 却一次都没退过"的原因。用 :func:`model_lifecycle
        .idle_seconds` 看真实用量，才不会被一条空连接骗过去。
        """
        while not self._stop.wait(MODEL_UNLOAD_POLL):
            try:
                idle = model_lifecycle.idle_seconds()
                if idle < self.model_unload_seconds:
                    continue
                res = model_lifecycle.release_models(reason="idle-timeout")
                # busy（有人在用）不是错误：下一轮自然会再试。只在**真的卸了**
                # 或者出了意外时留痕，避免日志被"每次都说忙"刷满。
                if res.get("ok"):
                    print(
                        f"[daemon] 空闲卸载模型 pid={os.getpid()} "
                        f"idle={idle:.0f}s 释放={res.get('reclaimed_mb')}MB "
                        f"({res.get('before_mb')}→{res.get('after_mb')}MB) "
                        f"pools={res.get('pools')} conns={self._active_conns}",
                        file=sys.stderr, flush=True,
                    )
                elif res.get("reason") != "busy":
                    print(f"[daemon] 卸载模型失败: {res}",
                          file=sys.stderr, flush=True)
            except Exception as exc:  # noqa: BLE001 —— 哨兵绝不能带崩 daemon
                print(f"[daemon] 卸载哨兵异常: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)

    # -- O2 睡眠窗口 ----------------------------------------------------
    def _sleep_watch(self) -> None:
        """睡眠哨兵：空闲期对每个驻留库跑一轮「睡眠批」（离线重放式巩固）。

        空闲判据是两个条件的**与**，各有不可替代的理由：

        - ``_active_conns == 0``：正在接客 ⇒ 不睡。但它**不能单独**作判据——
          宿主侧 launcher 会把 socket 攥到会话结束，僵尸 launcher 更是永不
          放手（model-unload 哨兵的同款病理）；
        - ``model_lifecycle.idle_seconds() >= sleep_idle_seconds``：大脑
          真实空闲。``idle_seconds()==0`` 表示「从未用过」⇒ 也不睡，这是
          **刻意**的：一台没服务过任何会话的 daemon 没有「最近会话」可以
          重放，睡眠无米下锅。

        中断契约：新连接进来时 ``serve_forever`` 立刻置 ``_sleep_cancel``，
        睡眠批（``brain.sleep_batch``）在簇/阶段等**单元边界**感知并尽快
        让路——在线请求与睡眠分属不同线程，请求不必等睡眠结束，睡眠只是
        尽快停手减少 CPU / 锁竞争。单飞由 ``_sleep_running`` 保证（脑内
        ``sleep_batch`` 的重入守卫是第二道防线）。
        """
        while not self._stop.wait(SLEEP_POLL):
            if self.sleep_idle_seconds <= 0:
                continue
            if self._active_conns > 0 or self._sleep_running:
                continue
            try:
                idle = model_lifecycle.idle_seconds()
            except Exception:  # noqa: BLE001 —— 计时失败按「不睡」处理
                continue
            if idle < self.sleep_idle_seconds:
                continue
            self._sleep_cancel.clear()
            self._sleep_running = True
            try:
                self._run_sleep_once(idle)
            except Exception as exc:  # noqa: BLE001 —— 哨兵绝不能带崩 daemon
                print(f"[daemon] 睡眠批异常: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
            finally:
                self._sleep_running = False

    def _run_sleep_once(self, idle: float) -> None:
        """对**驻留**库各跑一轮睡眠批。

        为什么只遍历 ``registry.keys()`` 而不用别的入口：构造一个 brain
        要 25–33s 还要加载模型，而睡眠恰恰发生在模型已卸载之后——为睡眠
        去构造 brain 是把语义整个倒过来。keys() 只返回驻留库；keys() 与
        get() 之间若有并发 get() 换出该库，get() 会重新构造——空闲期没有
        连接、没人触发换出，该窗口可忽略，且就算踩中也只是浪费（同 key
        = 同一份存储），不是正确性问题。
        """
        for key in self.registry.keys():
            if self._stop.is_set() or self._sleep_cancel.is_set():
                return
            # 🔴 keys() 返回的是**注册表键**（DEFAULT_KEY 或归一化绝对路径），
            # get() 吃的是**库路径**——DEFAULT_KEY 不是路径，直接喂给 get()
            # 会被 abspath 成 "<cwd>/__default__"，同一条逻辑库被建成两个
            # brain（C2e 目标测试实测抓到）。路径键喂回去是恒等（normcase
            # ∘ abspath 幂等），只有默认键需要映射回 None。
            brain = self.registry.get(None if key == DEFAULT_KEY else key)
            if brain is None or not hasattr(brain, "sleep_batch"):
                continue
            t0 = time.time()
            try:
                report = brain.sleep_batch(
                    cancel=self._sleep_cancel, budget_s=self.sleep_budget_s)
            except Exception as exc:  # noqa: BLE001 —— 单库失败不挡其它库睡
                print(f"[daemon] 睡眠批失败 lib={key}: "
                      f"{type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
                continue
            print(
                f"[daemon] 睡眠批 pid={os.getpid()} lib={key} "
                f"idle={idle:.0f}s took={time.time() - t0:.1f}s "
                f"clusters={report.get('replay_clusters')} "
                f"records={report.get('replay_records')} "
                f"strengthened={report.get('strengthened')} "
                f"consolidated={report.get('consolidated')} "
                f"dormant={report.get('dormant_archived')} "
                f"interrupted={report.get('interrupted')}",
                file=sys.stderr, flush=True,
            )

    def serve_forever(self) -> None:
        if self.sock is None:
            raise RuntimeError("先 bind()")
        if self.model_unload_seconds > 0:
            # daemon=True：这条线程不持有任何需要收尾的东西
            if self._model_unload_thread is None:
                self._model_unload_thread = threading.Thread(
                    target=self._model_unload_watch, name="model-unload",
                    daemon=True)
                self._model_unload_thread.start()
        if self.sleep_idle_seconds > 0:
            # O2 睡眠哨兵。daemon=True：不持有需要收尾的东西（睡眠批
            # 中途被解释器关闭掐断与被 cancel 掐断同义——各步幂等）。
            if self._sleep_thread is None:
                self._sleep_thread = threading.Thread(
                    target=self._sleep_watch, name="sleep-watch",
                    daemon=True)
                self._sleep_thread.start()
        if self.idle_exit_seconds > 0:
            # daemon=False：**不要**让解释器关闭时把正在收尾的线程掐断
            self._idle_thread = threading.Thread(
                target=self._idle_watch, name="idle-watch", daemon=False)
            self._idle_thread.start()
        if self.heartbeat_seconds > 0:
            # daemon=True：心跳不持有任何需要收尾的东西，解释器关闭时可以被掐
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_watch, name="heartbeat", daemon=True)
            self._heartbeat_thread.start()
        while not self._stop.is_set():
            try:
                client, addr = self.sock.accept()
            except OSError:
                break
            if self._stop.is_set():
                # ``stop()`` 的自连接唤醒包：不能交给 handler，也不能计入
                # ``_active_conns``（否则空闲计时会被这条假连接重置）。
                # 这里直接关掉并退出循环，让主线程能走到 finally 的收尾。
                try:
                    client.close()
                except OSError:
                    pass
                break
            self._conn_seq += 1
            conn = Connection(client, addr, self._conn_seq)
            self._active_conns += 1
            self._last_conn_at = time.time()
            # O2：唤醒睡眠批——新连接立即被接（线程已起），睡眠批在下一个
            # 单元边界感知 cancel 让路，减少与在线请求的 CPU / 锁竞争。
            self._sleep_cancel.set()
            t = threading.Thread(
                target=serve_connection,
                args=(conn, self.handler or _default_handler),
                kwargs={"on_exit": self._on_conn_exit},
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def _on_conn_exit(self, conn: Connection) -> None:
        self._active_conns = max(0, self._active_conns - 1)
        self._last_conn_at = time.time()

    def _wake_accept(self) -> None:
        """把阻塞在 ``accept()`` 的主线程叫醒（跨平台，见下方说明）。

        🔴 Linux 上**从别的线程 close(listening fd) 不会中断已经在 accept()
        里阻塞的调用**：该 syscall 仍持有 file 结构的引用，``close()`` 只减掉
        fd 的引用，socket 照旧 LISTEN、accept 照旧等。两个后果：

        - ``serve_forever`` 永不返回 ⇒ 空闲哨兵调完 ``stop()`` 后进程仍然
          活着（CI 上 ``batch2_finalize_daemon`` 的 D1/D2 ``returncode=None``）；
        - 端口仍可连接 ⇒ ``f9_4_launcher_daemon`` 的 L4「stop 后连不上」红。

        Windows 的 ``closesocket`` 会让 accept 抛 OSError，所以本机从没暴露。
        唯一跨平台可靠的办法是**自己连自己一次**：accept 拿到一个连接自然
        返回，主循环看到 ``_stop`` 已置位就退出；随后 fd 的最后一个引用被
        释放，监听端口才真正关闭。
        """
        sock = self.sock
        if sock is None:
            return
        try:
            port = sock.getsockname()[1]
        except OSError:
            return
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                pass
        except OSError:
            # 端口已经关了 / 已经没人 listen 都不影响：那说明 accept 本来
            # 就能返回，唤醒只是白跑一趟。
            pass

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        # O2：叫停可能在跑的睡眠批（各步幂等，停在哪一层都安全）
        self._sleep_cancel.set()
        # 🔴 顺序：先唤醒再 close。close 之后 getsockname() 就失效了；而
        # Linux 上单靠 close 根本叫不醒 accept（见 _wake_accept 的说明）。
        self._wake_accept()
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        # 收尾落盘：**不能因为保存失败就不退出**，但要把失败吼出来
        errors = self.registry.save_all()
        if errors:
            print(f"[daemon] 退出前保存失败: {json.dumps(errors, ensure_ascii=False)}",
                  file=sys.stderr)
        # 清注册表：本 daemon 已退出，别让 launcher 撞上死条目多等一轮探测
        # （launcher 本就有 pid+port 存活校验，这里只是把现场擦干净）
        if self.key:
            path = registry_path(self.key)
            try:
                os.remove(path)
                print(f"[daemon] 已擦注册表: {path}", file=sys.stderr)
            except OSError as e:
                # 擦不掉要吼出来：残留条目会让下一个 launcher 多等一轮存活探测，
                # 而它看起来和"daemon 还活着"一模一样。
                print(f"[daemon] 擦注册表失败: {path} -> {type(e).__name__}: {e}",
                      file=sys.stderr)
        self._finished.set()

    def join_idle(self, timeout: float = 20.0) -> None:
        """等空闲回收线程把收尾跑完。

        🔴 为什么必须有这一步（实测踩过的真 bug）：
        空闲线程是 **daemon 线程**，它调用 ``stop()`` → 关 socket → 主线程 ``accept()``
        抛 OSError → ``serve_forever`` 返回 → 解释器进入关闭流程，**直接把还在跑收尾的
        daemon 线程掐断**。后果是：

        - 日志里明明写着「孤儿回收：…save 后退出」，但 ``save_all()`` 和
          **擦注册表** 都没跑完 ⇒ 注册表留下死条目，下一个 launcher 会撞上它
          多等一轮存活探测（表现与"daemon 还活着"一模一样）；
        - ``test_batch2_finalize_daemon`` 的 D1 就是被这个坑掉的（不是 flaky）。

        修法两条一起：空闲线程改**非 daemon**（解释器关闭前会等它）+ 这里显式 join。
        """
        t = self._idle_thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=timeout)
        # 兜底：万一线程没能自己收尾（例如被卡住），在这里补一次——
        # 幂等（_stopped 已置位时 stop() 直接返回，不会重复 save）。
        if not self._finished.is_set():
            self.stop()


def _default_handler(conn: Connection) -> None:
    """占位 handler：收到什么回什么（便于本地回环验证，也便于排障）。

    真正的 MCP 会话由 server 侧接手后替换；这里刻意**不**假装能处理 JSON-RPC。
    """
    _pump(conn.sock, conn.sock)


def _make_handler():
    """挑 handler。**默认必须是真 MCP 会话**，不是 echo 泵。

    echo 泵会让宿主「连上了、却永远收不到合法响应」，于是工具数静默变成 0——
    比「连不上」难排得多（实测 41% 丢工具就是这样没有线索的）。
    所以桥不可用就**明确报错**，让 launcher 退回 inline server；
    只有显式 ``HIPPOCAMPUS_DAEMON_ECHO=1`` 才允许退化成 echo（本地排障用）。
    """
    if os.environ.get("HIPPOCAMPUS_DAEMON_ECHO", "").strip() == "1":
        print("[daemon] HIPPOCAMPUS_DAEMON_ECHO=1 -> echo 泵（仅供本地排障）",
              file=sys.stderr)
        return _default_handler
    try:
        from .bridge import bridge_available, make_mcp_handler  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"MCP 会话桥不可用（缺 anyio / mcp）: {type(exc).__name__}: {exc}"
        ) from exc
    if not bridge_available():
        raise RuntimeError("MCP 会话桥自检失败：anyio / mcp 不可用")
    return make_mcp_handler()


def _install_crash_diagnostics() -> str:
    """给 **native 级**死亡留痕（此前唯一的观测缺口）。

    背景：真宿主里出现过 daemon **无声死亡**——日志无 traceback、无 WER 记录、
    注册表也没被清理（区别于正常的空闲回收），进程甚至没来得及初始化 brain。
    Python 层异常本来就会打到 stderr（launcher 已把 stderr 落盘到
    ``daemon-<key>.log``），所以这里补的是**连 Python 都没机会出手**的那部分：
    ``SIGSEGV`` / ``SIGABRT`` / ``SIGFPE`` / ``SIGILL`` 这类致命信号。

    一条判据（写在注释里备用，将来复盘无声死亡时直接用）：

    - 装了 faulthandler **仍然一字未留** ⇒ 进程是被外部终止的
      （Windows 的 ``TerminateProcess`` 不触发任何 handler，Python 的
      ``os.kill(pid, SIGTERM)`` 在 Windows 上同样是 TerminateProcess）；
    - 装了之后有 dump ⇒ 是 native 层崩溃，按栈继续查。

    为什么 ``chain=True`` 不能改成 ``False``：``False`` 会在 dump 完之后
    **不再调用原 handler**，也就是收到 ``SIGTERM`` 后进程**继续活着**——
    宿主以为自己杀了 daemon，其实它还在那里攥着 ~1GB。``chain=True``
    是「dump 之后照旧终止」，语义不变。

    逃生开关：``HIPPOCAMPUS_DAEMON_FAULTHANDLER=0`` 关闭。
    """
    if os.environ.get("HIPPOCAMPUS_DAEMON_FAULTHANDLER", "1").strip() in ("0", "false", "no", "off"):
        return "off"
    try:
        import faulthandler
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:  # noqa: BLE001 —— 诊断设施不能让它自己的失败拖垮 daemon
        return "unavailable"
    # enable() 本身就会给 SIGSEGV / SIGFPE / SIGABRT / SIGILL 这类**致命信号**
    # 装 handler —— 也就是说 native 崩溃留痕的主体能力它已经给了。
    # register() 只是补充"给任意信号也加一次 dump"；实测两个主流解释器
    # （CPython 3.11 与 3.13）的 faulthandler 都是**精简内置版**，
    # hasattr(faulthandler, "register") == False，故这条通道通常拿不到，
    # 拿到不到就明确记出来，别让复盘的人误以为注册过。
    if not hasattr(faulthandler, "register"):
        return "enable(no-register)"
    installed = ["enable"]
    for signame in ("SIGTERM", "SIGBREAK"):
        signum = getattr(__import__("signal"), signame, None)
        if signum is None:
            continue
        try:
            faulthandler.register(signum, file=sys.stderr, all_threads=True, chain=True)
            installed.append(signame)
        except Exception:  # noqa: BLE001 —— 个别平台没有这些信号，忽略即可
            pass
    return "+".join(installed)


def main(argv: list[str] | None = None) -> int:
    key = os.environ.get("HIPPOCAMPUS_DAEMON_KEY", "") or "default"
    library = os.environ.get("HIPPOCAMPUS_DAEMON_LIBRARY", "")
    model = os.environ.get("HIPPOCAMPUS_DAEMON_MODEL", "")
    # 孤儿回收：launchers 全死后空闲超时自动退出。默认 600s；
    # <=0 关闭（保留"永不自动退出"的旧行为，测试用）。
    try:
        idle_exit = float(os.environ.get("HIPPOCAMPUS_DAEMON_IDLE_EXIT", "600") or 600)
    except ValueError:
        idle_exit = 600.0
    registry = BrainRegistry(factory=_default_factory)
    d = Daemon(registry, handler=_make_handler(),
               idle_exit_seconds=idle_exit, key=key)
    diag = _install_crash_diagnostics()
    d.bind()
    if not d.announce(key, library=library, model=model):
        print(f"[daemon] 写注册表失败: {registry_path(key)}", file=sys.stderr)
        return 1
    print(f"[daemon] key={key} port={d.port} pid={os.getpid()} "
          f"idle_exit={idle_exit:.0f}s "
          f"sleep={'off' if d.sleep_idle_seconds <= 0 else f'{d.sleep_idle_seconds:.0f}s'} "
          f"faulthandler={diag}", file=sys.stderr)
    try:
        d.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        d.join_idle()      # 先等空闲回收线程把收尾（save + 擦注册表）跑完
        d.stop()           # 幂等兜底
    return 0


def _default_factory(library: str | None = None):
    """惰性建 brain（这里才 import 重库）。"""
    from .brain import HippoBrain
    return HippoBrain(persist_dir=library or None)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
