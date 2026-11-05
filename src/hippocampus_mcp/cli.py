"""``hippocampus-mcp`` 命令行入口（setup / sync / doctor / models / serve）。

五个子命令::

    hippocampus-mcp serve     启动 MCP 服务（默认经 launcher，共享 daemon）
    hippocampus-mcp setup     为一个宿主生成 / 合并 MCP 配置
    hippocampus-mcp sync      发现所有宿主配置并同步同一条目（默认预览）
    hippocampus-mcp doctor    体检：路径 / 库 / 模型 / 依赖是否就绪
    hippocampus-mcp models    本地模型缓存：占用明细 / 谁在用 / 回收冗余

**为什么要有 setup**：MCP 协议只规定传输（stdio / SSE）与 JSON-RPC，
**没有规定配置文件放哪**。各宿主自行实现，连 JSON 顶层键名都不同
（WorkBuddy 与 TRAE 用 ``mcpServers``，ZCode 用 ``mcp.servers``），
本项目无法让宿主改掉这些差异。

但差异可以**吸收进本命令**：用户只跑一条命令、只维护一份意图
（用哪个库、哪个剖面、哪个模型），由本命令翻译成各宿主各自的格式。
用户侧看到的体验是统一的 —— 这正是 setup 存在的全部理由。

**安全纪律**（写宿主配置时最容易出事的四点）：

1. **写前必备份**：目标文件先复制成 ``<名字>.bak-<时间戳>``，再写。
2. **解析失败就停**：既有配置 JSON 损坏时**不覆盖**，报错并让人手工修 ——
   那里面可能还有别的 MCP server，覆盖掉是连带损失。
3. **只动自己那一条**：合并而非整文件替换，别的 server 条目原样保留。
4. **写后即验**：写完自动按写入的配置**真实握手一次**（initialize +
   tools/list），起不来当场给出原因 —— 不把发现时机留给用户重启宿主
   撞上"工具数为 0"的那一刻。``--no-verify`` 可跳过。
5. **sync 默认只预览**：它面向 ``_HOSTS`` 之外的宿主（文件名约定 +
   形态嗅探，见 ``host_configs``），写错地方的代价比 setup 更高，
   所以与 ``models`` 的删盘同款纪律 —— 不带 ``--yes`` 一个字节都不动，
   且只对已存在的文件动手，绝不新建。
"""

# 豁免登记：本文件的 env 直读是自省命令的展示透传（读给人看，
# 值与 config 一致），封闭清单见 config.py「根层豁免清单」。

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from . import host_configs as hc

SERVER_NAME_DEFAULT = "hippocampus"
LIBRARY_DEFAULT = "~/.hippocampus/brain"
MODEL_DEFAULT = "BAAI/bge-base-zh-v1.5"
PROFILE_DEFAULT = "common"

# 宿主 → 配置文件的 JSON 键路径（**差异就在这里**，别处无差别）
# key_path 是嵌套键：("mcpServers",) 与 ("mcp", "servers") 两种都要能读写。
_HOSTS: dict[str, dict] = {
    "workbuddy": {
        "label": "WorkBuddy",
        "key_path": ("mcpServers",),
        "default_scope": "user",
    },
    "zcode": {
        "label": "ZCode",
        "key_path": ("mcp", "servers"),
        "default_scope": "project",
    },
    "trae": {
        "label": "TRAE SOLO CN",
        "key_path": ("mcpServers",),
        "default_scope": "user",
    },
    "claude": {
        "label": "Claude Desktop / Code",
        "key_path": ("mcpServers",),
        "default_scope": "user",
    },
}


# --------------------------------------------------------------------------
# 路径推导
# --------------------------------------------------------------------------
def user_config_path(host: str) -> Path | None:
    """宿主的用户级配置路径；该宿主没有用户级配置时返回 None。"""
    home = Path.home()
    appdata = os.environ.get("APPDATA", "")
    if host == "workbuddy":
        return home / ".workbuddy" / "mcp.json"
    if host == "trae":
        if appdata:
            return Path(appdata) / "TRAE SOLO CN" / "User" / "mcp.json"
        return (home / "Library" / "Application Support" / "TRAE SOLO CN"
                / "User" / "mcp.json")
    if host == "claude":
        if sys.platform == "darwin":
            return (home / "Library" / "Application Support" / "Claude"
                    / "claude_desktop_config.json")
        if appdata:
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
        return home / ".config" / "Claude" / "claude_desktop_config.json"
    if host == "zcode":
        # 2026-10 升级后用户级配置迁到 cli/ 子目录（旧 ~/.zcode/config.json 已废弃）。
        # 此前本函数对 zcode 返回 None——doctor/sync 看不见用户级配置，
        # 09-28 补在旧位置的条目升级后丢失也无人能体检出来（盲区事故，
        # 见 docs/design/ZCode用户级接入迁移_20261030.md §一）。
        return home / ".zcode" / "cli" / "config.json"
    return None


def project_config_path(host: str, workspace: Path) -> Path | None:
    """宿主的工作区级配置路径。"""
    if host == "workbuddy":
        return workspace / ".mcp.json"
    if host == "zcode":
        return workspace / ".zcode" / "config.json"
    return None


def package_src_dir() -> tuple[Path | None, Path]:
    """返回 (src 目录 | None, 包目录)。

    从 ``pip install`` 的 site-packages 里跑时**不需要** PYTHONPATH（包本来
    就在 sys.path 上），此时返回 None；从仓库的 ``src/`` 布局跑时才需要。
    这个区分很重要 —— 无脑写 PYTHONPATH 会在别人机器上指向一个不存在的目录。
    """
    pkg_dir = Path(__file__).resolve().parent
    parent = pkg_dir.parent
    if parent.name == "src":
        return parent, pkg_dir
    return None, pkg_dir


# --------------------------------------------------------------------------
# 配置读写（只动自己那一条）
# --------------------------------------------------------------------------
def _dig(cfg: dict, key_path: tuple[str, ...], create: bool = False) -> dict | None:
    """按键路径取容器；``create=True`` 时逐层补空字典。"""
    node: dict = cfg
    for key in key_path:
        nxt = node.get(key)
        if not isinstance(nxt, dict):
            if not create:
                return None
            nxt = {}
            node[key] = nxt
        node = nxt
    return node


def build_entry(*, python: str, src_dir: Path | None, library: str,
                model: str, profile: str, offline: bool) -> dict:
    """构造单个 MCP server 条目（各宿主共用同一份，差异只在壳上）。"""
    env: dict[str, str] = {}
    if src_dir is not None:
        env["PYTHONPATH"] = str(src_dir)
    env["HIPPOCAMPUS_PERSIST_DIR"] = library
    env["HIPPOCAMPUS_EMB_MODEL"] = model
    env["HIPPOCAMPUS_STORE_BACKEND"] = "sqlite"
    env["HIPPOCAMPUS_FTS5"] = "1"
    env["HIPPOCAMPUS_TOOL_PROFILE"] = profile
    env["HIPPOCAMPUS_LAUNCHER"] = "1"
    if offline:
        # 只在用户明确要离线时加：开源用户第一次跑需要下载模型，
        # 默认开 offline 会让人卡在"模型加载不出来"且无提示。
        env["HF_HUB_OFFLINE"] = "1"
    return {
        "type": "stdio",
        "command": python,
        "args": ["-m", "hippocampus_mcp.launcher"],
        "env": env,
    }


def _read_config(path: Path) -> dict:
    cfg, err = hc.load_json_config(path)
    if cfg is None:
        # 坏文件升级为停手：setup/sync 的写入路径必须在这里死掉，
        # 而不是把别人的 server 条目连带覆盖掉（发现式扫描可以带伤
        # 继续，写入不行——语义差异在 host_configs.load_json_config）。
        raise SystemExit(
            f"配置文件{err}，已停手不写：{path}\n"
            f"  它里面可能还有别的 MCP server，覆盖掉是连带损失。"
            f"请手工修好后再跑本命令。")
    return cfg


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path.with_name(f"{path.name}.bak-{stamp}")
    shutil.copy2(path, dst)
    return dst


def _detect_host(workspace: Path) -> str:
    """未指定 ``--host`` 时按工作区特征猜宿主。

    判据刻意保守：只认「目录里已经存在的宿主痕迹」（``.zcode/`` 目录、
    ``.mcp.json`` 文件），猜不出来就报错列出可选值，绝不静默选一个 ——
    setup 写的是宿主的真配置，写错宿主比不写更糟。
    """
    if (workspace / ".zcode").is_dir():
        return "zcode"
    if (workspace / ".mcp.json").is_file():
        return "workbuddy"
    raise SystemExit(
        "无法判断目标宿主：工作区里既没有 .zcode/ 也没有 .mcp.json。\n"
        f"  请显式指定 --host（可选：{', '.join(sorted(_HOSTS))}）。")


# --------------------------------------------------------------------------
# 写后即验（真实握手）
# --------------------------------------------------------------------------
VERIFY_TIMEOUT = 120.0


def _stderr_tail(fh) -> str:
    """读子进程 stderr 的尾部（探测失败时给用户的第一手线索）。"""
    try:
        fh.seek(0)
        txt = (fh.read() or "").strip()
    except Exception:  # noqa: BLE001 —— 尾巴读不出就算了，别盖住主因
        return ""
    if not txt:
        return ""
    if len(txt) > 600:
        txt = "…" + txt[-600:]
    return f"stderr 尾部：{txt}"


def _verify_entry(entry: dict, cwd: str = "",
                  timeout: float = VERIFY_TIMEOUT) -> tuple[bool, str]:
    """按写进配置的 command/args/env **真实拉起一次握手**。

    setup 此前只保证「配置写进去了」，不保证「宿主照着起得来」——解释器
    路径错了、PYTHONPATH 没生效、包没装上，都要等用户重启宿主撞上
    "工具数为 0" 才发现。这里把发现时机提前到 setup 当场：spawn 子进程，
    走一遍 initialize + tools/list，能用/不能用于几秒内见分晓（握手不触发
    模型加载，不会卡在分钟级的下载上）。

    返回 ``(ok, 明细)``。失败的明细必须**带原因**（超时 / 提前退出 /
    报错响应 / stderr 尾部），否则用户还是得自己去翻日志。

    走 launcher 路径时（配置里的 ``HIPPOCAMPUS_LAUNCHER=1``），探测会顺手
    把 daemon 拉起来；探测进程退出后 daemon 留存保温，空闲后自动回收，
    宿主下次启动反而是热的 —— 这是副作用，不是缺陷。
    """
    import queue
    import subprocess
    import tempfile
    import threading

    cmd = [entry.get("command") or sys.executable] + list(entry.get("args") or [])
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (entry.get("env") or {}).items()})
    try:
        err_fh = tempfile.TemporaryFile(mode="w+", encoding="utf-8",
                                        errors="replace")
        proc = subprocess.Popen(cmd, cwd=cwd or None, env=env,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=err_fh)
    except OSError as exc:
        return False, f"起不了进程 {cmd[0]}：{type(exc).__name__}: {exc}"

    lines_q: queue.Queue = queue.Queue()

    def _reader() -> None:
        try:
            for raw in proc.stdout:          # 按行阻塞读；进程退出后循环自然结束
                lines_q.put(raw)
        finally:
            lines_q.put(None)

    threading.Thread(target=_reader, daemon=True).start()

    def _readline(deadline: float) -> bytes:
        remain = deadline - time.monotonic()
        if remain <= 0:
            raise TimeoutError("超时")
        try:
            item = lines_q.get(timeout=remain)
        except queue.Empty:
            raise TimeoutError("超时") from None
        if item is None:
            raise EOFError("子进程提前退出（未等到响应）")
        return item

    def _send(obj: dict) -> None:
        proc.stdin.write(
            (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        proc.stdin.flush()

    deadline = time.monotonic() + timeout
    try:
        _send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
               "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                          "clientInfo": {"name": "hippocampus-setup-verify",
                                         "version": "0"}}})
        # 理论上第一条响应就是 initialize 的；按 id 对账可以扛住带外行。
        while True:
            obj = json.loads(_readline(deadline).decode("utf-8", "replace"))
            if obj.get("id") == 1:
                break
        if "result" not in obj:
            return False, f"initialize 返回错误：{obj.get('error')}"
        _send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        _send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        while True:
            obj = json.loads(_readline(deadline).decode("utf-8", "replace"))
            if obj.get("id") == 2:
                break
        if "result" not in obj:
            return False, f"tools/list 返回错误：{obj.get('error')}"
        tools = obj["result"].get("tools") or []
        if not tools:
            return True, "initialize + tools/list 正常（但 0 个工具？检查剖面）"
        return True, f"initialize + tools/list 正常，{len(tools)} 个工具可见"
    except TimeoutError:
        return False, (f"{timeout:.0f}s 内没等到握手响应（卡在加载或网络）——"
                       "慢≠坏，但反复如此就要看 daemon 日志了")
    except EOFError:
        return False, (f"子进程未完成握手即退出（exit={proc.poll()}）。"
                       f"{_stderr_tail(err_fh)}")
    except Exception as exc:  # noqa: BLE001 —— 探测器自己不能把 setup 带崩
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        for closer in (proc.stdin, proc.stdout):
            try:
                closer.close()
            except Exception:  # noqa: BLE001
                pass
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()
        err_fh.close()


# --------------------------------------------------------------------------
# 子命令：setup
# --------------------------------------------------------------------------
def _pick_value(value: str | None, old_value: str | None, default: str
                ) -> tuple[str, str]:
    """取值优先级：命令行 > 既有配置 > 默认。

    中间那档**不能省**：省略的话，在本机跑一次 setup / sync 就会把
    既有库换成新用户默认值 —— 用户以为只是"补个配置"，实际库被搬走了，
    而且是静默的。这是写宿主配置最容易造成的破坏，所以显式写出来。
    setup 与 sync 共用这条规则（sync 还要求**逐文件**继承：不同宿主
    指不同库是合法状态，不是要被抹平的差异）。
    """
    if value:
        return value, "命令行指定"
    if old_value:
        return old_value, "沿用既有配置"
    return default, "新装默认值"


def cmd_setup(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    if args.host:
        host = args.host
    else:
        # S4：不带 --host 时按工作区特征自动判定（有 .zcode/ → zcode；
        # 有 .mcp.json → workbuddy）。猜不出来会明确报错，绝不瞎选。
        host = _detect_host(workspace)
        print(f"未指定 --host，按工作区特征自动判定：{host}")
    spec = _HOSTS[host]

    if args.file:
        target = Path(args.file).expanduser().resolve()
    else:
        scope = args.scope or spec["default_scope"]
        guess = (user_config_path(host) if scope == "user"
                 else project_config_path(host, workspace))
        if guess is None:
            raise SystemExit(
                f"{spec['label']} 没有 {scope} 级配置位置，"
                f"请用 --file 明确给出配置文件路径。"
            )
        target = guess

    cfg = _read_config(target)
    container = _dig(cfg, spec["key_path"], create=True)
    old = container.get(args.name)
    old = old if isinstance(old, dict) else {}
    old_env = old.get("env") if isinstance(old.get("env"), dict) else {}
    existed = bool(old)

    src_dir, _pkg = package_src_dir()

    library, lib_src = _pick_value(args.library,
                                   old_env.get("HIPPOCAMPUS_PERSIST_DIR"),
                                   str(Path(LIBRARY_DEFAULT).expanduser()))
    model, model_src = _pick_value(args.model,
                                   old_env.get("HIPPOCAMPUS_EMB_MODEL"),
                                   MODEL_DEFAULT)
    profile, prof_src = _pick_value(args.profile,
                                    old_env.get("HIPPOCAMPUS_TOOL_PROFILE"),
                                    PROFILE_DEFAULT)
    python = args.python or old.get("command") or sys.executable
    offline = args.offline or old_env.get("HF_HUB_OFFLINE") == "1"

    entry = build_entry(python=python, src_dir=src_dir, library=library,
                        model=model, profile=profile, offline=offline)
    container[args.name] = entry

    payload = json.dumps(cfg, ensure_ascii=False, indent=2)

    print(f"宿主      : {spec['label']} ({host})")
    print(f"配置文件  : {target}")
    print(f"条目      : {args.name}（{'覆盖既有' if existed else '新建'}，"
          f"其余 server 条目原样保留）")
    print(f"解释器    : {python}")
    print(f"库目录    : {library}  [{lib_src}]")
    print(f"模型      : {model}  [{model_src}]")
    print(f"工具剖面  : {profile}  [{prof_src}]")
    if src_dir is None:
        print("PYTHONPATH: 未设置（包已装在 sys.path 上，不需要）")
    else:
        print(f"PYTHONPATH: {src_dir}")

    if args.dry_run:
        print("\n[dry-run] 以下内容**未写入**，仅预览：\n")
        print(payload)
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    bak = _backup(target)
    if bak:
        print(f"\n已备份    : {bak}")
    target.write_text(payload + "\n", encoding="utf-8")
    print(f"\n已写入    : {target}")

    if args.no_verify:
        print("[跳过写后即验]（--no-verify）：可用性未经验证，"
              "重启宿主后请调 library_whoami 确认")
    else:
        # S2 写后即验：按**刚写入**的 entry 真实握手一次。失败给原因，
        # 让"配置坏了"在 setup 当场现形，而不是等用户重启宿主后 silently 0 工具。
        print("\n写后即验：按写入的配置真实拉起一次握手（initialize + "
              "tools/list）……")
        ok, detail = _verify_entry(entry, cwd=str(workspace))
        print(f"  {'✓ 配置已验证可用' if ok else '✗ 握手验证失败'}：{detail}")
        if not ok:
            print("  配置已写入，但宿主里可能起不来。排查：")
            print(f"    python -m hippocampus_mcp.cli doctor --host {host} "
                  f"--project {workspace}")
            return 1

    print("\n下一步：")
    # 模型不在本地时，先把话说明白：模型是首启才下的，下载失败会让宿主里
    # 「一个工具都没有」，届时人已经在会话里了，最难排查。
    missing_model = False
    try:
        from .engine.embeddings import _cache_roots, _default_libs_path, local_model_dir
        missing_model = local_model_dir(model, _cache_roots(_default_libs_path())) is None
    except Exception:                                        # noqa: BLE001
        missing_model = False
    step = 0
    if missing_model:
        step += 1
        print(f"  {step}) 先预取模型（本机还没有 {model}，首启下载失败会让宿主"
              f"里一个工具都没有）：")
        print(f"     hippocampus-mcp models --fetch {model}")
    step += 1
    print(f"  {step}) 重启该宿主（MCP 配置在启动时读取，改完不重启不生效）")
    step += 1
    print(f"  {step}) 新会话里先调 library_whoami 确认落在预期库上，别静默串库")
    step += 1
    print(f"  {step}) 有疑问跑 hippocampus-mcp doctor")
    return 0


# --------------------------------------------------------------------------
# 子命令：sync
# --------------------------------------------------------------------------
def cmd_sync(args: argparse.Namespace) -> int:
    """发现所有宿主配置，把同一条目同步进去（"配一次、同步到各家"）。

    与 setup 的分工：``setup --host`` 认识的 4 家直写不扫描；sync 面向
    **其余所有**宿主 —— 不维护宿主清单，靠 host_configs 的「文件名约定
    + 形态嗅探」找配置。默认只预览（列候选 + 判定），--yes 才写。

    写入纪律：每文件独立继承取值（不同宿主指不同库是合法状态，不抹平）；
    只动自己那条；写入前重新解析、坏文件当场停手跳过；条目与既有完全
    一致则跳过（幂等，不产生冗余备份）；写完对**去重后的每份 entry**
    各握手一次（逐文件继承可能产出不同 entry，只验一份会漏）。
    """
    workspace = Path(args.workspace).expanduser().resolve()
    candidates = hc.discover_configs(workspace, entry_name=args.name)
    action = {"mine": "更新", "new": "新增", "suspect": "新增(低置信)",
              "broken": "停手", "skip": "跳过"}
    targets = [c for c in candidates if c.kind in ("mine", "new", "suspect")]

    print(f"sync 预览（工作区 {workspace}，"
          f"{'--yes 将写入' if args.yes else '未写盘'}）")
    n_skipped = 0
    for c in candidates:
        if c.kind == "skip":            # 形态不符的只计数，不刷屏
            n_skipped += 1
            continue
        note = c.reason or {"mine": "已有我方条目",
                            "new": "有其他 server 条目"}.get(c.kind, "")
        print(f"  [{action[c.kind]}] {c.origin:<7} {c.path}"
              f"{'  ' + note if note else ''}")
    if n_skipped:
        print(f"  （另有 {n_skipped} 个文件因形态不符未列出）")
    if not targets:
        print("  没有发现可同步的宿主配置 —— 已装的宿主用 setup --host，")
        print("  或把配置文件放到约定路径（*mcp*.json 等）后再跑。")
        return 0
    if not args.yes:
        print("\n预览模式，未写盘。确认清单无误后加 --yes 执行写入。")
        return 0

    written: list[tuple[Path, Path | None]] = []
    distinct: dict[str, dict] = {}
    print()
    for c in targets:
        cfg, err = hc.load_json_config(c.path)
        if cfg is None:
            # 预览到写入之间文件可能变坏：再解析失败 = 停手跳过，绝不覆盖。
            print(f"  ⚠ 停手 {c.path}：{err}（坏文件不写，请手工修好）")
            continue
        key_path = c.key_path or ("mcpServers",)
        container = _dig(cfg, key_path, create=True)
        old = container.get(args.name)
        old = old if isinstance(old, dict) else {}
        old_env = old.get("env") if isinstance(old.get("env"), dict) else {}
        library, _ = _pick_value(args.library,
                                 old_env.get("HIPPOCAMPUS_PERSIST_DIR"),
                                 str(Path(LIBRARY_DEFAULT).expanduser()))
        model, _ = _pick_value(args.model,
                               old_env.get("HIPPOCAMPUS_EMB_MODEL"),
                               MODEL_DEFAULT)
        profile, _ = _pick_value(args.profile,
                                 old_env.get("HIPPOCAMPUS_TOOL_PROFILE"),
                                 PROFILE_DEFAULT)
        python = args.python or old.get("command") or sys.executable
        offline = args.offline or old_env.get("HF_HUB_OFFLINE") == "1"
        src_dir, _pkg = package_src_dir()
        entry = build_entry(python=python, src_dir=src_dir, library=library,
                            model=model, profile=profile, offline=offline)
        if container.get(args.name) == entry:
            print(f"  = 已是最新 {c.path}")
            continue
        container[args.name] = entry
        c.path.parent.mkdir(parents=True, exist_ok=True)
        bak = _backup(c.path)
        c.path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        print(f"  ✓ 已写入 {c.path}" + (f"（备份 {bak}）" if bak else ""))
        written.append((c.path, bak))
        distinct[json.dumps(entry, ensure_ascii=False, sort_keys=True)] = entry

    if not written:
        print("\n没有需要写入的变更。")
        return 0

    if args.no_verify:
        print("\n[跳过写后即验]（--no-verify）：可用性未经验证，"
              "重启宿主后请调 library_whoami 确认")
        return 0

    print("\n写后即验：对去重后的每份 entry 真实拉起一次握手……")
    failed = False
    for entry in distinct.values():
        ok, detail = _verify_entry(entry, cwd=str(workspace))
        print(f"  {'✓' if ok else '✗'} {detail}")
        failed = failed or not ok
    if failed:
        print("\n握手验证失败 —— 已写入的文件与备份（可用备份回退）：")
        for path, bak in written:
            print(f"  {path}" + (f"（备份 {bak}）" if bak else ""))
        return 1

    print("\n下一步：重启各宿主（MCP 配置在启动时读取），新会话里调 "
          "library_whoami 确认没串库。")
    return 0


# --------------------------------------------------------------------------
# 子命令：doctor
# --------------------------------------------------------------------------
def _mark(ok: bool, warn: bool = False) -> str:
    return "[FAIL]" if not ok else ("[WARN]" if warn else "[ OK ]")


def _entry_verdict(cpath: Path, spec: dict) -> str:
    """单份配置的结论行：不存在 / 存在但没条目 / 存在且库指向哪。"""
    if not cpath.exists():
        return "不存在（该处未接入）"
    try:
        cfg = _read_config(cpath)
    except SystemExit:
        return f"存在但**不是合法 JSON**（修好它之前宿主读不到任何 server）—— {cpath}"
    ent = (_dig(cfg, spec["key_path"]) or {}).get(SERVER_NAME_DEFAULT)
    if not isinstance(ent, dict):
        return f"存在，但没有 {SERVER_NAME_DEFAULT} 条目（未接入）"
    env = ent.get("env") or {}
    lib = env.get("HIPPOCAMPUS_PERSIST_DIR", "")
    prof = env.get("HIPPOCAMPUS_TOOL_PROFILE", "") or "common(默认)"
    return f"存在，库={lib or '(未设 → 回落默认)'}  剖面={prof}"


def _doctor_host_project(host: str, project: Path) -> None:
    """宿主×目录的**结论式**判定（``doctor --host X --project <dir>``）。

    旧矩阵只「列出现有的」，缺「缺了谁」「该读哪份」——接入事故的典型形态
    （把 ZCode 的配置错写成 ``.mcp.json``）在旧矩阵里看起来一片绿。这里按
    「宿主会读什么 → 存不存在 → 库指向哪」给结论，并把目录里**存在但该宿主
    不读**的配置文件点名出来。
    """
    spec = _HOSTS[host]
    print(f"\n宿主×目录判定：{spec['label']} @ {project}")
    for scope, cpath in (("user", user_config_path(host)),
                         ("project", project_config_path(host, project))):
        if cpath is None:
            continue
        print(f"  会读 {scope} 级 : {cpath}")
        print(f"                → {_entry_verdict(cpath, spec)}")
    for other, ospec in _HOSTS.items():
        if other == host:
            continue
        opath = project_config_path(other, project)
        if opath is not None and opath.exists():
            print(f"  ⚠ 存在但 {spec['label']} 不读 : {opath}"
                  f"（{ospec['label']} 的项目级配置）")


def _doctor_workspace_binding(project: Path) -> None:
    """D3：工作区绑定判定（``doctor --project`` 的结论式输出）。

    该项目目录在用户级任务目录里登记了吗？命中 = 首个会话 HELLO 即绑到
    该库（并给出读侧档位）；未登记 = 首用会自曝"未绑定"，并给出 onboarding
    的一条命令形态。library/task_directory 都是轻导入（无 torch），
    任何失败降级为一句"查不了"，绝不炸体检。
    """
    print("\n工作区绑定判定（C1/D3）：")
    try:
        from .domain.task_directory import (normalize_entries,
                                            resolve_user_access,
                                            resolve_workspace)
        from .library import resolve_library, scan_libraries
        from .library.task_directory import load_directory
        try:
            ulib = resolve_library("user", libs=scan_libraries())
        except Exception:
            ulib = None
        if ulib is None:
            print("  用户级库不存在——绑定/任务目录功能未启用"
                  "（不影响单库使用）。")
            return
        data, issue = load_directory(ulib.path)
        if issue:
            print(f"  ⚠ 任务目录不可读（{issue}）：绑定状态查不了，"
                  "且 HELLO 反查会回落声明库。")
            return
        entry, status = resolve_workspace(
            normalize_entries((data or {}).get("entries")), str(project))
        if status == "bound":
            print(f"  已绑定：本项目 → {entry['path']}（任务={entry['task']}，"
                  f"读侧档位={entry.get('access') or 'masked'}）")
        elif status == "ambiguous":
            print("  ⚠ 歧义：多条目声明了同一 workspace_root——HELLO 会回落"
                  "声明库并自曝，请修 directory.json（一个工作区只指一个库）。")
        elif status == "unbound":
            print("  未登记：本项目首次使用会自曝「未绑定」。onboarding 一条路：")
            print("    library_create(name=..., workspace_root=r'%s')"
                  % project)
        else:
            print("  无工作区路径可判（--project 未给或为空）。")
    except Exception as exc:  # noqa: BLE001 —— 体检赠品不炸主流程
        print(f"  （绑定状态查不了：{type(exc).__name__}: {exc}）")


def _doctor_off_registry(project: Path) -> None:
    """表外宿主配置（sync 同源扫描，只读）——doctor 不能只认识 _HOSTS。

    串库事故不只发生在登记在册的那几家：用户随手在某个不认识的宿主里
    配了 hippocampus，库指错了一样静默写错。这里把扫描发现的、_HOSTS
    没登记的配置文件点名出来（含坏文件与低置信推断，跳过的纯噪音不计）。
    """
    print("\n表外发现的宿主配置（_HOSTS 之外的 MCP 配置文件）：")
    known: set[str] = set()
    for host in _HOSTS:
        for cpath in (user_config_path(host),
                      project_config_path(host, project)):
            if cpath is not None:
                known.add(os.path.normcase(str(cpath.resolve())))
    try:
        candidates = hc.discover_configs(project)
    except Exception as exc:  # noqa: BLE001 —— 体检赠品不炸主流程
        print(f"  （扫描失败：{type(exc).__name__}: {exc}）")
        return
    found = False
    for c in candidates:
        if c.kind == "skip" or (os.path.normcase(str(c.path.resolve()))
                                in known):
            continue
        found = True
        if c.kind == "mine":
            env = (c.entry or {}).get("env") or {}
            lib = env.get("HIPPOCAMPUS_PERSIST_DIR", "")
            prof = env.get("HIPPOCAMPUS_TOOL_PROFILE", "") or "common(默认)"
            print(f"  {c.origin} <{c.path}> : 我方条目  库={lib or '(未设 → 回落默认)'}"
                  f"  剖面={prof}")
        elif c.kind == "broken":
            print(f"  {c.origin} <{c.path}> : {c.reason}"
                  f" —— 修好之前宿主读不到、也不会自动写入")
        elif c.kind == "new":
            print(f"  {c.origin} <{c.path}> : 有其他 server 条目"
                  f"（未接入本服务；可跑 sync --yes 写入）")
        else:
            print(f"  {c.origin} <{c.path}> : {c.reason}"
                  f"（未接入本服务；可跑 sync --yes 写入）")
    if not found:
        print("  （没有发现表外配置）")


# 实测反馈 2026-10-31：库（尤其 user 库，含嵌入向量、只增不减）落系统盘会
# 持续挤占 C 盘。这里只体检 + 给手工迁移三步——刻意**不做 relocate 命令**：
# 一次性操作不配常驻命令（搬 GB 级向量还要处理 daemon 文件锁，成本远超收益），
# 迁移场景就说清楚在下面的体检输出里。
LIB_ON_SYSTEM_DISK_WARN_BYTES = 1024 ** 3  # 1 GB


def _dir_size_bytes(path: Path) -> int:
    """目录体积（st_size 求和）。walk / stat 失败按 0 计——体检不许炸。"""
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda e: None):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _system_disk_migration_hint(lib_path: Path, size_bytes: int,
                                system_drive: str) -> str | None:
    """库在系统盘时的体检结论（纯函数，供单测）。

    非系统盘 → None（不叨叨）；系统盘 + 超 1 GB → 迁移三步；
    系统盘但没超 → 一行「暂不必迁」。库身份按清单名判定不按路径，
    所以迁盘零代码改动（改 env + 搬目录 + 重启）。
    """
    try:
        lib_drive = os.path.splitdrive(str(lib_path))[0].upper() + "\\"
    except Exception:  # noqa: BLE001 —— 判不了盘符就当没这回事
        return None
    if not system_drive or lib_drive.upper() != system_drive.upper():
        return None
    gb = size_bytes / 1024 ** 3
    if size_bytes >= LIB_ON_SYSTEM_DISK_WARN_BYTES:
        return (f"⚠ 库在系统盘 {lib_drive}，当前体积 {gb:.1f} GB"
                f"（嵌入向量只增不减，会持续挤占系统盘）。\n"
                f"  迁移三步（没有 relocate 命令，手工迁移即可；库身份按"
                f"清单名判定，不按路径）：\n"
                f"  1) 关掉所有宿主会话（daemon 不能在搬目录时持有库文件）；\n"
                f"  2) 把库目录整体搬到新盘，如 D:\\hippocampus\\user；\n"
                f"  3) 改宿主配置里 HIPPOCAMPUS_PERSIST_DIR 指向新路径，"
                f"开新会话跑 library_whoami 验证。")
    return (f"库在系统盘 {lib_drive}，当前 {gb:.2f} GB（<1 GB，暂不必迁；"
            f"超 1 GB 再走迁盘三步）")


def cmd_doctor(args: argparse.Namespace) -> int:
    import importlib.util

    print("hippocampus-mcp doctor")
    print("-" * 60)

    if getattr(args, "host", None):
        _doctor_host_project(args.host,
                             Path(args.project).expanduser().resolve())
        _doctor_workspace_binding(Path(args.project).expanduser().resolve())
        print("-" * 60)

    py_ok = sys.version_info >= (3, 10)
    print(f"{_mark(py_ok)} Python {sys.version.split()[0]} "
          f"（要求 >=3.10）")

    src_dir, pkg_dir = package_src_dir()
    print(f"{_mark(True)} 包位置   {pkg_dir}")
    print(f"{_mark(True)} PYTHONPATH 需求 "
          f"{'需要：' + str(src_dir) if src_dir else '不需要（已安装到 sys.path）'}")

    for mod, required in (("mcp", True), ("torch", True),
                          ("numpy", True), ("sentence_transformers", False)):
        found = importlib.util.find_spec(mod) is not None
        print(f"{_mark(found, warn=not found and not required)} 依赖 {mod}"
              f"{'' if found else ('（缺失：核心功能不可用）' if required else '（缺失：降级）')}")

    # S7 自曝：shell 里裸跑 doctor 时没有宿主 env，检查的是默认库 —— 不说清楚
    # 会被误读成"宿主配置的库有问题"。模型同理。
    persist_env = os.environ.get("HIPPOCAMPUS_PERSIST_DIR", "").strip()
    library = Path(persist_env or LIBRARY_DEFAULT).expanduser()
    # 库 / 模型"还没有"不算故障：首次启动会自建 / 自动下载。报成 FAIL 会
    # 让人以为装坏了，所以这两项只给 WARN。
    lib_exists = library.exists()
    print(f"{_mark(True, warn=not lib_exists)} 库目录   {library}"
          f"{'' if lib_exists else '（还不存在：首次启动会自动建）'}")
    if not persist_env:
        print("       ⚠ 未设 HIPPOCAMPUS_PERSIST_DIR：本次检查的是**默认库**，"
              "不是任何宿主配置里的库")
    if lib_exists:
        backend = ("sqlite" if (library / "hippocampus.db").exists()
                   else ("json" if (library / "memories.json").exists() else "空"))
        print(f"{_mark(True)} 库后端   {backend}")
        # 实测反馈 2026-10-31：盘符 / 体积体检（relocate 场景在此说清楚）。
        size_b = _dir_size_bytes(library)
        home_drive = (os.path.splitdrive(os.path.expanduser("~"))[0]
                      + "\\")
        hint = _system_disk_migration_hint(library, size_b, home_drive)
        if hint:
            print("       " + hint.replace("\n", "\n       "))

    model_env = os.environ.get("HIPPOCAMPUS_EMB_MODEL", "").strip()
    model = model_env or MODEL_DEFAULT
    cache = os.environ.get("HIPPOCAMPUS_EMB_CACHE", "") or str(
        Path.home() / ".cache" / "huggingface" / "hub")
    slug = "models--" + model.replace("/", "--")
    have_model = (Path(cache) / slug).exists()
    print(f"{_mark(True, warn=not have_model)} 模型     {model}"
          f"{'' if have_model else f'（未见于 {cache}；若是自定义模型目录属正常，首次启动也可能现下载）'}")
    if not model_env:
        print("       ⚠ 未设 HIPPOCAMPUS_EMB_MODEL：本次检查的是默认模型"
              "（实际用哪个以宿主配置 env 为准）")

    profile = os.environ.get("HIPPOCAMPUS_TOOL_PROFILE", "") or "common（默认）"
    print(f"{_mark(True)} 工具剖面 {profile}")

    # 最有价值的一段：**各宿主实际指向哪个库**。串库是本项目最贵的坑
    # （写进错的库不报错），逐个宿主列出来才能一眼看出来。
    print("\n各宿主配置里的库（对照看有没有串库）：")
    found_any = False
    for host, spec in _HOSTS.items():
        candidates = (("user", user_config_path(host)),
                      ("project", project_config_path(host, Path.cwd())))
        for scope, cpath in candidates:
            if cpath is None or not cpath.exists():
                continue
            found_any = True
            try:
                cfg = _read_config(cpath)
            except SystemExit:
                print(f"  {spec['label']}({scope}) <{cpath}> : "
                      f"配置不是合法 JSON，已跳过")
                continue
            cont = _dig(cfg, spec["key_path"])
            ent = (cont or {}).get(SERVER_NAME_DEFAULT)
            if not isinstance(ent, dict):
                print(f"  {spec['label']}({scope}) <{cpath}> : "
                      f"无 {SERVER_NAME_DEFAULT} 条目（未接入）")
                continue
            env = ent.get("env") or {}
            lib = env.get("HIPPOCAMPUS_PERSIST_DIR", "")
            prof = env.get("HIPPOCAMPUS_TOOL_PROFILE", "") or "common(默认)"
            # 每行带配置文件路径：看出来"串库/没接上"之后，下一步永远是
            # "改哪个文件" —— 路径就在眼前，不用再猜。
            print(f"  {spec['label']}({scope}) <{cpath}> : "
                  f"库={lib or '(未设 → 回落默认)'}  剖面={prof}")
    if not found_any:
        print("  （没找到任何宿主配置文件 —— 先跑 setup --host <宿主>）")

    # 表外扫描与上面的矩阵互补：矩阵答「认识的宿主读到什么」，这段答
    # 「不认识的宿主里有没有躺着一份配置」——串库不分认识不认识。
    _doctor_off_registry(Path.cwd())

    print("-" * 60)
    print("提示：本命令只做静态检查，不加载模型（所以很快）。")
    print("      真实可用性以宿主里调 library_whoami 的结果为准。")
    return 0


# --------------------------------------------------------------------------
# 子命令：models
# --------------------------------------------------------------------------
def cmd_models(args: argparse.Namespace) -> int:
    """列出本地嵌入模型缓存，并按需回收冗余。

    **默认什么都不删**：不带 ``--prune`` 只看清单；带 ``--prune`` 只出计划；
    真正删盘必须再加 ``--yes``。三道闸是有意的 —— 这个命令唯一会做的事就是
    删文件，而删错的代价是「某个库下次启动加载不到模型」，离线环境下没法补救。
    """
    from .config import load_config
    from .engine import model_cache as MC
    from .engine.embeddings import (
        _cache_roots,
        _default_libs_path,
        default_model_for_system,
        local_model_dir,
    )
    from .engine.onnx_embedder import backend_kind
    from .library import scan_libraries

    roots = ([os.path.abspath(os.path.expanduser(args.cache))]
             if args.cache else _cache_roots(_default_libs_path()))
    models = MC.scan_models(roots)

    # 引用关系：库清单里的 embedder_model 是**唯一**可信来源（空串=没钉死）
    refs: dict[str, list[str]] = {}
    for info in scan_libraries():
        mid = (info.embedder_model or "").strip()
        if mid:
            refs.setdefault(mid, []).append(info.name or os.path.basename(info.path))

    default_id, default_why = default_model_for_system()
    protected = {default_id}
    cfg = load_config()
    if cfg.emb_model.strip():
        protected.add(cfg.emb_model.strip())

    if args.fetch:
        # 安装期把模型先下好。
        # 为什么必须单独有这一步：模型是**首启才下载**的，一旦断网/被墙，
        # MCP 服务直接起不来，而在宿主里表现为「一个工具都没有」——用户和
        # agent 都看不出根因是模型没下来。把这件事前移到安装期，并且失败时
        # 把三条出路讲清楚。
        want = args.fetch if isinstance(args.fetch, str) else default_id
        ready = None
        try:
            ready = local_model_dir(want, roots)
        except Exception:                                    # noqa: BLE001
            ready = None
        if ready:
            print(f"[已就绪] {want} 已在本地：{ready}")
            return 0

        # setdefault 的返回值就是生效值：不必再读一次环境变量
        # （根层豁免清单钉着 cli.py 的 env 读取数，见 config.py 的表格）。
        hf_endpoint = os.environ.setdefault(
            "HF_ENDPOINT", "https://hf-mirror.com")
        cache = os.path.abspath(os.path.expanduser(
            args.cache or (roots[0] if roots else
                           os.path.join(_default_libs_path(), "models"))))
        print(f"预取模型   : {want}")
        print(f"缓存根目录 : {cache}（走 HF 镜像 {hf_endpoint}）")
        print("首次下载约 90MB（英文基线）～400MB（中文引擎），请耐心等待……")
        try:
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer(want, cache_folder=cache)
            # 真跑一次编码：只验「文件在位」不够，权重/分词器坏了要到
            # 第一次检索才炸，那时人已经在会话里了。
            vec = st.encode(["warmup"], show_progress_bar=False)
            dim = int(len(vec[0]))
        except Exception as e:                               # noqa: BLE001
            print(f"\n[失败] {want} 下载或加载失败：{type(e).__name__}: {e}")
            print("三条出路（按成本从低到高）：")
            print(f"  1) 换镜像重试：设 HF_ENDPOINT=https://hf-mirror.com 后重跑"
                  f" `hippocampus-mcp models --fetch {want}`")
            print(f"  2) 手动放：把模型目录放到 {cache}{os.sep}<模型名>/ 再重跑本命令校验")
            print("  3) 换英文基线（88MB，中文检索会明显变差）："
                  "`hippocampus-mcp models --fetch all-MiniLM-L6-v2`，"
                  "并在 setup 时加 --model all-MiniLM-L6-v2")
            return 1
        print(f"[完成] {want} 已就绪（{dim} 维，缓存于 {cache}）")
        return 0

    if args.json:
        print(json.dumps({
            "ok": True,
            "cache_roots": roots,
            "default_model": default_id,
            "default_reason": default_why,
            "backend": backend_kind(),
            "models": [
                {
                    "model_id": m.model_id,
                    "path": m.path,
                    "total_bytes": m.total_bytes,
                    "weight_bytes": m.weight_bytes,
                    "onnx_bytes": m.onnx_bytes,
                    "int8_bytes": m.int8_bytes,
                    "other_bytes": m.other_bytes,
                    "libraries": sorted(refs.get(m.model_id) or []),
                    "is_default": m.model_id in protected,
                }
                for m in models
            ],
        }, ensure_ascii=False, indent=2))
        return 0

    print(MC.format_report(models, in_use=refs, protected=protected,
                           cache_roots=roots))
    print(f"当前默认模型 : {default_id}（{default_why}）")
    print(f"推理后端     : {backend_kind()}")

    if not (args.prune or args.drop):
        print("\n提示：加 --prune 看能回收多少（仍不删），再加 --yes 才真删。")
        return 0

    keep_onnx = backend_kind() != "torch"
    plan = MC.plan_prune(models, in_use=refs, protected=protected,
                         keep_onnx=keep_onnx)

    if args.drop:
        by_id = {m.model_id: m for m in models}
        picked: list[MC.PruneAction] = []
        for mid in [d.strip() for d in args.drop if d.strip()]:
            m = by_id.get(mid)
            if m is None:
                print(f"[跳过] {mid}：缓存里没有这个模型")
                continue
            if refs.get(mid):
                print(f"[拒绝] {mid}：被库 {', '.join(sorted(refs[mid]))} 引用，"
                      f"删了那个库下次启动加载不到模型")
                continue
            if mid in protected:
                print(f"[拒绝] {mid}：是当前默认模型，删了下次启动要联网重下")
                continue
            picked.append(MC.PruneAction(
                kind=MC.KIND_MODEL, model_id=mid, path=m.path,
                bytes_=m.total_bytes, reason="由 --drop 显式点名"))
        # 保留 int8 那类「无论点名与否都可回收」的动作
        plan.actions = picked + [a for a in plan.actions
                                 if a.kind != MC.KIND_MODEL]

    print()
    print(MC.format_plan(plan))
    if not plan.actions:
        return 0

    if not args.yes:
        print(f"\n[dry-run] 以上 {MC.human_size(plan.total_bytes)} 尚未删除。"
              f"确认后加 --yes 执行。")
        return 0

    freed, errors = MC.apply_prune(plan.actions, roots)
    print(f"\n已释放 {MC.human_size(freed)}")
    for e in errors:
        print(f"  [FAIL] {e}")
    if errors:
        return 1
    print("提示：下一次启动会重新统计缓存；若还想回到 ONNX 快路径，"
          "跑 scripts/export_onnx.py 重生成 onnx/。")
    return 0


# --------------------------------------------------------------------------
# 子命令：serve
# --------------------------------------------------------------------------
def cmd_serve(args: argparse.Namespace) -> int:
    """启动服务。

    默认走 **launcher**：多宿主 / 多会话共享单份 daemon，嵌入模型只装一份。
    ``--direct`` 才退回直连 server（等价旧行为，每个会话各装一份模型）。
    """
    if args.direct:
        from .server import main as _server_main
        return _server_main() or 0
    from .launcher import main as _launcher_main
    return _launcher_main([])


# --------------------------------------------------------------------------
# 子命令：context
# --------------------------------------------------------------------------
def cmd_context(args: argparse.Namespace) -> int:
    """hook 注入预览：按一条 query 走 hook_client 全链路。

    与 UserPromptSubmit hook 共用 :func:`hook_client.fetch_context`，用于
    手工验证与排查（daemon 活着才有效；本命令**不拉起** daemon）。
    env 缺省口径与 launcher 相同：key 三要素读 HIPPOCAMPUS_DAEMON_LIBRARY /
    HIPPOCAMPUS_EMB_MODEL / HIPPOCAMPUS_DAEMON_VERSION，HELLO 声明库读
    HIPPOCAMPUS_PERSIST_DIR。
    """
    from .hook_client import fetch_context
    trace: list[str] = []
    text = fetch_context(
        args.query or "", str(Path(args.cwd).resolve()),
        top_k=args.top_k, token_budget=args.budget,
        item_max_chars=args.item_chars,
        key_library=os.environ.get("HIPPOCAMPUS_DAEMON_LIBRARY", ""),
        emb_model=args.model or os.environ.get("HIPPOCAMPUS_EMB_MODEL",
                                               MODEL_DEFAULT),
        daemon_version=os.environ.get("HIPPOCAMPUS_DAEMON_VERSION", ""),
        hello_library=os.environ.get("HIPPOCAMPUS_PERSIST_DIR", ""),
        trace=trace)
    if not text:
        print("（无注入内容）", file=sys.stderr)
        for line in trace:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print(text)
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hippocampus-mcp",
        description="HippoMCP 记忆服务：启动 / 安装到宿主 / 体检。")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("serve", help="启动 MCP 服务（默认经 launcher 共享 daemon）")
    sp.add_argument("--direct", action="store_true",
                    help="退回直连 server（每个会话各装一份模型，不推荐）")
    sp.set_defaults(func=cmd_serve)

    su = sub.add_parser("setup",
                        help="为一个宿主生成 / 合并 MCP 配置（写前自动备份，写后即验）")
    su.add_argument("--host", default=None, choices=sorted(_HOSTS),
                    help="目标宿主；不填则按工作区特征自动判定"
                         "（有 .zcode/ → zcode，有 .mcp.json → workbuddy）")
    su.add_argument("--scope", choices=("user", "project"), default=None,
                    help="用户级 / 工作区级；默认按宿主惯例")
    su.add_argument("--workspace", default=".",
                    help="工作区路径（project 级配置写在这里），默认当前目录")
    su.add_argument("--file", default="",
                    help="直接指定配置文件路径（覆盖 --scope 推导）")
    su.add_argument("--library", default=None,
                    help="记忆库目录；不填则沿用既有配置，全新安装用 "
                         f"{LIBRARY_DEFAULT}")
    su.add_argument("--model", default=None,
                    help="嵌入模型；不填则沿用既有配置，全新安装用 "
                         f"{MODEL_DEFAULT}（中文语料用它）")
    su.add_argument("--profile", default=None,
                    help="工具剖面；不填则沿用既有配置，全新安装用 "
                         f"{PROFILE_DEFAULT}")
    su.add_argument("--python", default="",
                    help="解释器绝对路径，默认取当前 sys.executable")
    su.add_argument("--name", default=SERVER_NAME_DEFAULT,
                    help=f"配置里的 server 条目名，默认 {SERVER_NAME_DEFAULT}")
    su.add_argument("--offline", action="store_true",
                    help="加 HF_HUB_OFFLINE=1（模型已下载好时用它；"
                         "首次安装不要加）")
    su.add_argument("--dry-run", action="store_true",
                    help="只预览将要写入的内容，不写盘")
    su.add_argument("--no-verify", action="store_true",
                    help="跳过写后即验（默认写完会真实握手一次验证可用）")
    su.set_defaults(func=cmd_setup)

    sy = sub.add_parser(
        "sync",
        help="发现所有宿主配置并同步同一条目（默认预览，--yes 写入）")
    sy.add_argument("--workspace", default=".",
                    help="工作区路径（project 级配置从这里发现），默认当前目录")
    sy.add_argument("--library", default=None,
                    help="记忆库目录；不填则逐文件沿用既有配置，全新安装用 "
                         f"{LIBRARY_DEFAULT}")
    sy.add_argument("--model", default=None,
                    help="嵌入模型；不填则逐文件沿用既有配置，全新安装用 "
                         f"{MODEL_DEFAULT}（中文语料用它）")
    sy.add_argument("--profile", default=None,
                    help="工具剖面；不填则逐文件沿用既有配置，全新安装用 "
                         f"{PROFILE_DEFAULT}")
    sy.add_argument("--python", default="",
                    help="解释器绝对路径，默认取当前 sys.executable")
    sy.add_argument("--name", default=SERVER_NAME_DEFAULT,
                    help=f"配置里的 server 条目名，默认 {SERVER_NAME_DEFAULT}")
    sy.add_argument("--offline", action="store_true",
                    help="加 HF_HUB_OFFLINE=1（模型已下载好时用它；"
                         "首次安装不要加）")
    sy.add_argument("--yes", action="store_true",
                    help="真正写入（默认只预览，一个字节都不动）")
    sy.add_argument("--no-verify", action="store_true",
                    help="跳过写后即验（默认写完会真实握手验证可用）")
    sy.set_defaults(func=cmd_sync)

    sd = sub.add_parser(
        "doctor",
        help="体检：路径 / 库（含盘符体积与迁盘提示）/ 模型 / 依赖 / 宿主×目录判定")
    sd.add_argument("--host", default=None, choices=sorted(_HOSTS),
                    help="给出后追加「宿主×目录」结论式判定："
                         "该目录会被此宿主读取的配置在哪、存在与否、缺了谁")
    sd.add_argument("--project", default=".",
                    help="宿主×目录判定的目标工作区（默认当前目录）")
    sd.set_defaults(func=cmd_doctor)

    sm = sub.add_parser(
        "models", help="本地模型缓存：占用明细 / 谁在用 / 回收冗余（默认不删）")
    sm.add_argument("--prune", action="store_true",
                    help="列出可回收项（仍不删，需再加 --yes）")
    sm.add_argument("--yes", action="store_true",
                    help="真正执行删除（必须与 --prune 或 --drop 同用）")
    sm.add_argument("--drop", action="append", default=[], metavar="MODEL",
                    help="显式指定要整体删除的模型 id，如 all-MiniLM-L6-v2"
                         "（被库引用的会被拒绝）")
    sm.add_argument("--cache", default="",
                    help="显式指定模型缓存根目录（默认自动探测）")
    sm.add_argument("--fetch", nargs="?", const=True, default=None, metavar="MODEL",
                    help="预取嵌入模型到本地缓存（不填模型名＝系统默认模型）。"
                         "安装期跑一次：模型是首启才下的，下载失败会让 MCP "
                         "在宿主里表现为「一个工具都没有」")
    sm.add_argument("--json", action="store_true", help="以 JSON 输出，便于脚本消费")
    sm.set_defaults(func=cmd_models)

    cx = sub.add_parser(
        "context",
        help="hook 注入预览：按一条 query 走 hook_client 全链路（不拉起 daemon）")
    cx.add_argument("query", nargs="?", default="",
                    help="模拟的用户提示词（用作检索 query）")
    cx.add_argument("--cwd", default=".",
                    help="工作区根目录（C1 反查定库依据，默认当前目录）")
    cx.add_argument("--top-k", type=int, default=3, help="命中条数（默认 3）")
    cx.add_argument("--budget", type=int, default=600,
                    help="注入字符预算（默认 600）")
    cx.add_argument("--item-chars", type=int, default=150,
                    help="单行截断长度（默认 150，0 = 不截）")
    cx.add_argument("--model", default="",
                    help="daemon key 的模型要素（默认 env HIPPOCAMPUS_EMB_MODEL "
                         "或内置默认）")
    cx.set_defaults(func=cmd_context)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        # 无子命令 = 启动服务（与旧 entry point 行为一致，但现在走 launcher）
        return cmd_serve(argparse.Namespace(direct=False))
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
