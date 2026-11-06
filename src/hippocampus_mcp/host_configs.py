"""宿主 MCP 配置的**通用**发现与分类（sync / doctor 共用引擎）。

**为什么要有这个模块**：MCP 协议只规定传输与 JSON-RPC，没规定配置文件
放哪、顶层键叫什么。逐宿主硬编码路径是一张永远追不完的表（本项目的
``cli._HOSTS`` 只登记了 4 家）；但各家的文件名其实是**社区公共约定**
（``mcp.json`` / ``mcp_config.json`` / ``mcp_settings.json`` /
``claude_desktop_config.json``…），顶层键位总共就几种形态。所以这里
不维护「宿主 → 路径」表，改维护「**文件名约定 + 形态嗅探**」：

1. **有界扫描**：家目录 dot-dir（深度≤3）、平台配置根一层子目录、
   工作区 dot-dir（深度≤2）——只在这些角落里找约定文件名。
2. **形态嗅探**：JSON 能解析 + 顶层能找到「server 条目容器」
   （``mcpServers`` / ``mcp.servers`` / ``servers``，兜底「值是含
   command/url 的 dict」的通用形态）才算数。
3. **绝不新建**：只对**已存在**的文件做候选，新建文件走 ``setup
   --file``——sync 写错地方的代价是污染宿主配置，宁缺勿滥。

安全纪律与 setup 同源：坏 JSON 停手不写、只动自己那条、写前必备份、
写后即验。分类结论（mine/new/suspect/broken/skip）由 ``classify``
给出，写入决策在 ``cli.cmd_sync``。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

#: 我方条目在宿主配置里的默认名字（与 cli.SERVER_NAME_DEFAULT 同值；
#: 这里不能反向 import cli —— cli 依赖本模块，反向会成环）。
ENTRY_NAME_DEFAULT = "hippocampus"

#: 文件名约定的两类：任何位置都认的（*mcp*.json / claude 专用名），
#: 和只在 dot-dir 下一层才认的（settings.json 太通用，必须限定目录）。
CONVENTION_GLOBS = ("*mcp*.json", "claude_desktop_config.json")
DOTDIR_ONLY_GLOBS = ("settings.json",)

#: 「仅凭文件名推断」的精确名单（低置信写入用）。发现用的 glob 故意宽
#: （*mcp*.json 能接住不知道的宿主），但**没有结构证据**就动手写时必须
#: 收紧 —— 否则 mcp-tool-list.json 这类同名缓存文件会被误写。
EXACT_NAME_SUSPECTS = ("mcp.json", "mcp_config.json", "mcp_settings.json",
                       "claude_desktop_config.json")

#: 键位探测顺序：空容器也算命中（新装宿主常见 {"mcpServers": {}}）。
_KEY_PROBES: tuple[tuple[str, ...], ...] = (
    ("mcpServers",), ("mcp", "servers"), ("servers",), ("mcp",),
)


def load_json_config(path: Path) -> tuple[dict | None, str]:
    """读一份 JSON 宿主配置。返回 ``(配置, "")`` 或 ``(None, 失败原因)``。

    坏文件**返回错误而不抛异常**——发现式扫描要逐个试文件，一个坏的
    不该打断整轮；需要「停手不写」语义的调用方（setup）自己升级成
    SystemExit。文件不存在视为空配置（与 setup 既有口径一致）。
    """
    if not path.exists():
        return {}, ""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"unreadable ({type(exc).__name__}: {exc})"
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"not valid JSON ({exc})"
    if not isinstance(cfg, dict):
        return None, "top level is not a JSON object"
    return cfg, ""


def _looks_like_server_entry(value: object) -> bool:
    """一个 dict 值长得像 server 条目吗（command=本地 / url=远端）。"""
    return (isinstance(value, dict)
            and (isinstance(value.get("command"), str)
                 or isinstance(value.get("url"), str)))


def _container_shape(node: object) -> bool:
    """dict 是否长得像「server 条目容器」：空（新装）或装着条目。"""
    if not isinstance(node, dict):
        return False
    if not node:
        return True
    return any(_looks_like_server_entry(v) for v in node.values())


def find_container(cfg: dict) -> tuple[str, ...] | None:
    """在配置里找到 server 条目容器的键路径；找不到返回 None。

    先按已知键位探测（``_KEY_PROBES`` 顺序即优先级），**两遍取值**：
    先取装着条目的键位、都空才取空容器——否则 ``{"mcp": {}}`` 这种
    空壳会抢在 ``mcp.servers`` 之前被命中。已知键位全空再兜底扫
    「顶层 / 二层里装着 server 条目的 dict」——兜底是为了不认识的
    宿主改了键名也能接上。
    """
    probed: list[tuple[bool, tuple[str, ...]]] = []
    for key_path in _KEY_PROBES:
        node = _dig(cfg, key_path)
        if node is not None:
            probed.append(
                (any(_looks_like_server_entry(v) for v in node.values()),
                 key_path))
    for want_entries in (True, False):
        for has_entries, key_path in probed:
            if has_entries == want_entries:
                return key_path
    for key, value in cfg.items():                    # 兜底：顶层
        if key not in {p[0] for p in _KEY_PROBES} and _container_shape(value):
            return (key,)
    for key, value in cfg.items():                    # 兜底：二层
        if not isinstance(value, dict):
            continue
        for key2, value2 in value.items():
            if _container_shape(value2):
                return (key, key2)
    return None


def _dig(cfg: dict, key_path: tuple[str, ...]) -> dict | None:
    node: object = cfg
    for key in key_path:
        node = node.get(key) if isinstance(node, dict) else None
    return node if isinstance(node, dict) else None


def _convention_name(path: Path) -> bool:
    """文件名是否值得「无结构证据也动手写」——精确名单，收紧口径。

    发现用的 glob（``CONVENTION_GLOBS``）故意宽；这里决定的是写入
    侧的低置信判定，宽了会把 mcp-tool-list.json 之类同名缓存写脏。
    settings.json 仍只在 dot-dir 下一层才认（太通用）。
    """
    name = path.name.lower()
    if name == "settings.json":
        return path.parent.name.startswith(".")
    return name in EXACT_NAME_SUSPECTS


@dataclass
class Candidate:
    """一个候选配置文件的判定结果。"""
    path: Path
    origin: str                        # user / project（来自哪次扫描）
    kind: str                          # mine / new / suspect / broken / skip
    key_path: tuple[str, ...] | None   # 容器键路径（写入位置）
    reason: str = ""
    entry: dict | None = None          # 我方既有条目（kind=mine 时）


def classify(path: Path, origin: str,
             entry_name: str = ENTRY_NAME_DEFAULT) -> Candidate:
    """按「能解析 + 有容器 + 有没有我方条目」给一份配置下结论。

    - ``mine``：已有我方条目 —— sync 的更新目标；
    - ``new``：有**别的** server 条目 —— 活的宿主配置，新增需确认；
    - ``suspect``：容器为空或仅靠文件名推断 —— 新增需确认（低置信）；
    - ``broken``：文件名像 MCP 配置但解析失败 —— 点名，绝不写；
    - ``skip``：既无形又无名 —— 不提。
    """
    cfg, err = load_json_config(path)
    if cfg is None:
        if _convention_name(path):
            return Candidate(path, origin, "broken", None, err)
        return Candidate(path, origin, "skip", None, err)
    key_path = find_container(cfg)
    container = _dig(cfg, key_path) if key_path else None
    ours = container.get(entry_name) if container else None
    if isinstance(ours, dict):
        return Candidate(path, origin, "mine", key_path, entry=ours)
    if key_path is not None:
        if any(_looks_like_server_entry(v) for v in container.values()):
            return Candidate(path, origin, "new", key_path,
                             reason="other server entries exist")
        return Candidate(path, origin, "suspect", key_path,
                         reason="container is empty")
    if _convention_name(path):
        return Candidate(path, origin, "suspect", None,
                         reason="no MCP container yet (identified by filename convention)")
    return Candidate(path, origin, "skip", key_path,
                     reason="unrecognized MCP config shape")


def _walk_names(root: Path, globs: tuple[str, ...], max_depth: int):
    """在 root 下按约定文件名找文件，深度 ≤ max_depth（只报真实文件）。"""
    for depth in range(max_depth):          # 中间层数 0..max_depth-1
        prefix = "*/" * depth
        for pattern in globs:
            for hit in root.glob(prefix + pattern):
                if hit.is_file():
                    yield hit


def _platform_roots() -> list[Path]:
    """平台配置根（一层子目录里找约定名）。

    Windows 的 ``%APPDATA%`` 与 macOS 的 ``~/Library/Application Support``
    各家桌面宿主扎堆；两层子目录只探 VS Code 扩展（globalStorage）与
    「<宿主>/User/」这两个公认层级，不再往下。
    """
    roots: list[Path] = []
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        roots.append(Path(appdata))
    lib = Path.home() / "Library" / "Application Support"
    if lib.is_dir():
        roots.append(lib)
    return roots


def discover_configs(workspace: Path, *, entry_name: str = ENTRY_NAME_DEFAULT,
                     home: Path | None = None,
                     platform_roots: list[Path] | None = None) -> list[Candidate]:
    """有界扫描出所有候选宿主配置并逐个分类。

    ``home`` / ``platform_roots`` 是**测试注入口**：不传用真值（生产
    路径），测试传 tmp 路径 / 空列表即可与本机真实配置完全隔离。
    """
    home = Path(home) if home is not None else Path.home()
    found: dict[str, tuple[Path, str]] = {}   # 规范化路径 → (原路径, origin)

    def record(path: Path, origin: str) -> None:
        key = os.path.normcase(str(path.resolve()))
        found.setdefault(key, (path, origin))

    # 家目录 dot-dir：*mcp* 深度≤3；settings.json / claude 名深度≤2
    if home.is_dir():
        for child in sorted(home.iterdir()):
            if not (child.is_dir() and child.name.startswith(".")):
                continue
            for hit in _walk_names(child, CONVENTION_GLOBS, 3):
                record(hit, "user")
            for hit in _walk_names(child, DOTDIR_ONLY_GLOBS, 2):
                record(hit, "user")

    # 平台配置根：一层子目录 + VS Code 扩展 + <宿主>/User/ 两个公认层级
    for root in (platform_roots if platform_roots is not None
                 else _platform_roots()):
        if not root.is_dir():
            continue
        for pattern in ("*/" + g for g in CONVENTION_GLOBS):
            for hit in root.glob(pattern):
                if hit.is_file():
                    record(hit, "user")
        for pattern in ("*/User/" + g for g in CONVENTION_GLOBS):
            for hit in root.glob(pattern):
                if hit.is_file():
                    record(hit, "user")
        for hit in root.glob("Code/User/globalStorage/*/*mcp*.json"):
            if hit.is_file():
                record(hit, "user")

    # 工作区：根上只认 .mcp.json；dot-dir 深度≤2（含 config.json ——
    # 这个名字在家目录里太吵，只在工作区 dot-dir 里认）
    ws = Path(workspace)
    if ws.is_dir():
        direct = ws / ".mcp.json"
        if direct.is_file():
            record(direct, "project")
        for child in sorted(ws.iterdir()):
            if not (child.is_dir() and child.name.startswith(".")):
                continue
            for hit in _walk_names(child, CONVENTION_GLOBS + ("config.json",), 2):
                record(hit, "project")

    return [classify(p, o, entry_name) for p, o in found.values()]
