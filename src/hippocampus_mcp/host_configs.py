"""**Generic** discovery and classification of host MCP configs (shared engine
of sync / doctor).

**Why this module exists**: the MCP protocol only specifies transport and
JSON-RPC; it does not say where config files live or what the top-level key
is called. Hardcoding paths per host is a table that can never be fully
chased down (this project's ``cli._HOSTS`` registers only 4 hosts); but the
various file names are actually **community conventions**
(``mcp.json`` / ``mcp_config.json`` / ``mcp_settings.json`` /
``claude_desktop_config.json``...), and the top-level key shapes come in
only a few forms. So instead of maintaining a "host -> path" table, we
maintain "**filename conventions + shape sniffing**":

1. **Bounded scan**: home dot-dirs (depth<=3), one level of subdirectories
   under platform config roots, workspace dot-dirs (depth<=2) -- convention
   file names are only sought in these corners.
2. **Shape sniffing**: it counts only if the JSON parses + a "server entry
   container" can be found at the top (``mcpServers`` / ``mcp.servers`` /
   ``servers``, falling back to the generic shape "a value that is a dict
   with command/url").
3. **Never create**: only **existing** files become candidates; creating
   files goes through ``setup --file`` -- sync writing to the wrong place
   costs a polluted host config; better to miss than to misfire.

Safety discipline is shared with setup: stop on bad JSON without writing,
touch only our own entry, always back up before writing, verify after
writing. The classification verdicts (mine/new/suspect/broken/skip) come
from ``classify``; the write decision is in ``cli.cmd_sync``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

#: Default name of our entry in host configs (same value as
#: cli.SERVER_NAME_DEFAULT; this module must not import cli in reverse --
#: cli depends on this module, so reversing would create a cycle).
ENTRY_NAME_DEFAULT = "hippocampus"

#: Two classes of filename convention: recognized anywhere (*mcp*.json /
#: claude's dedicated name), and recognized only one level under a dot-dir
#: (settings.json is too generic; it must be directory-limited).
CONVENTION_GLOBS = ("*mcp*.json", "claude_desktop_config.json")
DOTDIR_ONLY_GLOBS = ("settings.json",)

#: The exact list for "inferred from filename alone" (used for low-confidence
#: writes). Discovery globs are deliberately broad (*mcp*.json catches
#: unknown hosts), but writing **without structural evidence** must tighten
#: -- otherwise same-named cache files like mcp-tool-list.json get wrongly
#: written.
EXACT_NAME_SUSPECTS = ("mcp.json", "mcp_config.json", "mcp_settings.json",
                       "claude_desktop_config.json")

#: Key-probe order: an empty container also counts as a hit (freshly
#: installed hosts commonly have {"mcpServers": {}}).
_KEY_PROBES: tuple[tuple[str, ...], ...] = (
    ("mcpServers",), ("mcp", "servers"), ("servers",), ("mcp",),
)


def load_json_config(path: Path) -> tuple[dict | None, str]:
    """Read one JSON host config. Returns ``(config, "")`` or ``(None, failure reason)``.

    A bad file **returns an error instead of raising** -- discovery-style
    scanning tries files one by one; a single bad one must not interrupt the
    whole round. Callers needing "stop, do not write" semantics (setup)
    escalate to SystemExit themselves. A missing file counts as an empty
    config (consistent with setup's existing convention).
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
    """Does a dict value look like a server entry (command=local / url=remote)?"""
    return (isinstance(value, dict)
            and (isinstance(value.get("command"), str)
                 or isinstance(value.get("url"), str)))


def _container_shape(node: object) -> bool:
    """Does the dict look like a "server entry container": empty (fresh install) or holding entries?"""
    if not isinstance(node, dict):
        return False
    if not node:
        return True
    return any(_looks_like_server_entry(v) for v in node.values())


def find_container(cfg: dict) -> tuple[str, ...] | None:
    """Find the key path of the server entry container in the config; None
    when not found.

    First probe known key positions (the ``_KEY_PROBES`` order is the
    priority), **two passes**: first take the key position holding entries;
    only if all are empty take the empty container -- otherwise an empty
    shell like ``{"mcp": {}}`` would be hit before ``mcp.servers``. If all
    known key positions are empty, fall back to scanning "a dict at the top
    / second level holding server entries" -- the fallback exists so a host
    we do not know that renamed keys can still be caught.
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
    for key, value in cfg.items():                    # fallback: top level
        if key not in {p[0] for p in _KEY_PROBES} and _container_shape(value):
            return (key,)
    for key, value in cfg.items():                    # fallback: second level
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
    """Is the file name worth "writing even without structural evidence" --
    an exact list, tightening the criterion.

    The discovery globs (``CONVENTION_GLOBS``) are deliberately broad; what
    is decided here is the write-side low-confidence verdict -- too broad
    and same-named caches like mcp-tool-list.json get dirtied. settings.json
    is still only recognized one level under a dot-dir (too generic).
    """
    name = path.name.lower()
    if name == "settings.json":
        return path.parent.name.startswith(".")
    return name in EXACT_NAME_SUSPECTS


@dataclass
class Candidate:
    """Verdict for one candidate config file."""
    path: Path
    origin: str                        # user / project (which scan produced it)
    kind: str                          # mine / new / suspect / broken / skip
    key_path: tuple[str, ...] | None   # container key path (write location)
    reason: str = ""
    entry: dict | None = None          # our existing entry (when kind=mine)


def classify(path: Path, origin: str,
             entry_name: str = ENTRY_NAME_DEFAULT) -> Candidate:
    """Give a config a verdict along "parses + has a container + has our entry
    or not".

    - ``mine``: already has our entry -- sync's update target;
    - ``new``: has **other** server entries -- a live host config; adding
      needs confirmation;
    - ``suspect``: container empty, or inferred from the filename alone --
      adding needs confirmation (low confidence);
    - ``broken``: filename looks like an MCP config but parsing failed --
      named explicitly, never written;
    - ``skip``: neither shape nor name matches -- not mentioned.
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
    """Find files by convention file names under root, depth <= max_depth
    (only real files are reported)."""
    for depth in range(max_depth):          # intermediate levels 0..max_depth-1
        prefix = "*/" * depth
        for pattern in globs:
            for hit in root.glob(prefix + pattern):
                if hit.is_file():
                    yield hit


def _platform_roots() -> list[Path]:
    """Platform config roots (convention names sought one level of
    subdirectories below).

    Windows' ``%APPDATA%`` and macOS' ``~/Library/Application Support`` are
    where the desktop hosts cluster; two levels of subdirectories only probe
    the two recognized layers -- VS Code extensions (globalStorage) and
    "<host>/User/" -- and no deeper.
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
    """Bounded scan for all candidate host configs, classifying each one.

    ``home`` / ``platform_roots`` are **test injection points**: omitted,
    real values are used (production path); tests pass tmp paths / empty
    lists to fully isolate from this machine's real configs.
    """
    home = Path(home) if home is not None else Path.home()
    found: dict[str, tuple[Path, str]] = {}   # normalized path -> (original path, origin)

    def record(path: Path, origin: str) -> None:
        key = os.path.normcase(str(path.resolve()))
        found.setdefault(key, (path, origin))

    # Home dot-dirs: *mcp* depth<=3; settings.json / claude name depth<=2
    if home.is_dir():
        for child in sorted(home.iterdir()):
            if not (child.is_dir() and child.name.startswith(".")):
                continue
            for hit in _walk_names(child, CONVENTION_GLOBS, 3):
                record(hit, "user")
            for hit in _walk_names(child, DOTDIR_ONLY_GLOBS, 2):
                record(hit, "user")

    # Platform config roots: one level of subdirs + VS Code extensions + the
    # two recognized layers <host>/User/
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

    # Workspace: at the root only .mcp.json is recognized; dot-dirs depth<=2
    # (including config.json -- that name is too noisy in the home directory,
    # recognized only in workspace dot-dirs)
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
