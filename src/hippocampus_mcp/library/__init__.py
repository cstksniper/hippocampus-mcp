# -*- coding: utf-8 -*-
"""库清单与发现层（**只读**：不改任何现有数据文件）。

这一层存在的理由：多库的第一件事不是"切"，而是"**先知道有哪些库**"。
在库名能被解析之前，``library_use("写作")`` 这类调用只能靠猜，而猜错就是
把记忆写进别人的库 —— 那是比丢数据更难发现的事故（两边都无报错）。

三条设计前提（都来自实测，别凭直觉改）：

1. **路径必须归一**：实测同一工作区会给出 ``d:\\`` 与 ``D:\\`` 两种盘符，
   不归一就会把同一个库当成两个，进而"建了两个 brain 指向一份存储"。
   全模块一律用 :func:`normpath`。
2. **备份只读**：``.bak-*`` / ``.snap-*`` / ``corrupt`` 目录**必须**标 ``backup``
   且不参与名字解析 —— 否则 ``library_use("写作")`` 可能选中一次历史快照，
   之后的写入全落在快照里，真库一条没动。
3. **老库懒补、不写盘**：存量库没有 ``library.json``，只在**内存里**
   按目录名补出名字（去掉 ``hippo_`` 前缀），**不落任何文件** —— 本层的
   硬约束就是「不改任何现有数据文件」。真正写清单只在 ``library_create``
   建新库时发生。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

# 构造注入缝：旋钮登记在 config（library_roots），生产路径由 engine
# import 时灌入；解析序（env 先、注入值回退）见 common/config_bridge。
from ..common.config_bridge import apply_config, resolve as _resolve

#: 库清单文件名（放在库目录内）
MANIFEST_NAME = "library.json"

# 判定「这个目录是不是一个记忆库」：命中任一层文件即可。
# 用层文件而不是"有没有子目录"，是因为空库也该算库（零库首启会建它）。
_LAYER_MARKERS = (
    "memories.json",
    "brain_meta.json",
    "brain_raw.json",
    MANIFEST_NAME,
    # ⚠️ 注意：较新的库可能只有 hippocampus.db（sqlite 后端），从没写过
    # json 层。漏了它 ⇒ 纯 sqlite 库进不了库清单 ⇒ library_use / 工作区引导文件
    # 对它全部解析失败（实测三种写法全抛
    # LibraryResolutionError；宿主 PERSIST_DIR 直通不过清单，所以一直没暴露）。
    "hippocampus.db",
)

# 备份 / 快照 / 损坏隔离目录：**只读**，不参与自动选择
_BACKUP_PREFIXES = (".bak", ".snap", ".corrupt")
_BACKUP_TOKENS = ("corrupt",)

# 老库懒补时要剥掉的目录名前缀（data/hippo_写作 → 写作）
_NAME_PREFIXES = ("hippo_",)


class LibraryResolutionError(ValueError):
    """库寻址失败。

    **必须显式抛出，绝不静默回落到默认库** —— 多库正确性的硬规则。
    "找不到你指定的库"和"用默认库"是两件完全不同的事，混在一起就是串库。
    """


def normpath(p: str) -> str:
    """路径归一：大小写 + 绝对化（实测盘符大小写会不一致）。"""
    return os.path.normcase(os.path.abspath(p))


def library_id_for(path: str) -> str:
    """库 id：由**归一后的绝对路径**派生。

    用路径而不是名字派生，保证 id 稳定：库改名、或被复制成
    ``.bak`` 之后 id 仍同源，两处同 id 可被识别为 ``duplicate_of``。
    """
    return "lib_" + hashlib.sha1(normpath(path).encode("utf-8")).hexdigest()[:12]


def _looks_like_library(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for marker in _LAYER_MARKERS:
        if os.path.exists(os.path.join(path, marker)):
            return True
    return False


def _is_backup(dirname: str) -> bool:
    """备份/快照判定。

    ⚠️ **不能只查前缀**：实际备份目录名是 ``hippo_写作.bak-<时间戳>`` ——
    ``.bak`` 在**中间**而不是开头（备份脚本用 `<原名>.bak-<时间戳>` 命名）。
    只判 ``startswith(".bak")`` 会把这类备份全判成真库，直接违反
    「备份不参与自动选择」。故改判"包含"。
    """
    low = os.path.basename(dirname.rstrip("\\/")).lower()
    for tok in (*_BACKUP_PREFIXES, *_BACKUP_TOKENS):
        if tok in low:
            return True
    return low.startswith("bak-") or low.startswith("snap-")


def _guess_name(path: str) -> str:
    """老库懒补：目录名去 ``hippo_`` 前缀、去前导点。"""
    base = os.path.basename(path.rstrip("\\/")) or path
    low = base.lower()
    for pref in _NAME_PREFIXES:
        if low.startswith(pref):
            base = base[len(pref):]
            break
    return base.lstrip(".") or base or "unnamed"


def display_name(path: str) -> str:
    """库的展示名：library.json 的 name → 目录名懒补。

    读不到清单时按 ``_guess_name`` 的规则从 path 推——权威身份
    始终在 session_start 返回体的 ``library`` 键里，这里只是卡片上的
    一行标签，推不出来时用目录名不丢人。

    与 ``_info_for(path).name`` 同口径但**刻意不复用**：那边先
    ``normpath`` 再推名，本函数按调用方给的原始 path 推（brain 的
    persist_dir 已是归一绝对路径，两种输入下等价，但边缘形态——
    如未归一路径——的行为以本实现为准）。
    """
    try:
        fp = os.path.join(path or "", MANIFEST_NAME)
        with open(fp, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        name = str((data or {}).get("name") or "").strip()
        if name:
            return name
    except Exception:  # noqa: BLE001 —— 缺文件/坏文件都按"没有"处理
        pass
    return _guess_name(path or "") or "unnamed"


@dataclass
class LibraryInfo:
    """一个被发现的记忆库。

    Attributes:
        path: 归一后的绝对路径（**唯一标识**）。
        library_id: ``lib_<sha1 前 12 位>``，由 path 派生 ⇒ 稳定。
        name: 人类可读名字（清单优先，否则懒补）。
        aliases: 别名列表，参与名字解析。
        backup: True = 备份/快照/损坏目录，**只读且不参与自动选择**。
        explicit: True = 目录里有 ``library.json``（False = 懒补，内存态）。
    """

    path: str
    library_id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    created_at: str = ""
    embedder_model: str = ""
    backup: bool = False
    explicit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "library_id": self.library_id,
            "name": self.name,
            "aliases": list(self.aliases),
            "path": self.path,
            "created_at": self.created_at,
            "embedder_model": self.embedder_model,
            "backup": self.backup,
            "explicit": self.explicit,
        }


def _read_manifest(path: str) -> dict[str, Any]:
    """读 ``library.json``。缺失/损坏**都不是错误**（老库本就没有）。"""
    fp = os.path.join(path, MANIFEST_NAME)
    try:
        with open(fp, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001 —— 缺文件是常态，坏文件按"没有"处理
        return {}
    return data if isinstance(data, dict) else {}


def _info_for(path: str) -> LibraryInfo:
    path = normpath(path)
    manifest = _read_manifest(path)
    base = os.path.basename(path.rstrip("\\/"))
    name = str(manifest.get("name") or "").strip() or _guess_name(path)
    aliases = manifest.get("aliases")
    aliases = [str(a).strip() for a in aliases if str(a).strip()] if isinstance(aliases, list) else []
    return LibraryInfo(
        path=path,
        library_id=str(manifest.get("library_id") or "").strip() or library_id_for(path),
        name=name,
        aliases=aliases,
        created_at=str(manifest.get("created_at") or ""),
        embedder_model=str(manifest.get("embedder_model") or ""),
        backup=_is_backup(base),
        explicit=bool(manifest),
    )


# ----------------------------------------------------------------------
# 扫描
# ----------------------------------------------------------------------
def _repo_root() -> str:
    """仓库根 = ``src/hippocampus_mcp/library`` 的上三级。

    ⚠️ 三级 dirname 与本模块在 ``src/hippocampus_mcp/library/`` 包内的
    深度一一对应；模块若再挪层级必须同步加减，否则 ``<仓库>/data`` 会
    算成 ``src/data`` 而静默少扫一个库根。
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))


def default_roots() -> list[str]:
    """默认扫描根：env ``HIPPOCAMPUS_LIBRARY_ROOTS``（``os.pathsep`` 分隔）>
    ``~/.hippocampus_mcp`` + ``<仓库>/data``。

    两个根都要扫：前者是 MCP 默认 persist_dir（老部署就在那），
    后者是本项目把库按目录铺开的地方。
    """
    env = _resolve("HIPPOCAMPUS_LIBRARY_ROOTS", "library_roots", "").strip()
    if env:
        return [x.strip() for x in env.split(os.pathsep) if x.strip()]
    roots = [
        os.path.join(os.path.expanduser("~"), ".hippocampus_mcp"),
        os.path.join(_repo_root(), "data"),
    ]
    return roots


def scan_libraries(roots: list[str] | None = None) -> list[LibraryInfo]:
    """扫描目录发现库。**只读**，不创建也不修改任何文件。

    每个 root 先自查是不是库（老部署把库直接放在 ``~/.hippocampus_mcp``），
    再扫它的**直接**子目录（不递归：库不会层层嵌套）。
    """
    found: dict[str, LibraryInfo] = {}
    for root in (roots if roots is not None else default_roots()):
        root = normpath(os.path.expanduser(root))
        if not os.path.isdir(root):
            continue
        candidates: list[str] = []
        if _looks_like_library(root):
            candidates.append(root)
        try:
            for entry in os.listdir(root):
                sub = os.path.join(root, entry)
                if os.path.isdir(sub):
                    candidates.append(sub)
        except OSError:  # noqa: BLE001 —— 无权读的目录跳过，不该让整个扫描失败
            continue
        for cand in candidates:
            if not _looks_like_library(cand):
                continue
            key = normpath(cand)
            if key not in found:
                found[key] = _info_for(key)
    return sorted(found.values(), key=lambda i: (i.backup, i.name))


# ----------------------------------------------------------------------
# 寻址：三态解析（id → 名字/别名 → 路径）
# ----------------------------------------------------------------------
def _match_by_id(query: str, libs: list[LibraryInfo]) -> LibraryInfo | None:
    q = query.strip().lower()
    for info in libs:
        if info.library_id.lower() == q:
            return info
    return None


def _match_by_name(query: str, libs: list[LibraryInfo]) -> tuple[LibraryInfo | None, list[LibraryInfo]]:
    """按名字/别名精确匹配。**歧义不猜** —— 返回候选让调用方显式报错。"""
    q = query.strip()
    ql = q.lower()
    hits = [i for i in libs if not i.backup and (i.name.lower() == ql or ql in [a.lower() for a in i.aliases])]
    if len(hits) == 1:
        return hits[0], hits
    if len(hits) > 1:
        return None, hits
    return None, []


def _match_by_path(query: str, libs: list[LibraryInfo]) -> LibraryInfo | None:
    key = normpath(os.path.expanduser(query))
    for info in libs:
        if info.path == key:
            return info
    # 路径存在且像个库，但不在扫描结果里（例如扫描根没覆盖它）→ 就地补一个
    if _looks_like_library(key):
        return _info_for(key)
    return None


def resolve_library(
    query: str | None,
    roots: list[str] | None = None,
    libs: list[LibraryInfo] | None = None,
    default_path: str | None = None,
) -> LibraryInfo | None:
    """三态解析：``lib_`` id → 名字/别名 → 路径。

    Args:
        query: 调用方给的库标识。**留空返回 None**（＝"没指定"，由调用方决定
            是否走粘性/默认库）—— "没指定"与"指定了一个不存在的"必须分开处理。
        default_path: 默认库路径。它即便不在扫描结果里也必须可解析（老部署
            的 persist_dir 常常不在任何扫描根下）。

    Raises:
        LibraryResolutionError: 指定了但解析不明（含重名歧义）。
            **绝不静默回落**。
    """
    if query is None or not str(query).strip():
        return None
    q = str(query).strip()
    pool = list(libs) if libs is not None else scan_libraries(roots)
    if default_path:
        dkey = normpath(default_path)
        if not any(i.path == dkey for i in pool) and _looks_like_library(dkey):
            pool.append(_info_for(dkey))

    if q.lower().startswith("lib_"):
        hit = _match_by_id(q, pool)
        if hit is not None:
            return hit
        raise LibraryResolutionError(
            f"找不到 id 为 {q} 的记忆库。用 library_list 看看有哪些库。"
        )

    hit, ambiguous = _match_by_name(q, pool)
    if hit is not None:
        return hit

    # 名字只命中**备份**目录时必须把话说清楚。说"找不到"是错的：库确实在，
    # 只是只读。运维看到"找不到"会以为路径配错了，转头去改配置 —— 白折腾。
    ql = q.lower()
    bak_hits = [
        i for i in pool if i.backup
        and (i.name.lower() == ql or ql in [a.lower() for a in i.aliases])
    ]
    if bak_hits:
        raise LibraryResolutionError(
            f"{q} 是备份/快照目录，只读，不能作为写入目标（共 {len(bak_hits)} 处："
            f"{'、'.join(i.path for i in bak_hits[:3])}）。请用真库名或 library_id 指定。"
        )

    if ambiguous:
        names = "、".join(f"{i.name}({i.library_id})" for i in ambiguous)
        raise LibraryResolutionError(
            f"库名 {q} 有歧义，匹配到 {len(ambiguous)} 个库：{names}。"
            f"请用 library_id（lib_…）或完整路径指定。"
        )

    hit = _match_by_path(q, pool)
    if hit is not None:
        return hit

    # 走到这里＝"指定了但解析不明"，必须报错。给出近似名建议（编辑距离 ≤2）。
    suggestions = _suggest(q, pool)
    tail = f"你是不是想找：{suggestions}？" if suggestions else "用 library_list 看看有哪些库。"
    raise LibraryResolutionError(f"找不到记忆库 {q}。{tail}")


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _suggest(query: str, libs: list[LibraryInfo], max_dist: int = 2) -> str:
    ql = query.strip().lower()
    out = []
    for info in libs:
        if info.backup:
            continue
        cands = [info.name.lower()] + [a.lower() for a in info.aliases]
        if any(_levenshtein(ql, c) <= max_dist for c in cands):
            out.append(f"{info.name}({info.library_id})")
    return "、".join(out[:3])


# ----------------------------------------------------------------------
# 建库：扫描路径不调用，留给 library_create
# ----------------------------------------------------------------------
def write_manifest(path: str, name: str, aliases: list[str] | None = None,
                   embedder_model: str = "") -> LibraryInfo:
    """新建库清单。**只在 library_create 时调用**，扫描路径绝不写盘。"""
    path = normpath(path)
    os.makedirs(path, exist_ok=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload = {
        "library_id": library_id_for(path),
        "name": name,
        "aliases": list(aliases or []),
        "created_at": now,
        "embedder_model": embedder_model,
    }
    with open(os.path.join(path, MANIFEST_NAME), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return _info_for(path)
