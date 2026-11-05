# -*- coding: utf-8 -*-
"""源文件签名与漂移审计（从 brain/core.py 提取）。

收录四个 brain 方法（``_resolve_source_path`` / ``_file_sig`` /
``record_source_sig`` / ``source_drift``）与一个 pure 助手
（``_PureMixin._drift_row``）的实现体，brain 侧只留薄包装。

为什么放 store/ 而不是 brain / domain / application：
- 这是**库存记录上的查询组合**（``drift_audit`` 遍历 store 记录算引用
  事实，再逐源对磁盘现状）——同 contradictions 的落点判据；
- 要用 ``store.memory.MemoryState`` 状态机判 active，domain 受依赖
  矩阵约束（只准依赖 common）碰不到它，application 又是 brain 的
  下游（「用例逻辑只从 brain 单向流向 application，不回头」）；
- 文件 I/O（os.stat / open / hashlib）在 store 层有先例（backend）。

**旋钮纪律**（承 tag_ops / ledger）：``source_root`` / ``sig_enabled``
两个 Config 旋钮由包装层在调用时从 brain 侧现取传入——本模块不 import
config（store 受矩阵约束也不许 import config）。唯一直接 import 的常量
是 ``SOURCE_SIG_CACHE_MAX``（``_file_sig`` 的 FIFO 缓存上限，全仓无
monkeypatch 先例，同 tag_ops 的 TAG_OPS_FILE 例外）。

状态语义（与 brain 侧原实现逐字一致）：
- ``file_sig`` 的缓存键含 ``(path, mtime_ns, size)``——文件被改自然
  重算，没改则同进程反复 ingest 只算一次哈希；OSError（文件正被删 /
  无权限）返回 None，调用方按「解不出」处理，不炸写入路径；
- 缓存淘汰是 FIFO（dict 保序）丢最旧，纯性能缓存不追求命中率；
- ``record_source_sig`` 内容事实没变就不动层（连 dict 都不碰）——
  层指纹每次 ingest 都变会让每次 save 都写盘（批量导入 500 条 =
  500 次写盘）；``first_seen`` 从既有条目继承，只有首次见到才落 now；
- ``drift_audit`` 只读：不改 ``source_sigs``、不写文件，磁盘事实
  （mtime/size/sha1）当场现算。
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Iterable

from ..common.constants import SOURCE_SIG_CACHE_MAX, _now
from .memory import MemoryState


def resolve_source_path(source: str, persist_dir: str | None,
                        source_root: str) -> str | None:
    """把 source 标签解析成磁盘上的绝对路径；解不出返回 None。

    解析次序：绝对路径 → 记忆库目录 → ``source_root`` → 进程 CWD。
    **解不出是常态而不是错误**：source 也承载会话 id、自由文本来源
    （"群聊" / "口头"），它们本来就没有对应文件——对它们记一份假签名
    只会把 unbaselined 池搅浑。
    """
    s = str(source or "").strip()
    if not s:
        return None
    candidates = [s] if os.path.isabs(s) else [
        os.path.join(persist_dir or "", s),
        os.path.join(source_root, s) if source_root else s,
        os.path.abspath(s),
    ]
    for c in candidates:
        try:
            if c and os.path.isfile(c):
                return os.path.abspath(c)
        except OSError:
            continue
    return None


def file_sig(path: str, cache: dict[tuple[str, int, int], str],
             cache_max: int = SOURCE_SIG_CACHE_MAX) -> dict[str, Any] | None:
    """算一个文件的 {mtime_ns, size, sha1}；带 (路径, mtime, size) 缓存。

    缓存键含 mtime/size：文件被改（mtime 变）自然重算，没改则同一
    进程内反复 ingest 同一文件只算一次哈希。OSError（文件正在被删 /
    无权限）返回 None——调用方按"解不出"处理，不炸写入路径。

    淘汰是 FIFO（dict 保序）丢最旧：纯性能缓存，不追求命中率最优。
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_mtime_ns, st.st_size)
    cached = cache.get(key)
    if cached is None:
        h = hashlib.sha1()
        try:
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        except OSError:
            return None
        cached = h.hexdigest()
        if len(cache) >= cache_max:
            # FIFO 丢最旧（dict 保序）：纯性能缓存，不追求命中率最优
            cache.pop(next(iter(cache)))
        cache[key] = cached
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size, "sha1": cached}


def record_source_sig(source: str, source_sigs: dict[str, dict[str, Any]],
                      persist_dir: str | None, source_root: str,
                      sig_cache: dict[tuple[str, int, int], str], *,
                      sig_enabled: bool) -> dict[str, Any] | None:
    """写入侧：给一个 source 标签记源文件签名（ingest 调用）。

    **内容事实没变就不动层**：签名条目含 sig_at，若每次 ingest 都重盖，
    层指纹每次都变 ⇒ 每次 save 都写盘（批量导入 500 条 = 500 次写盘）。
    没变时连 dict 都不碰，层保持干净。

    Args:
        source: source 标签。
        source_sigs: brain 的签名层（本函数就地改它）。
        persist_dir / source_root: 路径解析输入（brain 侧现取）。
        sig_cache: ``_file_sig`` 的 sha1 缓存（brain 实例属性，就地改）。
        sig_enabled: Config 旋钮 source_sig（关着直接返回 None）。

    Returns:
        当前生效的签名条目（新记或既有），解不出路径 / 开关关着返回 None。
    """
    if not sig_enabled:
        return None
    s = str(source or "").strip()
    if not s:
        return None
    path = resolve_source_path(s, persist_dir, source_root)
    if not path:
        return None
    sig = file_sig(path, sig_cache)
    if sig is None:
        return None
    prev = source_sigs.get(s)
    if (isinstance(prev, dict)
            and prev.get("path") == path
            and prev.get("mtime_ns") == sig["mtime_ns"]
            and prev.get("size") == sig["size"]
            and prev.get("sha1") == sig["sha1"]):
        return prev  # 没变：不碰层
    now = _now()
    entry = dict(sig)
    entry["path"] = path
    entry["sig_at"] = now
    entry["first_seen"] = (float(prev.get("first_seen") or now)
                           if isinstance(prev, dict) else now)
    source_sigs[s] = entry
    return entry


def drift_row(source: str, category: str,
              ref: dict[str, int] | None,
              sig: dict[str, Any] | None,
              cur: dict[str, Any] | None,
              diff: list[str] | None = None) -> dict[str, Any]:
    """漂移审计的一行：引用事实 + 记录签名 vs 磁盘现状（sha1 只给前 8 位）。"""
    row: dict[str, Any] = {
        "source": source,
        "category": category,
        "memories": (ref or {}).get("memories", 0),
        "active": (ref or {}).get("active", 0),
    }
    if sig:
        row["recorded"] = {
            "sig_at": sig.get("sig_at"),
            "sha1": str(sig.get("sha1") or "")[:8],
            "size": sig.get("size"),
            "path": sig.get("path"),
        }
    if cur:
        row["current"] = {
            "sha1": str(cur.get("sha1") or "")[:8],
            "size": cur.get("size"),
            "mtime_ns": cur.get("mtime_ns"),
        }
    if diff:
        row["diff_fields"] = diff
    return row


def drift_audit(records: Iterable[Any],
                source_sigs: dict[str, dict[str, Any]], *,
                source: str = "", limit: int = 50, include_ok: bool = False,
                persist_dir: str | None = None, source_root: str = "",
                sig_enabled: bool = True,
                sig_cache: dict[tuple[str, int, int], str] | None = None,
                ) -> dict[str, Any]:
    """C1b：源文件漂移审计——"文件改了但记忆没更新"的服务端答案。

    借鉴前作的 verify 四分类，按记忆库的实际形态落地为：

    ============== ======================================================
    分类           含义
    ============== ======================================================
    ``stale``      记过签名、文件还在，但 mtime/size/sha1 有任一不同
                   ⇒ 该源下的记忆可能已过期（文件是较新真相）
    ``missing``    记过签名、文件不见了 ⇒ 记忆还在、源没了（imported_
                   but_deleted）；回查原话的链路已断
    ``unbaselined``能解析到文件、但**从没记过签名**（本功能上线前导入的
                   存量、或 ``HIPPOCAMPUS_SOURCE_SIG=0`` 期间写的）⇒
                   不是"没漂移"，是"无法判断"——单列一类，不混进 ok
    ``orphan``     记过签名、但库里没有任何记忆引用这个 source ⇒ 签名
                   条目本身过期了（记忆被 forget 干净 / 换标签重导）
    ``unresolvable``source 不是文件（会话 id / 自由文本）⇒  informational，
                   不参与漂移判定
    ``ok``         记过签名、文件还在、内容事实一致
    ============== ======================================================

    每条带**分母**（各类计数 + 总源数 + 引用记忆数）：``count:0`` 有两种
    真相（真没漂移 vs 压根没采集），不报分母调用方一律读成前者——
    与 contradictions 的 ``candidate_pool`` 同一条纪律。

    **不做**的事：``never_ingested``（某目录下的文件从没导入过）需要知道
    "监视目录"是什么，那是项目侧脚本的输入（``export_memories(source=)``
    已能算），本工具不假设监视目录。

    Args:
        records: 库存记录可迭代（brain 传 ``store.iter_records()``）。
        source_sigs: 签名层（只读）。
        source: 只审计这一个 source（空 = 全库）。
        limit: 每类最多回多少行（0 = 只给计数）。
        include_ok: 是否把 ``ok`` 的源也列出来（默认只列有问题的）。
        persist_dir / source_root: 路径解析输入（brain 侧现取）。
        sig_enabled: Config 旋钮 source_sig（进 denominators 自曝）。
        sig_cache: sha1 缓存（brain 实例属性，就地改；缺省用本地空缓存）。
    """
    cache = sig_cache if sig_cache is not None else {}
    # 1) 引用事实：source 标签 → {memories, active}（分母来自这里）
    refs: dict[str, dict[str, int]] = {}
    for rec in records:
        s = str(rec.source or "").strip()
        if not s:
            continue
        row = refs.setdefault(s, {"memories": 0, "active": 0})
        row["memories"] += 1
        if rec.state != MemoryState.ARCHIVED.value:
            row["active"] += 1
    # 2) 判定：签名层 ∪ 引用集合（任一侧独有的都要露面）；
    #    点名了 source 就**收窄到它一个**（不是并集加一项——那会让
    #    "只查 a" 返回全库结果，调用方以为过滤生效了）。
    want = source.strip()
    labels = ([want] if want
              else sorted(set(source_sigs) | set(refs)))
    buckets: dict[str, list[dict[str, Any]]] = {
        "stale": [], "missing": [], "unbaselined": [],
        "orphan": [], "unresolvable": [], "ok": [],
    }
    for s in labels:
        sig = source_sigs.get(s)
        ref = refs.get(s)
        if sig is None:
            # 从没记过签名：能解析到文件才是 unbaselined（无法判断），
            # 否则是"这个 source 压根不是文件"（informational）
            if resolve_source_path(s, persist_dir, source_root):
                buckets["unbaselined"].append(drift_row(
                    s, "unbaselined", ref, None, None))
            else:
                buckets["unresolvable"].append(drift_row(
                    s, "unresolvable", ref, None, None))
            continue
        path = resolve_source_path(s, persist_dir, source_root)
        if path is None:
            buckets["missing"].append(drift_row(
                s, "missing", ref, sig, None))
            continue
        cur = file_sig(path, cache)
        if cur is None:
            buckets["missing"].append(drift_row(
                s, "missing", ref, sig, None))
            continue
        if ref is None:
            buckets["orphan"].append(drift_row(s, "orphan", None, sig, cur))
            continue
        diff = [k for k in ("mtime_ns", "size", "sha1")
                if sig.get(k) != cur.get(k)]
        if diff:
            buckets["stale"].append(drift_row(
                s, "stale", ref, sig, cur, diff))
        else:
            buckets["ok"].append(drift_row(s, "ok", ref, sig, cur))
    # 3) 计数 + 截断。orphan 按定义 ref 为 None，其"引用记忆数"是 0。
    counts = {k: len(v) for k, v in buckets.items()}
    out_rows: dict[str, list[dict[str, Any]]] = {}
    for k, rows in buckets.items():
        if k == "ok" and not include_ok:
            continue
        out_rows[k] = rows[:limit] if limit and limit > 0 else rows
    actionable = counts["stale"] + counts["missing"] + counts["unbaselined"]
    return {
        "ok": True,
        "scope": source.strip() or "(全库)",
        "counts": counts,
        # 可行动的（要宿主看一眼的）三类合计——"0"才是"没漂移"，
        # 但 unbaselined>0 时它说的是"无法判断"，必须分开读。
        "actionable": actionable,
        "verdict": ("clean" if actionable == 0 else
                    ("unknown" if counts["unbaselined"] and
                     counts["stale"] == counts["missing"] == 0 else
                     "drifted")),
        "denominators": {
            "sources_total": len(labels),
            "recorded_sigs": len(source_sigs),
            "referenced_sources": len(refs),
            "memories_with_source": sum(r["memories"] for r in refs.values()),
            "sig_enabled": sig_enabled,
        },
        "rows": out_rows,
        "truncated": {k: max(0, len(v) - len(out_rows.get(k, [])))
                      for k, v in buckets.items()
                      if k in out_rows},
        "note": (
            "stale=文件已变、记忆未更新（文件是较新真相）；"
            "missing=文件没了、记忆还在（回查原话链路已断）；"
            "unbaselined=有文件但从没记过签名（无法判断，不是没漂移）；"
            "orphan=签名还在但没有记忆引用它；"
            "unresolvable=source 不是文件（会话 id 等），不参与判定。"
            "never_ingested（某目录从没导入过）不在本工具范围——"
            "那需要监视目录，请用 export_memories(source=...) 项目侧算。"
        ),
    }
