# -*- coding: utf-8 -*-
"""用户级任务目录：**纯数据**的归一/选择/渲染。

一句话：用户层除了「内容」（偏好/方法论）还有「导航」——``任务 → 库``
的映射是跨项目、用户拥有、每个任务的 agent 都要用的知识。用户说「参考下 A
任务」时，查目录即得 A 库的确切地址，**不猜、不扫盘、不依赖扫描根**。

本模块只做纯函数，与 :mod:`domain.user_scope` / :mod:`domain.scope_recall`
同口径：

* sidecar 的读写（``<user 库>/directory.json``）在
  :mod:`library.task_directory`（L1 infra，有文件 I/O）；
* 组装与注入在 ``brain/session.py::_task_directory_brief``；
* domain **不 import library / store**（架构约束：domain 只准依赖 common），
  ``path`` 在这里只是字符串——归一比对是读写侧的事。

四条口径（设计 §3.2/§3.3/§3.6/§7，逐条落在这里）：

1. **不做向量记忆、不进召回池**：目录要精确命中，语义检索会漏检，漏检＝
   寻址失败，正面撞「绝不静默」红线 ⇒ 这里是确定性选择/排序，不是相似度。
2. **``path`` 必填**（绝对路径）：``lib_`` id 匹配走扫描池，库不在扫描根下
   时 id 解析会失败，而路径寻址可「就地补」；``library_id`` 只作人读与将来
   去重，**不作寻址主键**。
3. **``limit <= 0`` 是「关」不是切片**：``entries[:limit]`` 在负数上是
   「倒数第 N 条之外全要」，与「关＝零痕迹」不是一回事（U1/D4 同款教训）。
4. **不猜**：目录里没有 A 就报「目录里没有 A」——本模块只提供精确查找
   :func:`find_task`，模糊匹配/自动认领**刻意不做**。
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "DIRECTORY_SCHEMA",
    "VALID_ACCESS",
    "find_task",
    "normalize_entries",
    "normalize_entry",
    "render_task_directory",
    "resolve_user_access",
    "resolve_workspace",
    "select_task_directory",
]

#: sidecar schema 版本（§3.3）。升级时按「读旧写新」处理：旧字段一律可缺省。
DIRECTORY_SCHEMA = 1

#: 条目字段的固定投影（键序敏感：JSON 落盘的字段顺序由这里决定）。
_ENTRY_KEYS = ("task", "name", "library_id", "path", "note", "updated_at",
               "workspace_root", "access")

#: D2b 读侧 access 三档（设计稿 §4.2）：full（全量可见）／masked（默认，
#: sensitive:* 排除）／none（user 库整体不可见）。空串 = 未声明 → 按
#: masked 走（登记工作区的默认档）；unregistered（目录无此工作区）→
#: none（fail-closed，Q-8 拍板倾向）。
VALID_ACCESS = ("full", "masked", "none")

#: 渲染时 ``note`` 的截断长度：目录是**索引**不是正文（§3.4），备注只作
#: 「这条任务是干什么的」的一句话提示，与 ``block_task_hits`` 的 200 字同精神。
_NOTE_CAP = 40


def normalize_entry(raw: Any) -> dict[str, Any] | None:
    """把一个原始条目归一成固定投影；**缺 ``task`` 或 ``path`` 返回 None**。

    ``path`` 必填是硬约束（§3.3）：id 会在库迁址/换部署后解析不到，路径
    可以「就地补」——目录存在的意义是**精确可达**，宁可丢一条坏条目（调用
    方统计 ``dropped`` 自曝），也不能把不可达的条目当有效索引。
    """
    if not isinstance(raw, dict):
        return None
    task = str(raw.get("task") or "").strip()
    path = str(raw.get("path") or "").strip()
    if not task or not path:
        return None
    try:
        updated = float(raw.get("updated_at") or 0.0)
    except (TypeError, ValueError):
        updated = 0.0
    return {
        "task": task,
        # name 缺省回落 task：``library_create`` 建库即登记时两者同名，
        # 手工补条目的用户也不必写两遍。
        "name": str(raw.get("name") or "").strip() or task,
        "library_id": str(raw.get("library_id") or "").strip(),
        "path": path,
        "note": str(raw.get("note") or "").strip(),
        "updated_at": updated,
        # C1：本条目绑定的工作区根（可选）。非空时 HELLO 反查可把该工作区
        # 的连接默认库定为 ``path``——「绑库而非装机」的解析依据。
        # 空串 = 未绑定（老条目缺省，行为与 U5 逐位一致）。
        "workspace_root": str(raw.get("workspace_root") or "").strip(),
        # D2b：读侧 access 档（可选）。非法值一律归空串（= masked）——
        # 写错的 access 不能变成提权（fail-closed 方向：宁可误降不误升）。
        "access": (str(raw.get("access") or "").strip()
                   if str(raw.get("access") or "").strip() in VALID_ACCESS
                   else ""),
    }


def normalize_entries(raw_entries: Any) -> list[dict[str, Any]]:
    """批量归一（丢弃坏条目；非列表一律空列表——坏 sidecar 不在这里炸）。"""
    if not isinstance(raw_entries, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in raw_entries:
        entry = normalize_entry(raw)
        if entry is not None:
            out.append(entry)
    return out


def select_task_directory(entries: list[Any], *, limit: int) -> list[dict[str, Any]]:
    """选出要注入的条目：**最近更新的在前**，cap 到 ``limit``。

    Args:
        entries: 条目（原始 dict 或已归一的都可以，内部再归一一次）。
        limit: 带几条。**``<= 0`` 返回空列表**——这是"关闭"，不是切片
            （负数切片在这里是错的，同 U1/D4）。

    排序：``(-updated_at, task.casefold())`` 降序/升序组合。第二键是**确定性
    兜底**：同一批 ``library_create`` 常在同一秒落地，只按时间排会让注入顺序
    随 dict 遍历顺序漂移，测试与人工复核都不可复现。
    """
    if limit <= 0:
        return []
    pool = normalize_entries(entries)
    pool.sort(key=lambda e: (-e["updated_at"], e["task"].casefold()))
    return pool[:limit]


def render_task_directory(entries: list[dict[str, Any]]) -> list[str]:
    """渲染成紧凑索引行：``任务 → 库名（路径）``（有备注再缀一句）。

    与 ``block_*`` 一族的纪律一致：一行一条、不换行、可直接拼 prompt；
    路径原样给出（**不给相对化/缩写**——寻址要的是能直接喂给 ``library_use``
    或 ``library=`` 的绝对路径，缩写了就得再猜一次）。
    """
    lines: list[str] = []
    for e in entries:
        line = f"{e['task']} → {e['name']}（{e['path']}）"
        note = str(e.get("note") or "")
        if note:
            line += f" — {note[:_NOTE_CAP]}"
        lines.append(line)
    return lines


def find_task(entries: list[Any], task: str) -> dict[str, Any] | None:
    """按任务名精确查（``casefold`` 归一）；没有返回 None。

    **不做模糊匹配**（§7）：猜错库的代价是读写到别人的库，用户明确要求
    「目录里没有 A 就说没有」。
    """
    q = str(task or "").strip().casefold()
    if not q:
        return None
    for entry in normalize_entries(entries):
        if entry["task"].casefold() == q:
            return entry
    return None


def resolve_workspace(entries: list[Any], cwd: str) -> tuple[dict[str, Any] | None, str]:
    """按连接工作区路径反查家库（C1：``workspace_root`` 精确匹配）。

    Args:
        entries: 目录条目（原始或已归一均可，内部再归一一次）。
        cwd: 连接工作区路径（launcher HELLO 的 ``cwd``）。

    Returns:
        ``(entry | None, status)``。status 四值：
        ``bound``（恰好一条命中，entry 为该条目）／``unbound``（无命中）／
        ``ambiguous``（两条及以上命中——**不取第一个**，报错态留给调用方
        自曝，歧义的家库比没有家库更危险）／``no_cwd``（cwd 空，无从反查）。

    匹配是 ``os.path.normcase`` 归一后的**全等**（v1 定案：不做前缀匹配，
    monorepo 子目录场景留观察期后决定——设计稿 §3.2）。domain 只管判定；
    目录从哪读、命中后怎么接管连接默认库，是读写侧（server）的事。
    """
    q = os.path.normcase(str(cwd or "").strip())
    if not q:
        return None, "no_cwd"
    hits = [e for e in normalize_entries(entries)
            if e["workspace_root"] and os.path.normcase(e["workspace_root"]) == q]
    if not hits:
        return None, "unbound"
    if len(hits) > 1:
        return None, "ambiguous"
    return hits[0], "bound"


def resolve_user_access(
    *, home_is_user: bool, binding_status: str, entry_access: str,
    escape_open: bool,
) -> str:
    """D2b 读侧档位判定（单一真相，组合根与测试共用）。

    Args:
        home_is_user: 本连接家库是否就是 user 库（**只在无工作区声明形态下
            作为 full 依据**——2026-10-31 收窄，见下）。
        binding_status: C1 绑定状态（bound/unbound/ambiguous/no_cwd/
            no_user_library/directory_unreadable/lookup_error/undeclared）。
        entry_access: 命中条目的 ``access`` 声明（空 = 未声明）。
        escape_open: 逃生门（HIPPOCAMPUS_CROSS_LIBRARY_WRITE=1）——运维
            窗口读写全开，与写侧隔离同口径。

    Returns:
        ``full`` / ``masked`` / ``none``。

    收窄（2026-10-31 拍板，迁移文档 §六）：旧版首判 ``home_is_user → full``
    在用户级安装下有洞——unbound 工作区回落声明库（=user 库）会被判"家"
    拿 full 档。新语义（对称：full = 被授权把 user 库当自己家）：

    * ``bound``：条目声明优先——声明 ``full`` 才 full（提权要显式，查户口
      不靠推断），缺省 masked，声明 none 即 none；
    * ``unbound``/``ambiguous``/``no_user_library``/``directory_unreadable``/
      ``lookup_error`` → **none**（fail-closed，Q-8 拍板倾向；回落判"家"的
      洞就此闭合——哪怕家库恰好是 user 库）；
    * ``no_cwd``/``undeclared``（无工作区声明的直连）：家库=user 即
      **full**——这不是回落，是 A1 日常宿主显式单库部署形态（cwd 都没有，
      不存在"陌生项目回落"的攻击面）；其余 masked（D2b 既有裁决：直连
      缺省档，单库部署的 U1 继承槽不能整槽杀掉）。
    """
    if escape_open:
        return "full"
    if binding_status == "bound":
        return entry_access or "masked"
    if binding_status in ("unbound", "ambiguous", "no_user_library",
                          "directory_unreadable", "lookup_error"):
        return "none"
    if home_is_user:
        return "full"
    return "masked"
