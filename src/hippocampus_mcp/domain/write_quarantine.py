# -*- coding: utf-8 -*-
"""写侧隔离（D2a：user 库受管写通道的待审队列判据与审计事件构造）。

拍板①（〔评审报告 20261028〕P1-1）的纯函数层：非家工作区对 user 库的
managed-write **降级为待审**——写入带 quarantine 标记、不进召回池、
不进 ``user_scope`` 槽，drift_audit 报告过目后显式转正或删除。

为什么要这个模块：``write_boundary.classify_write_permission`` 对目标库名
``user`` 一律放 ``MANAGED_WRITE``（U4 的受管通道豁免，本意是给 U3 提升流
用的），但 ``store_memory(library="user")`` 裸写走的是**同一条豁免**——
不受信仓库的注入者可以直写一条伪造"用户偏好"，经 ``user_scope`` 槽注入
此后所有工作区的会话。本模块把"豁免通道上的直写"与"受管提升"区分开：
**豁免保留，直写降级待审**。

全部纯函数：不读 env、不做 I/O、不 import 本包其它模块——
判据三要素（目标名 / 是否家库 / 逃生门）由调用方（组合根工厂）算好传入。
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

__all__ = [
    "USER_LIBRARY_NAME",
    "ACCESS_LOG_FILENAME",
    "should_quarantine",
    "sensitivity_label",
    "build_access_event",
]

#: 用户级库约定名（与 write_boundary 的 user 角色豁免同名同源——
#: 拍板口径：豁免的**通道**不变，豁免通道上的**直写**降级待审）。
USER_LIBRARY_NAME = "user"

#: 写侧审计事件的 sidecar 文件名（住 user 库目录下，与 directory.json
#: 同级同款先例；JSONL 追加，一行一事件）。
ACCESS_LOG_FILENAME = "access_log.jsonl"


def should_quarantine(
    *, target_name: str, same_as_home: bool, escape_open: bool
) -> bool:
    """待审判据（单一真相，组合根工厂与测试共用）。

    True 当且仅当：目标是 user 库（角色豁免通道）**且**目标不是家库
    **且**逃生门未开。家库 = user 库的日常助手宿主（拍板③的 full 档）
    写自己家，不隔离；逃生门（``HIPPOCAMPUS_CROSS_LIBRARY_WRITE=1``）
    是显式运维窗口，迁移批量操作不该进待审队列。
    """
    return (
        (target_name or "").strip() == USER_LIBRARY_NAME
        and not same_as_home
        and not escape_open
    )


def sensitivity_label(
    classification: dict[str, Any], *, personal_default: bool = True
) -> str:
    """把 D1 打标结果映射成卡片上的 ``sensitivity`` 字符串标签。

    有命中类别 → ``sensitive:类别1+类别2``（``+`` 连接，**内部排序**：
    标签与传入顺序无关，同一内容永远得到同一标签）；
    无命中 → ``personal``（宁枉勿纵：user 库自动入库未命中任何类别
    默认 personal，public 只留给显式声明——〔评审〕P2-3 采纳）。
    ``personal_default=False`` 给"显式声明 public"的将来通道留口，
    当前调用方不传。
    """
    cats = sorted(classification.get("categories") or [])
    if cats:
        return "sensitive:" + "+".join(cats)
    return "personal" if personal_default else ""


def build_access_event(
    *,
    kind: str,
    tool: str,
    home: str,
    target_name: str,
    target_path: str,
    quarantined: bool,
    sensitivity: str = "",
    content: str = "",
    ts: float | None = None,
) -> dict[str, Any]:
    """构造一条写侧审计事件（access_log.jsonl 的一行）。

    **绝不落原文**：内容只留 16 位指纹（sha256 前 16 hex）——审计要的是
    "谁在什么时候对 user 库动了笔、隔离与否"，明文回看走记忆卡本身。
    ``kind``：``store_memory`` / ``ingest`` / ``ingest_batch`` /
    ``quarantine_clear``。
    """
    digest = hashlib.sha256((content or "").encode("utf-8")).hexdigest()[:16]
    return {
        "ts": float(time.time() if ts is None else ts),
        "kind": kind,
        "tool": tool or "",
        "home": home or "",
        "target": target_name or "",
        "target_path": target_path or "",
        "quarantined": bool(quarantined),
        "sensitivity": sensitivity or "",
        "content_sha16": digest,
    }
