# -*- coding: utf-8 -*-
"""``update_tags`` 的计划/回显/落写三段（从 ``brain.core`` 内联抽取）。

为什么值得搬
============

``update_tags`` 是存量标签回填的**唯一通路**（``update_memory`` 不接受
``tags``），也是真库上一次动过 2,010 条的批量工具。它原来是一整段 140 行的
入口方法，里面四类逻辑缠在一起：

1. **入参归一与校验**（mode 大小写/非法值、tags 与 clear_keys 双空）——
   纯函数，却要构造整个 HippoBrain + 嵌入器才能测；
2. **计划构建**（before/after 逐条算、值未变跳过）——幂等性的全部语义；
3. **回显装配**（payload 十一个键 + sample 投影 + dry_run/confirm 两道
   闸门的提示语）——纯字符串与数字组装；
4. **落写循环**（逐条 ``store.update``、地图条目刷 timestamp、单条失败
   不吞整批）——缠着 L1 存储端口与时钟。

搬到本模块后四段都能毫秒级单测；``brain.core.update_tags`` 只剩薄壳
（派发 + 选择 + 限额 + 记账 + 落盘 + 覆盖率）。

搬动**零行为变化**的保障：
1. 四段代码逐字复制，只把 ``self.`` / 模块级常量换成注入参数；
2. 两个闸门常量（``UPDATE_TAGS_SAMPLE_MAX`` / ``UPDATE_TAGS_CONFIRM_ABOVE``）
   **不 import**，由壳在调用时从 core 命名空间现取后传入——J1 测试
   monkeypatch ``core.UPDATE_TAGS_CONFIRM_ABOVE``，快照进本模块会让闸门
   静默失效（与 ``store.tag_ops_undo`` 同款纪律）；
3. 时钟以**可调用对象**注入（``now=_now``）：没有地图条目时一次都不调，
   传值会让开关失效时照样付代价（且冻不住测试里的时间）；
4. parity 探针（``_probe_out/m5_batch20_parity.py``）用 HEAD 的旧
   ``update_tags``（exec 进 ``brain_core.__dict__`` 本体、挂子类）与新壳
   跑双种子库，15 个场景完整返回体 + 库状态 + 账本文件逐字段比对，
   DRIFT 必须为 0。

依赖方向：本模块**零 import**（连 common 都没用上）——记录对象按鸭子类型
（``.id`` / ``.tags``）使用，存储/时钟/常量全部注入，符合 domain → 仅
common 的分层纪律（这里比纪律还严）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# 计划条目：(记录 id, 改前标签, 改后标签)。before/after 都是**新 dict**
# （``dict(...)`` 拷贝）——账本与 sample 都要长期持有它们。
PlanEntry = tuple[str, dict[str, Any], dict[str, Any]]


def normalize_request(
    mode: str,
    tags: dict[str, Any] | None,
    clear_keys: list[str] | None,
) -> tuple[str, dict[str, Any], list[str], dict[str, Any] | None]:
    """入参归一与校验（原 ``core.update_tags`` 开头三段，逐字保留）。

    Returns:
        ``(mode, new_tags, ckeys, err)``。``err`` 非 None 时壳直接返回它
        （显式报错，绝不静默空跑）；合法时前三项是归一后的值。

    三条保留语义：

    * mode 归一 ``strip().lower()`` 后只认 ``merge`` / ``replace``；
    * ``clear_keys`` 过滤空串键（``["", "章"]`` → ``["章"]``）——空键名
      的 ``pop`` 是无声的 no-op，但会污染回显里的 ``clear_keys``；
    * tags 与 clear_keys 双空 = 「没有要改的东西」，显式报错。
    """
    mode = (mode or "merge").strip().lower()
    if mode not in ("merge", "replace"):
        return mode, {}, [], {
            "ok": False, "error": f"非法 mode: {mode!r}",
            "allowed_modes": ["merge", "replace"]}
    new_tags = dict(tags or {})
    ckeys = [str(k) for k in (clear_keys or []) if str(k)]
    if not new_tags and not ckeys:
        return mode, new_tags, ckeys, {
            "ok": False,
            "error": "tags 与 clear_keys 不能同时为空（没有要改的东西）"}
    return mode, new_tags, ckeys, None


def build_plan(
    recs: list[Any],
    *,
    mode: str,
    new_tags: dict[str, Any],
    ckeys: list[str],
) -> list[PlanEntry]:
    """逐条算 before/after，值未变的不进计划（幂等性的全部语义）。

    **为什么「值未变不计入」是幂等的前提**：原需求以为 merge 语义本身
    能带来幂等——第二次跑每条键都已存在，但**值可能不同**（第一次写
    ``域=F``、宿主中途手改成 ``域=Z``，重跑会再改回 F）。只有整包
    before/after 相等才跳过，重跑才能得到 ``to_change=0``。

    ``replace`` 模式**不看 before**：``after = dict(new_tags)`` 再删
    clear_keys——旧键整体清掉是 replace 的定义。
    """
    plan: list[PlanEntry] = []
    for rec in recs:
        before = dict(rec.tags or {})
        if mode == "replace":
            after = dict(new_tags)
        else:
            after = dict(before)
            after.update(new_tags)
        for k in ckeys:
            after.pop(k, None)
        if after == before:
            continue  # 值未变 → 幂等：重跑时这里不计入
        plan.append((rec.id, before, after))
    return plan


def project_sample(
    plan: list[PlanEntry],
    *,
    new_tags: dict[str, Any],
    ckeys: list[str],
    sample_max: int,
) -> list[dict[str, Any]]:
    """sample 投影：只给前 ``sample_max`` 条的**改前/改后**。

    只投影**本次涉及的键**（``new_tags`` 的键 + ``ckeys``）——一条记忆
    可能带几十个历史标签，全量回显会让返回体爆炸，而调用方要核对的只是
    这次动了什么。
    """
    return [
        {
            "id": rid,
            "before": {k: v for k, v in before.items()
                       if k in new_tags or k in ckeys},
            "after": {k: v for k, v in after.items()
                      if k in new_tags or k in ckeys},
        }
        for rid, before, after in plan[:sample_max]
    ]


def build_payload(
    *,
    dry_run: bool,
    confirm: bool,
    mode: str,
    new_tags: dict[str, Any],
    ckeys: list[str],
    filt: dict[str, Any],
    matched: int,
    considered: int,
    capped: bool,
    to_change: int,
    plan: list[PlanEntry],
    sample_max: int,
    confirm_above: int,
) -> tuple[dict[str, Any], bool]:
    """装配计划/执行回显体，并执行 dry_run 与 confirm 两道闸门。

    Returns:
        ``(payload, proceed)``。``proceed=False`` 时壳直接返回 payload
        （dry_run 计划，或闸门拒绝）；``True`` 时才进入落写循环。

    两道闸门（都是踩过同类坑才加的）：

    * **默认 ``dry_run=True``**：批量改标签没有事务，跑错了不会报错，
      只会在过滤时表现为「查不到东西」；
    * **超过 ``confirm_above`` 条必须 ``confirm=True``**：真库一次动
      2,010 条，过滤条件一个字符写错就是 2,010 条垃圾标签。

    闸门常量由壳调用时从 core 命名空间现取后传入（测试 monkeypatch
    ``core.UPDATE_TAGS_CONFIRM_ABOVE``）——本模块不 import 它们。
    """
    payload: dict[str, Any] = {
        "ok": True,
        "dry_run": bool(dry_run),
        "mode": mode,
        "tags": new_tags,
        "clear_keys": ckeys,
        "filter": filt,
        "matched": matched,
        "considered": considered,
        "capped": capped,
        "to_change": to_change,
        "skipped_unchanged": considered - to_change,
        "sample": project_sample(plan, new_tags=new_tags, ckeys=ckeys,
                                 sample_max=sample_max),
    }
    payload["reverse_hint"] = (
        "可撤销：服务端记了全量前后影像账本，执行后带 op_id；"
        "update_tags(undo=op_id 或 \"last\", dry_run=false) 即 fail-closed "
        "回滚（当前值仍等于 after 才改回 before，被改过的条目会跳过并自曝）"
    )

    if dry_run:
        payload["hint"] = (
            "这是计划，没有写入。确认无误后用 dry_run=false 执行"
            + (f"；影响 {to_change} 条 > {confirm_above}，"
               f"还需同时 confirm=true" if to_change > confirm_above else "")
        )
        return payload, False

    if to_change > confirm_above and not confirm:
        payload.update({
            "ok": False,
            "applied": False,
            "error": (f"影响 {to_change} 条 > {confirm_above}，"
                      f"拒绝执行：请先 dry_run=true 看 sample，确认后加 confirm=true"),
        })
        return payload, False
    return payload, True


@dataclass(frozen=True)
class ApplyPorts:
    """落写循环的全部依赖（brain 侧注入；domain 不 import 任何上层）。

    Attributes:
        store: L1 存储端口。只用到 ``update(rid, **fields)`` 一个面
            （返回真值算成功，假值/抛异常都进 ``failed``）。
        map_name_key: 地图保留标签键（``map.constants.MAP_NAME_KEY``）。
            带这个键的条目改标签 = 维护动作，``timestamp`` 必须跟着刷——
            索引行/闸四按 timestamp 算「N 天前更新」，不刷的话 agent
            每次恪守「改了状态走 update_tags」索引行都照样标 ⚠️。
            住在 map 包，domain 看不着，故注入。
        now: 时钟（brain 的 ``_now``）。以**可调用对象**注入——没有地图
            条目时一次都不调。
    """
    store: Any
    map_name_key: str
    now: Callable[[], float]


@dataclass(frozen=True)
class ApplyResult:
    """一次落写循环的三路产出（执行回显体的原料）。"""
    changed: int = 0
    failed: list[str] = field(default_factory=list)
    applied_changes: list[dict[str, Any]] = field(default_factory=list)


def apply_plan(plan: list[PlanEntry], ports: ApplyPorts) -> ApplyResult:
    """逐条 ``store.update``，单条失败不吞整批。

    三条保留语义：

    * 地图条目（``map_name_key in after``）顺手刷 ``timestamp``——副作用
      是好的：地图条目从此不会被 consolidate 当沉寂记忆归档；
    * ``store.update`` 返回假值 = 该条没落成，进 ``failed``（不抛异常
      也算失败）；
    * 单条抛异常只把 ``"id: ExcType: msg"`` 记进 ``failed`` 继续跑——
      批量工具半途而废比带着错误跑完更难收场（``noqa: BLE001`` 同原处）。
    """
    changed = 0
    failed: list[str] = []
    applied_changes: list[dict[str, Any]] = []
    for rid, before, after in plan:
        try:
            fields = {"tags": dict(after)}
            if ports.map_name_key in after:
                fields["timestamp"] = ports.now()
            if ports.store.update(rid, **fields):
                changed += 1
                applied_changes.append({"id": rid, "before": before,
                                        "after": after})
            else:
                failed.append(rid)
        except Exception as exc:  # noqa: BLE001 —— 单条失败不该吞掉整批
            failed.append(f"{rid}: {type(exc).__name__}: {exc}")
    return ApplyResult(changed=changed, failed=failed,
                       applied_changes=applied_changes)
