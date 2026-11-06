"""层 GC 族：L0 自动落盘 / 删除账本拦截与瘦身 / 层合并
路由 / 派生层重裁 / 启动对账。

从 ``brain.core`` 整搬六件（I/O 密集落 store；与 ``store/layer_save.py``
分工——那边管"全量保存编排"，这里管"层级的增删并瘦身边角"）。

**状态容器不搬**：``layer_del`` / ``layer_stamp`` 由壳注入本体或以
"返回新值、壳回写"的口径处理（``_gc_layer_del`` 原实现就是**重绑新
dict**，模块返回 ``(新 dict, 丢了几条)``、壳回写属性，语义一致）；
``raw_log`` 同理（多写并集后壳回写 ``self.raw_log``）。

**副作用留壳**：``_save_l0_only`` 的 try/except、
``manager.last_save_error`` 记账与 logger 自曝、``_l0_pending`` 清零
都留在壳（"收尾落盘留壳"的同一判断——注入端口会比原代码还长）；
本模块只做落盘核心，异常向上抛。

逐字保留的语义（动一个就是行为变化，parity 会抓）：

  * save_l0_only：多写进程下 L0 append-only，必须先读盘并集再写
    （整份覆写会把别的进程写的 raw 静默写没）；写完盖层戳；
  * apply_layer_deletions：开关关或账本空直接放行；放行条件是条目
    时间戳**晚于**删除时刻（删后又命中=新留痕，不是复活）；list 与
    dict 两种形态分别拦截；被拦条数由壳累加 ``_layer_del_blocked``；
  * gc_layer_del：先按 TTL 丢超龄（非数值时间戳一并丢），仍超量按
    时间戳降序保留最新 max_keys 条；返回丢掉几条；
  * merge_layer：按层规则路由（未知层=dict 规则）；``del_ledger`` 层
    不参与拦截（自己没有删除语义）；合并完统一过删除账本；
  * reclip_derived_layers：L1 视图为空（count==0）直接返回——宁可漏裁
    不可误裁；实体重算统计后摘空壳；画像走 prune；主题层走 reclip；
  * reconcile_ledger_on_boot：纯解析（末条 end/begin/fail 三态判
    lost/mismatch），日志留壳。
"""
from __future__ import annotations

from typing import Any, Callable

from .ledger import reconcile_on_boot


def save_l0_only(
    *, persist_dir: str, raw_log: list, multi_writer_fn: Callable[[], bool],
    read_raw_fn: Callable[[], list], merge_raw_fn: Callable[..., list],
    backend: Any, get_backend_fn: Callable[[str], Any],
    layer_stamp: dict,
) -> list:
    """L0 落盘核心：多写并集 → 后端写 → 盖层戳。返回（可能并集过的）
    新 raw_log；异常向上抛（记账与自曝在壳）。"""
    if multi_writer_fn():
        # 多写进程下 L0 是 append-only：必须与磁盘取并集，
        # 整份覆写会把别的进程写的 raw 静默写没（同款事故）
        raw_log = merge_raw_fn(read_raw_fn(), raw_log)
    # G1：走后端（json 写单文件；sqlite 只更新 layers 表里这一行）
    backend = backend or get_backend_fn(persist_dir)
    backend.write_layers({"brain_raw.json": raw_log})
    layer_stamp["brain_raw.json"] = backend.layer_stamp("brain_raw.json")
    return raw_log


def apply_layer_deletions(
    name: str, merged: Any, *, enabled: bool, layer_del: dict,
    layer_key_fn: Callable[[dict], str], entry_ts_fn: Callable[[dict], float],
) -> tuple[Any, int]:
    """合并之后：把账本里已标记删除的键挡掉。返回 ``(合并结果, 拦截数)``。"""
    blocked = 0
    if not enabled or not layer_del:
        return merged, blocked
    prefix = name + "::"
    if isinstance(merged, list):
        out = []
        for it in merged:
            if isinstance(it, dict):
                dead = layer_del.get(prefix + layer_key_fn(it))
                if dead is not None and entry_ts_fn(it) <= float(dead):
                    blocked += 1
                    continue
            out.append(it)
        return out, blocked
    if isinstance(merged, dict):
        out = {}
        for k, v in merged.items():
            dead = layer_del.get(prefix + str(k))
            if dead is not None and (
                    not isinstance(v, dict) or entry_ts_fn(v) <= float(dead)):
                blocked += 1
                continue
            out[k] = v
        return out, blocked
    return merged, blocked


def gc_layer_del(
    layer_del: dict, *, now_fn: Callable[[], float], ttl: float,
    max_keys: int,
) -> tuple[dict, int]:
    """账本瘦身：丢超龄的；仍超量就丢最旧的。返回 ``(新账本, 丢掉几条)``。

    ⚠️ 已知折中：丢掉的墓碑理论上可能让**更老**的快照复活对应条目。
    30 天远超任何一份内存快照的合理寿命，接受这个风险并写在这里。
    """
    if not layer_del:
        return layer_del, 0
    before = len(layer_del)
    now = now_fn()
    if ttl > 0:
        layer_del = {
            k: v for k, v in layer_del.items()
            if isinstance(v, (int, float)) and now - float(v) <= ttl
        }
    if max_keys and len(layer_del) > max_keys:
        keep = sorted(layer_del.items(), key=lambda kv: kv[1],
                      reverse=True)[:max_keys]
        layer_del = dict(keep)
    return layer_del, before - len(layer_del)


def merge_layer(
    name: str, disk: Any, mine: Any, *, rules: dict, fns: dict,
    apply_del_fn: Callable[[str, Any], Any], miss_log_max: int,
    list_caps: dict[str, int] | None = None,
) -> Any:
    """按层口径合并（磁盘打底 + 本地补充），再按删除账本拦截。

    ``list_caps``：list_cap 规则的 per-层上限表（缺省回落
    ``miss_log_max``——历史行为逐位不变）。brain_topics 走这里传
    ``OPEN_TOPICS_MAX``。
    """
    rule = rules.get(name, "dict")
    if rule == "del_ledger":
        # 账本自己不参与拦截：它的键形如 "层::键"，别的层的 prefix 命中
        # 不到它；反过来它也拦不了自己（自己没有删除语义）。
        return fns["del_ledger"](disk, mine)
    if rule == "raw":
        merged = fns["raw"](disk if isinstance(disk, list) else [], mine)
    elif rule == "list":
        merged = fns["list"](disk, mine)
    elif rule == "list_cap":
        merged = fns["list_cap"](
            disk, mine, cap=(list_caps or {}).get(name, miss_log_max))
    elif rule == "entity":
        merged = fns["entity"](disk, mine)
    elif rule == "session":
        merged = fns["session"](disk, mine)
    elif rule == "intent":
        merged = fns["intent"](disk, mine)
    elif rule == "counter":
        merged = fns["counter"](disk, mine)
    elif rule == "topic":
        merged = fns["topic"](disk, mine)
    elif rule == "sourcesig":
        merged = fns["sourcesig"](disk, mine)
    else:
        merged = fns["dict"](disk, mine)
    # 合并做完（并集只会做加法）之后，按删除账本把已删的键挡回去。
    return apply_del_fn(name, merged)


def reclip_derived_layers(
    entities: dict, profile: dict, knowledge: dict, *, count_fn: Callable[[], int],
    recompute_entity_stats_fn: Callable[[], Any], prune_fn: Callable[[], Any],
    reclip_knowledge_fn: Callable[[], Any],
) -> None:
    """并集之后，按**当前真相**重新裁剪派生层。

    实体 / 画像本来就是 L1 的派生副本，不需要额外的删除日志：合并完拿
    当前存活记忆重新裁一遍即可。L1 视图为空（count==0）直接返回——
    宁可漏裁，不可误裁（与 prune_stale_profile 同源）。
    """
    if count_fn() == 0:
        return
    if isinstance(entities, dict) and entities:
        recompute_entity_stats_fn()
        for k in [k for k, e in list(entities.items())
                  if not (e.get("memory_ids") or [])]:
            entities.pop(k, None)
    if isinstance(profile, dict) and profile:
        prune_fn()
    # 主题层同样是 L1 的派生副本：并集会把已删记忆的成员并回来。
    if isinstance(knowledge, dict) and knowledge:
        reclip_knowledge_fn()


def reconcile_ledger_on_boot(
    last: Any, tail: list, attrs: list,
    fingerprint_fn: Callable[[str], str],
) -> dict:
    """启动对账的纯解析：末条 end/begin/fail 三态判 lost/mismatch。

    行里没有 ``layer_fps``（旧格式/指纹关）→ ``layer_fps_absent``，不判
    内容。双钟纪律：不新增时间字段，lost 的时间语义引用账本行的
    at_wall/at_mono。
    """
    return reconcile_on_boot(
        last, tail, attrs,
        lambda name: fingerprint_fn(name),
    )
