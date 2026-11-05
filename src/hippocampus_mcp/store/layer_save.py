# -*- coding: utf-8 -*-
"""层保存的账本编排与写盘编排（core 两个方法的落点）。

原 ``brain.core._save_layers`` / ``_save_layers_core`` 是上帝类里最后一块
大编排：前者给本次保存建账（begin / end / fail 三段），后者做派生层的
「读-合并-写」。搬来 store 的理由：它们全程只与**层文件 I/O + 账本 +
层状态**打交道，是持久化设施，不是认知逻辑；且底层本来就在本包
（``ledger.py`` 的账本行 I/O、``backend`` 的层读写、``vector_store``
的多写开关）。

**注入面纪律**：brain 侧能力一律以可调用对象/值经 ports
注入，store 不 import brain；开关类（``load_config().layer_fp``、
``multi_writer_enabled``）在**调用时**求值，保住测试对
``core.load_config`` 等命名空间的 monkeypatch 面。core 侧只留两个薄壳
（``_save_layers`` / ``_save_layers_core``），壳负责 brain 记账状态的
写回——原实现在写盘成功后才更新指纹戳/序号，失败不记账，这条纪律由
壳的"store 返回后才写回"天然保证。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from ..common.constants import LAYER_DEL_FILE
from .backend import get_backend
from .ledger import ledger_path
from .vector_store import multi_writer_enabled


# ----------------------------------------------------------------------
# 写盘段（原 _save_layers_core）
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LayerSavePorts:
    """brain 侧层 I/O 能力与层状态句柄。

    全部字段要么是值（``layer_attrs`` / ``begin_layer_fps`` /
    ``written_fp`` / ``backend``），要么是**可调用对象**（调用时才取值，
    不是调用结果——这样测试 monkeypatch brain 实例/类属性后仍生效）。
    """

    persist_dir: str | None
    layer_attrs: dict[str, str]
    # ---- 层值读写（attr 名口径，与 _LAYER_ATTRS 的值一致）----
    get_layer: Callable[[str], Any]
    set_layer: Callable[[str, Any], None]
    # ---- 合并族（本体在 brain，纯函数在 brain/pure.py）----
    layer_fingerprint: Callable[[Any], str]
    layer_changed_by_other: Callable[[str, str], bool]
    read_layer_from_disk: Callable[[str], Any]
    read_raw_log_from_disk: Callable[[], list]
    merge_raw_log: Callable[..., Any]
    merge_del_ledger_layer: Callable[..., Any]
    merge_layer: Callable[..., Any]
    disk_only_count: Callable[..., int]
    reclip_derived_layers: Callable[[], None]
    gc_layer_del: Callable[[], int]
    reset_del_blocked: Callable[[], None]
    # ---- 后端 ----
    backend: Any | None
    # ---- 账本指纹（只读引用：core 段只用不写）----
    begin_layer_fps: dict[str, str]
    written_fp: dict[str, str]
    # ---- 开关（调用时现取）----
    layer_fp_on: Callable[[], bool]


@dataclass
class LayerSaveResult:
    """写盘段的产出。壳按此写回 brain 记账状态。

    ``persisted=False`` 表示纯内存 brain（无 persist_dir）——原实现在这种
    情况下**一个状态都不碰**（不清 ``last_layer_merge``、不推进序号），
    壳据此跳过全部写回。
    """

    persisted: bool = False
    last_layer_merge: dict[str, int] = field(default_factory=dict)
    layers_skipped: list[str] = field(default_factory=list)
    end_layer_fps: dict[str, str] = field(default_factory=dict)
    written_fp: dict[str, str] = field(default_factory=dict)
    layer_stamp: dict[str, Any] = field(default_factory=dict)
    del_gc_last: int | None = None


def save_layers_core(ports: LayerSavePorts) -> LayerSaveResult:
    """只写 L0 / L2 / L3 等 brain 负责的层（不碰 manager 的 L1 + 向量）。

    **多进程下所有派生层都必须与 L1 同语义**：L1（``memories.json``）走
    「冲突即合并」（加法），而 brain 负责的层此前是**整份覆写** ——
    两个进程共用一个库时，谁最后落盘谁的版本就是全库版本，对方写的实体
    挂载 / 会话登记 / 墓碑会被静默写没（实测事故）。无异常、无日志。

    所以这里做**读-合并-写**：落盘前先读磁盘版本，按层口径取并集再写。
    单进程（``multi_writer_enabled()`` 为假）行为与改动前完全一致。

    合并进来的条数记在返回值的 ``last_layer_merge``，``stats()`` 里自曝 ——
    "别人的东西被我救回来几条"必须是看得见的数字，不能靠考古。
    """
    if not ports.persist_dir:
        return LayerSaveResult(persisted=False)
    result = LayerSaveResult(persisted=True)
    if multi_writer_enabled():
        ports.set_layer(ports.layer_attrs["brain_raw.json"],
                        ports.merge_raw_log(ports.read_raw_log_from_disk(),
                                            ports.get_layer(
                                                ports.layer_attrs[
                                                    "brain_raw.json"])))
        # 删除账本必须**最先**并。它是别的层做拦截的依据，
        # 放在最后并 = 本次 save 里 miss_log 先按旧账本合并完才拿到新墓碑，
        # 恰好漏掉"对方刚才删的"这一批（顺序错了就白做）。
        ports.reset_del_blocked()
        _del_path = os.path.join(ports.persist_dir, LAYER_DEL_FILE)
        if ports.layer_changed_by_other(LAYER_DEL_FILE, _del_path):
            _disk_del = ports.read_layer_from_disk(LAYER_DEL_FILE)
            if isinstance(_disk_del, dict):
                ports.set_layer(ports.layer_attrs[LAYER_DEL_FILE],
                                ports.merge_del_ledger_layer(
                                    _disk_del,
                                    ports.get_layer(
                                        ports.layer_attrs[LAYER_DEL_FILE])))
        for name, attr in ports.layer_attrs.items():
            if name == LAYER_DEL_FILE:
                continue      # 已在上面单独并过
            if name == "brain_raw.json":
                continue
            _path = os.path.join(ports.persist_dir, name)
            if not ports.layer_changed_by_other(name, _path):
                continue      # 没人动过：内存这份就是最新的
            disk = ports.read_layer_from_disk(name)
            if disk is None:
                continue          # 磁盘没这层：老库首次升级，按本地写
            mine = ports.get_layer(attr)
            merged = ports.merge_layer(name, disk, mine)
            # 报的是「磁盘里有、我内存里没有、被这次合并救回来几条」。
            # 不是 len(merged) - len(disk)：本地为空时那等于 0，
            # 恰恰是**最该报警**的情形（整份覆写就发生在这一刻）。
            rescued = ports.disk_only_count(name, disk, mine)
            if rescued > 0:
                result.last_layer_merge[name] = rescued
            # 内存也并进来：否则本进程后续操作仍按"自己的那一份"判断，
            # 下一次 save 之前看到的还是残缺视图。
            ports.set_layer(attr, merged)
        # 并集只会做加法：合并完按当前存活记忆重新裁一遍，
        # 否则磁盘上的旧版本会把本进程已删的东西复活。
        ports.reclip_derived_layers()
        # 账本瘦身。放在合并之后 —— 刚并进来的墓碑当次就要生效，
        # 不能还没用就被 GC 掉。
        result.del_gc_last = ports.gc_layer_del()
    # G0 / F14-1：只写**内容变了**的层（派生层脏集）。
    # 判据 = 内容指纹没变 **且** 磁盘没被别人改过。两个条件缺一不可：
    # 只看指纹 → 别人删了磁盘文件我们却跳过，层就丢了；只看磁盘戳 →
    # 本进程改了但指纹相同（改回原值）时白写一次。
    files: dict[str, Any] = {}
    new_fp: dict[str, str] = {}
    # begin 指纹能否复用：begin 在 core 之前算，中间若无多进程合并
    # （合并会在下面 set_layer 改层内容）则层值未变，可省一次指纹计算。
    reuse_begin_fps = not multi_writer_enabled() and bool(ports.begin_layer_fps)
    for name, attr in ports.layer_attrs.items():
        value = ports.get_layer(attr)
        if ports.layer_fp_on():
            if reuse_begin_fps:
                fp = ports.begin_layer_fps[name]
            else:
                fp = ports.layer_fingerprint(value)
            if fp == ports.written_fp.get(name) and not ports.layer_changed_by_other(
                name, os.path.join(ports.persist_dir, name)
            ):
                result.layers_skipped.append(name)
                # 跳过层此刻的指纹就是磁盘实况（指纹未变且没人动过）
                result.end_layer_fps[name] = fp
                continue
            new_fp[name] = fp
        files[name] = value
    # 实写层的指纹并入 end 指纹集（跳过层已在循环里落定）
    result.end_layer_fps.update(new_fp)
    # 层在**一个事务**里落盘。json 后端仍是逐文件 replace（行为
    # 不变）；sqlite 后端一次 BEGIN IMMEDIATE 提交，中途失败整批回滚 ——
    # 这正是 JSON 方案在文件系统层补不了的原语（当初选 SQLite 的第一理由）。
    backend = ports.backend or get_backend(ports.persist_dir)
    if files:
        backend.write_layers(files)
    for name in files:
        result.layer_stamp[name] = backend.layer_stamp(name)
    # 指纹只在**真正写成功了**之后才更新：写失败却记账"已写"会让下次
    # 跳过一个其实没落盘的层（与脏集只在写成功后清空同一条纪律）。
    result.written_fp = new_fp
    # F14-5：end 行的层指纹已在循环里逐层落定（跳过层=写前现值、
    # 实写层=new_fp），写失败时 end_layer_fps 保持空 ⇒ end 行不带指纹，
    # 启动对账不会拿半成品指纹当真相。
    return result


# ----------------------------------------------------------------------
# 账本段（原 _save_layers）
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LayerLedgerPorts:
    """账本三段编排的注入面。

    ``core_fn`` 是 brain 的 ``_save_layers_core``（可被测试整体替换为
    故障函数——原实现 ``self._save_layers_core()`` 经实例查找的语义不变）。
    """

    persist_dir: str | None
    save_seq: int
    layer_attrs: dict[str, str]
    now_wall: Callable[[], float]
    now_mono: Callable[[], float]
    get_pid: Callable[[], int]
    compute_all_fps: Callable[[], dict[str, str]]
    layer_count: Callable[[str], int]
    ledger_append: Callable[[dict[str, Any]], None]
    set_begin_fps: Callable[[dict[str, str]], None]
    core_fn: Callable[[], LayerSaveResult | None]


def run_layer_save(ports: LayerLedgerPorts) -> None:
    """给本次保存建账，真正的写盘交给 :func:`save_layers_core`。

    三段各有理由：``begin`` 必须在写盘**之前**（崩溃时至少留下「想写什么」），
    ``end`` 记实际结果，``fail`` 记失败原因 —— 不记就等于「曾经想写但没写成」
    这件事从未发生过（设计 §8 风险 8）。
    """
    path = ledger_path(ports.persist_dir)
    seq = ports.save_seq + 1
    now_wall = ports.now_wall()
    # F14-5：begin 带层指纹（"我声明要写的样子"）。算一次暂存，core 里
    # 复用（仅当本次没走多进程合并——合并会在指纹循环前改层内容，
    # 那时必须重算）。
    ports.set_begin_fps({})
    if path:
        fps_all = ports.compute_all_fps()
        begin_entry = {
            "seq": seq,
            "phase": "begin",
            "at_wall": now_wall,
            "at_mono": ports.now_mono(),
            "pid": ports.get_pid(),
            "layers": {n: ports.layer_count(a)
                       for n, a in ports.layer_attrs.items()},
        }
        if fps_all:
            begin_entry["layer_fps"] = fps_all
            ports.set_begin_fps(fps_all)
        ports.ledger_append(begin_entry)
    t0 = ports.now_mono()
    try:
        result = ports.core_fn()
    except BaseException as e:  # noqa: BLE001 —— 失败也要入账，然后照常抛
        if path:
            ports.ledger_append({
                "seq": seq,
                "phase": "fail",
                "at_wall": ports.now_wall(),
                "at_mono": ports.now_mono(),
                "pid": ports.get_pid(),
                "duration_ms": int((ports.now_mono() - t0) * 1000),
                "error": f"{type(e).__name__}: {e}",
            })
        raise
    result = result or LayerSaveResult(persisted=False)
    if path:
        end_entry = {
            "seq": seq,
            "phase": "end",
            "at_wall": ports.now_wall(),
            "at_mono": ports.now_mono(),
            "pid": ports.get_pid(),
            "duration_ms": int((ports.now_mono() - t0) * 1000),
            "layers": {n: ports.layer_count(a)
                       for n, a in ports.layer_attrs.items()},
            # 从磁盘并回来的层 = 这段时间里有外部写者（四象限右下角）。
            # 切片 1 只记事实，不做判定（判定要等 F14-1 的层版本号）。
            "merged_from_disk": dict(result.last_layer_merge or {}),
            # 本次因"内容没变"跳过的层（派生层脏集的实际效果）。
            # 看得见"省了几次无谓重写"，也防"该写却跳过"没人发现。
            "layers_skipped": list(result.layers_skipped or []),
        }
        # F14-5：end 带层指纹（写完后的样子；core 里已补齐——实写层取
        # new_fp，跳过层继承 begin 值）。
        if result.end_layer_fps:
            end_entry["layer_fps"] = dict(result.end_layer_fps)
        ports.ledger_append(end_entry)
