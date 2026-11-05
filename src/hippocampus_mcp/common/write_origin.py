# -*- coding: utf-8 -*-
"""写入凭据（origin）：把「谁在什么任务下写的」变成**结构化字段**。

**要解决的具体病**：``MemoryRecord.source`` 是自由字符串（文件名 / 相对路径 /
会话标识），由宿主自己填。多 Worker 共享一个库之后，这条通道回答不了三件事：

1. 这条记忆是 Router 写的还是 Worker-3 写的？（``source`` 里可能只有路径）
2. 它属于哪个工单 / 哪次任务？（现在只能靠人翻 ``raw_log`` 反推）
3. 事后归因：某个 Worker 跑歪了，它污染了哪几条？（没有主体字段就查不出来）

Super-agent（Router + 每工单一个 Worker）把一个「架构假设」变成了硬需求：
**同一时刻多个主体在往同一个库写**。缺主体归属时，坏结果只能整体回滚，
没法精确回收。

三条口径（与 ``domain/write_quarantine`` / ``domain/task_directory`` 同族）：

1. **只写不筛**：本模块产出的 origin **不进检索路径**（recall 不读它）。
   先让数据落下来，等真库上验证过取值分布，再接过滤——一次只动一层，
   写路径与检索路径同时改是 G 段吃过的教训。
2. **稀疏**：取值为空就不落键，存量卡逐字节不变（同 ``quarantine`` /
   ``sensitivity`` 的序列化纪律）。
3. **不猜**：没有 ``task`` 来源就留空，不从正文里猜任务名。
   ``task`` 的真相源应当是宿主的 ``session_start(task=...)``；本模块只搬运。

依赖方向：本模块是纯函数，只依赖标准库，不 import 本包其它模块
（``common`` 是 L0，任何层都可以依赖它，它不能反过来依赖任何人）。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ORIGIN_FIELDS",
    "MAX_LEN",
    "MAX_VALUE_CHARS",
    "build_origin",
    "merge_origin",
    "normalize_origin",
]

#: origin 的固定字段集合。**顺序即语义优先级**：``session`` 永远有值
#: （daemon 下是 per-connection key），``task`` 依赖宿主显式传入，
#: ``surface`` 是写入面（工具名 / CLI / 导入脚本）。
ORIGIN_FIELDS: tuple[str, ...] = ("session", "task", "surface")

#: 逐字段长度上限。origin 是**索引维度**不是正文，一条记忆多存 200 字
#: 任务全文会让 5.28MB 的 memories.json 无谓膨胀（且这些字已经进了正文）。
MAX_LEN: dict[str, int] = {"session": 64, "task": 200, "surface": 32}

#: 全字段总长兜底（防止宿主塞进超长 JSON 串绕过逐字段上限）。
MAX_VALUE_CHARS = 320


def normalize_origin(raw: Any) -> dict[str, str]:
    """把任意形态的输入归一到 ``{session/task/surface}`` 三键字典。

    只保留白名单字段（未知键丢弃——origin 是要按字段查的，多出来的键
    既不会被读也不会被索引，留着只是噪声）；值统一 ``str().strip()``
    并按 :data:`MAX_LEN` 截断。
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key in ORIGIN_FIELDS:
        value = raw.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        out[key] = text[: MAX_LEN.get(key, 64)]
    if not out:
        return {}
    # 总长兜底：按字段顺序截，保 session（归因的第一主键）
    budget = MAX_VALUE_CHARS
    trimmed: dict[str, str] = {}
    for key in ORIGIN_FIELDS:
        if key not in out:
            continue
        value = out[key]
        if len(value) > budget:
            value = value[:budget]
        if value:
            trimmed[key] = value
            budget -= len(value)
    return trimmed


def build_origin(
    session: str = "", task: str = "", surface: str = ""
) -> dict[str, str]:
    """构造一条写入凭据（空值不落键，结果可能是 ``{}``）。

    ``session`` 取宿主连接标识（``common.session_ctx.session_key()``）；
    stdio 直连形态下它恒为 ``__default__``——那不是 bug，是**一进程一会话**
    的正常态（与 LNN 分路同源限制，见 ``common/session_ctx`` 的 docstring）。
    """
    return normalize_origin(
        {"session": session, "task": task, "surface": surface}
    )


def merge_origin(
    old: dict[str, str] | None, new: dict[str, str] | None
) -> dict[str, str]:
    """合并两组 origin：**既存值优先，只补空缺键**。

    与 :func:`store.memory.merge_tag_dicts` 同一原则——去重合并意味着两条
    内容近乎相同，主体通常也相同；真出现不同主体（两个 Worker 独立写到
    同一件事实上），保留先写的那个至少是**确定性**的，且新主体已在
    ``raw_log`` 里留痕，不会丢。
    """
    merged = dict(normalize_origin(new))
    merged.update(normalize_origin(old))
    return merged
