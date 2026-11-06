# -*- coding: utf-8 -*-
"""幂等索引：``idem_key -> memory_id`` 的进程内倒排。

**为什么要它**：编排层（Router + 多 Worker）一定会重试 —— 网络瞬断、
effect 队列退避重放、Ticket 被 resume 后重跑。现在的写路径只有**相似度**
去重（``dedup_threshold``），那是"内容像不像"，不是"这次请求是不是同一次"：

* 相似度合并后 id 会变（并进旧卡），重放拿不到稳定的回执；
* 两句话相似度 0.84 < 0.85 时会各建一张卡，重放一次库就多一条；
* 更糟的是「同一件事」和「措辞很像的两件事」在相似度空间里无法区分。

真需求是**精确键**：宿主说"这是第 N 次重试"，服务端就返回第一次的 id。

设计口径
--------
1. **索引是派生物，不是新的一层盘**。``idem_key`` 随记录落盘
   （``MemoryRecord.idem`` 稀疏字段），进程启动后**首次用到时**扫一遍
   内存记录重建，不引入新的 sidecar 文件（多一个文件就多一个不一致面，
   见 ``common/sidecar_lock`` 的血案）。
2. **键不归一化内容，只归一空白与长度**。``"a" -> " a "`` 视为同键是
   宿主有意的（重试时多打了空格不该被当成新请求）；大小写**不归一**
   ——Base62 随机串大小写敏感，误归一 = 两次不同请求被吞。
3. **命中即不写**。查不到才写；写入方负责把新 id 登记进来
   （``remember``）。**历史不存**：只保留"这个键最终落在哪条记忆"，
   不记第几次调用（那是 ledger 的事）。
"""

from __future__ import annotations

from typing import Any, Iterable

__all__ = ["MAX_KEY_CHARS", "IdemIndex", "normalize_key"]

#: 幂等键长度上限。超长键几乎一定是宿主把整段 JSON 塞进来了（误用），
#: 截断后仍然可用（前缀足够特异），但不会让每条记忆多背几 KB。
MAX_KEY_CHARS = 200


def normalize_key(key: Any) -> str:
    """归一幂等键：去首尾空白 + 内部连续空白压一格 + 截断。

    空串 / None 一律返回 ``""``，调用方按「没用幂等」处理（**绝不**把
    空串当成一个合法键——那会让所有不带键的写入互相命中）。
    """
    if key is None:
        return ""
    text = str(key).strip()
    if not text:
        return ""
    # 内部空白压缩：重放时多打一个换行不该变成新请求
    text = " ".join(text.split())
    return text[:MAX_KEY_CHARS]


class IdemIndex:
    """``idem_key -> memory_id`` 的薄字典（无 I/O、无锁）。

    线程安全由调用方（``MemoryManager``）的写路径锁保证——本类只做
    映射，不猜并发模型。daemon 下 N 条连接共享一个 manager 实例，
    这把锁本来就已经串住了写入。
    """

    def __init__(self) -> None:
        self._map: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._map)

    def rebuild(self, records: Iterable[Any]) -> int:
        """从记忆记录重建索引，返回登记条数。

        同一键在多条记录上出现时（合并路径可能留下这种历史），**取最新
        时间戳那条**——它是"这个请求最终停在哪"的最接近真相。
        """
        self._map.clear()
        best: dict[str, float] = {}
        for rec in records:
            key = normalize_key(getattr(rec, "idem", ""))
            if not key:
                continue
            ts = float(getattr(rec, "timestamp", 0.0) or 0.0)
            if key in best and best[key] >= ts:
                continue
            best[key] = ts
            self._map[key] = getattr(rec, "id", "")
        return len(self._map)

    def lookup(self, key: Any) -> str | None:
        """按键查既有记忆 id（未命中返回 ``None``）。"""
        norm = normalize_key(key)
        if not norm:
            return None
        return self._map.get(norm)

    def remember(self, key: Any, memory_id: str) -> bool:
        """登记一条映射。**先到先得**：已存在的键不覆盖，返回 False。

        为什么不覆盖：同一个键第二次写入本就不该发生（会先被
        ``lookup`` 拦住）；能走到这里的都是"查的时候还没有"的并发窗口，
        此时先落地的那个 id 才是重放应当拿到的一致回执。
        """
        norm = normalize_key(key)
        if not norm or not memory_id:
            return False
        if norm in self._map:
            return False
        self._map[norm] = memory_id
        return True

    def forget(self, key: Any) -> bool:
        """删除一条映射（记忆被 ``forget()`` 真删除后调用，防悬空回执）。"""
        norm = normalize_key(key)
        if not norm:
            return False
        return self._map.pop(norm, None) is not None

    def forget_memory(self, memory_id: str) -> int:
        """删除指向某条记忆的全部映射，返回删除条数。"""
        if not memory_id:
            return 0
        dead = [k for k, v in self._map.items() if v == memory_id]
        for key in dead:
            del self._map[key]
        return len(dead)
