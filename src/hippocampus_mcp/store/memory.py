"""记忆数据模型。

定义单条记忆记录 ``MemoryRecord`` 以及记忆类型枚举 ``MemoryType``。
本模块不依赖任何外部嵌入模型，嵌入向量由调用方提供。
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# 旧格式（单向量）的迁移槽位标识。
# 早期版本的 memories.json 每条只存一个 ``embedding``，且不记录来自哪个模型；
# 加载时统一归入该槽位，等用户显式认领（reembed --assume-model）或由当前模型重新嵌入。
LEGACY_MODEL_KEY = "__legacy__"

# 重要性钉住线：importance >= 该值的记忆豁免时间衰减（bigeye: importance>=7 不衰减）。
# 这是**唯一**的阈值定义处——MemoryRecord.__post_init__ 与 VectorStore.update
# 都必须引用它，否则新建与更新两条路径会给出不一致的 pin（缺陷 D4）。
PIN_IMPORTANCE = 0.7


def _as_f32_vector(value: Any) -> array:
    """向量统一存 ``array('f')``（双存修复）。

    此前向量以 **Python float list** 存进 ``record.embeddings``：768 维 ≈
    24.6KB/条，真库 9859 条实测 **240.9MB**——与检索矩阵（npz → 惰性
    `_active_matrix`）同内容**双存**。
    ``array('f')`` 每维 4B ≈ 3.1KB/条（**−87%**），且消费方全兼容：

    * ``np.asarray(v)`` 直接吃（检索矩阵 / 相似度数学零改动）；
    * ``list(v)`` / ``v.tolist()`` 照旧出 JSON（``to_dict`` 兼容不变）；
    * npz / sqlite blob 保存路径都是 ``np.asarray`` 后序列化，无感。

    接受 list / tuple / np.ndarray / array('f')（f 型原样返回避免复制）。
    """
    if isinstance(value, array) and value.typecode == "f":
        return value
    return array("f", value)


def merge_id_lists(
    old: list[str] | None, new: list[str] | None
) -> list[str]:
    """把两组 id 合并为**有序并集**（去重，旧值在前、新值按序追加）。

    去重合并（dedup）时用来累积 ``source_ids`` / ``entity_ids``：
    直接覆盖会让新来的 L0 原文失去回指（审计链断裂）、让新提到的实体
    与这条记忆失去关联（缺陷 D3）。
    """
    merged = list(old or [])
    seen = set(merged)
    for item in new or []:
        if item not in seen:
            seen.add(item)
            merged.append(item)
    return merged


def merge_tag_dicts(
    old: dict[str, Any] | None, new: dict[str, Any] | None
) -> dict[str, Any]:
    """合并两组结构化标签。

    键冲突时**保留既存值**：命中去重意味着两条内容近乎相同（>= dedup 阈值），
    标签几乎总是同值；真出现不同值，说明这条记录确实横跨两个取值
    （如内容同时涉及 ch43 与 ch44），此时任取其一都会让"按章过滤"给出
    半对的结果——保留既存值至少是**确定性**的，且合并结果会写进
    ingest 的返回里，调用方看得见。
    """
    merged = dict(new or {})
    merged.update(old or {})
    return merged


class MemoryType(str, Enum):
    """记忆类型枚举。

    继承 ``str`` 使其既能作为枚举使用，也可直接当作字符串参与序列化与比较。
    """

    EPISODIC = "episodic"      # 情节记忆：具体事件 / 经历
    SEMANTIC = "semantic"      # 语义记忆：事实 / 知识
    PROCEDURAL = "procedural"  # 程序记忆：技能 / 流程
    GENERAL = "general"        # 通用记忆：默认类型


class MemoryState(str, Enum):
    """记忆生命周期状态（借鉴 bigeye relation_store 的事实状态机）。"""

    ACTIVE = "active"          # 当前有效
    SUPERSEDED = "superseded"  # 已被更新的记忆取代（保留，不删除）
    ARCHIVED = "archived"      # 长期未调用，已归档（仍在库，不进默认召回）


class CognitiveType(str, Enum):
    """认知类型：记忆的认识论来源，用于反幻觉引用。"""

    EXPERIENCE = "experience"  # 亲历 / 一起做过的事
    WORLD = "world"            # 客观事实（可外部验证）
    OPINION = "opinion"        # 主观看法 / 判断


@dataclass
class MemoryRecord:
    """单条记忆记录的数据模型（v0.4：向量与模型解耦）。

    Attributes:
        id: 记忆唯一标识。
        content: 记忆的文本内容（记忆本体，与嵌入模型无关）。
        timestamp: 记忆创建 / 发生的时间戳（单位：秒）。
        embeddings: 模型标识 -> 该模型产出的嵌入向量。
            **向量是可重建的派生物**：切换嵌入模型时无需重新导入文档，
            只需为当前模型补算向量（旧模型向量可保留用于横向对比）。
        importance: 记忆重要程度，取值范围通常为 [0, 1]，默认 0.5。
        recall_count: 被回忆 / 检索命中的次数，默认 0。
        memory_type: 记忆类型字符串，默认 "general"。
        state: 生命周期状态，默认 active。
        cognitive_type: 认知类型（experience/world/opinion），默认 world。
        source_ids: L0 原始层回指 id 列表（保证"核对原话"能力）。
        entity_ids: 关联实体 id 列表（索引层用）。
        superseded_by: 若被取代，指向取代者的记忆 id。
        last_recalled_at: 最近一次被召回的时间戳。
        pin: 是否钉住（豁免时间衰减）。importance>=0.7 自动为真。
        source: 来源标签（文件名 / 相对路径 / 会话标识）。
            批量导入场景下是"这条记忆出自哪个 md 文件"，让 recall 命中的
            结果能直接回指原始文档（覆盖报告·高 3：此前只存 L0 id，
            回查不到原文件，按文件审计与增量重导都做不了）。
        tags: 结构化标签（zcode 需求 A1）。强约束写作项目的检索大多是
            **条件检索**：按章号、按年份月份、按单源文件，靠语义匹配碰运气。
            tags 让这类查询从"猜"变成"查"。
            与实体的分工：tags 放**结构化硬约束**（章/卷/年/文件），
            人物/专名继续走实体倒排索引——两套重复造会互相打架。
    """

    id: str
    content: str
    timestamp: float
    embeddings: dict[str, array] = field(default_factory=dict)
    importance: float = 0.5
    recall_count: int = 0
    memory_type: str = "general"
    state: str = MemoryState.ACTIVE.value
    cognitive_type: str = CognitiveType.WORLD.value
    source_ids: list[str] | None = None
    entity_ids: list[str] | None = None
    superseded_by: str | None = None
    last_recalled_at: float = 0.0
    # 检索强度：成功被召回（进结果集）或被重放强化时累加，
    # 打分按 +w_strength·log1p(强度) 加权——生物侧"检索即巩固"（测试效应）
    # 的最小实现。与 recall_count 的分工见 engine/manager.reinforce 的
    # docstring：recall_count 记候选集（多样性惩罚用），本字段记结果集。
    recall_strength: float = 0.0
    pin: bool = False
    source: str = ""
    tags: dict[str, Any] = field(default_factory=dict)
    # D2a 写侧隔离：True = 待审（非家工作区经 user 豁免通道的直写）。
    # 不进召回池（recall_flow）、不进 user_scope 槽（user_scope.select）；
    # drift_audit 过目后经受管通道转正（判据单一真相见
    # domain/write_quarantine.should_quarantine）。默认 False = 存量库
    # 零变化；序列化**稀疏**（False 不落键），旧库加载零迁移。
    quarantine: bool = False
    # D1 隐私打标：空串 = 未打（存量库态）；"personal" /
    # "sensitive:类别1+类别2"。打标是提示不是执法（漏检常态，见
    # domain/sensitive.classify_sensitivity docstring）；消费方是 D2b
    # 读侧 masked 档过滤与 drift_audit 复审分组。
    sensitivity: str = ""
    # P0-1 写入凭据（Super-agent 适配）：{session, task, surface}，
    # 由 common/write_origin.build_origin 在工具边界构造、原样搬运到这里。
    # **只写不筛**——recall 不读它（先攒真库取值分布，再接检索过滤；
    # 写路径与检索路径同时改是 G 段的教训）。稀疏序列化：空 dict 不落键，
    # 存量卡逐字节不变。
    origin: dict[str, str] = field(default_factory=dict)
    # P0-2 幂等键：宿主声明的「这是同一次请求」（重试 / 重放 / resume 后重跑）。
    # 真相源是 store/idem.py 的进程内倒排，本字段是它**落盘后的重建依据**
    # ——索引是派生物，不另开 sidecar 文件。空串 = 未声明幂等（默认）。
    idem: str = ""
    # Provenance of the pin: "auto" (importance threshold), "manual"
    # (caller supplied pin=True), "legacy" (record written before this
    # field existed), or "" (not pinned).  Declared LAST so positional
    # construction of existing fields is unchanged.  "legacy" is never
    # serialized, keeping old cards byte-identical on disk.
    pin_source: str = ""

    def __post_init__(self) -> None:
        # importance >= PIN_IMPORTANCE 自动钉住（bigeye: importance>=7 不衰减）
        if self.importance >= PIN_IMPORTANCE:
            self.pin = True
        if self.pin_source != "legacy":
            if self.pin and not self.pin_source:
                self.pin_source = (
                    "auto" if self.importance >= PIN_IMPORTANCE else "manual")
            if not self.pin:
                self.pin_source = ""
        # H8 双存修复：所有构造路径（from_dict / 直接构造 / 测试）统一归一化成
        # array('f')——只在这里和 set_vector 转换，消费方（np 数学 / json / npz
        # / sqlite blob）全部无感。
        if self.embeddings:
            self.embeddings = {
                k: _as_f32_vector(v) for k, v in self.embeddings.items()
            }

    def sync_pin(self) -> None:
        """按当前 ``importance`` 重算 ``pin``。

        ``__post_init__`` 只在**构造**时生效，而 ``VectorStore.update`` 走的是
        ``setattr``，会绕过它——于是"提升 importance 自动钉住"只在新建路径成立，
        合并（dedup）路径提升 importance 时 pin 依然是 False（缺陷 D4）。
        任何改动 importance 的地方都应调用本方法（显式设置 pin 时除外）。
        """
        self.pin = self.importance >= PIN_IMPORTANCE
        if self.pin and self.pin_source in ("", "legacy"):
            self.pin_source = "auto"
        elif not self.pin:
            self.pin_source = ""

    def is_live(self) -> bool:
        """是否参与默认召回（archived 不参与）。"""
        return self.state != MemoryState.ARCHIVED.value

    # ------------------------------------------------------------------
    # 向量读写（按模型标识存取，模型无关）
    # ------------------------------------------------------------------
    def vector_for(self, model_key: str) -> array | None:
        """取指定模型的向量（``array('f')``；``np.asarray`` 可直接吃）。

        该模型未算过则返回 ``None``（触发补算）。
        """
        return self.embeddings.get(model_key)

    def has_vector(self, model_key: str) -> bool:
        """是否已有指定模型的向量。"""
        return model_key in self.embeddings

    def set_vector(self, model_key: str, vector: Any) -> None:
        """写入 / 覆盖指定模型的向量（统一归一化为 ``array('f')``）。"""
        self.embeddings[model_key] = _as_f32_vector(vector)

    def drop_vectors(self, keep: set[str]) -> int:
        """丢弃不在 ``keep`` 中的模型向量，返回丢弃个数。"""
        drop = [k for k in self.embeddings if k not in keep]
        for k in drop:
            del self.embeddings[k]
        return len(drop)

    def model_keys(self) -> list[str]:
        """返回本条记忆已有向量的模型标识列表。"""
        return list(self.embeddings.keys())

    def to_dict(self, with_embeddings: bool = False) -> dict[str, Any]:
        """将记录序列化为可 JSON 化的字典。

        Args:
            with_embeddings: 是否把向量一并写入 JSON。默认 ``False``——
                向量体积大（768 维 float64 序列化后约 15KB/条），
                统一由 ``VectorStore`` 写入二进制 .npz 文件。
        """
        data: dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "recall_count": self.recall_count,
            "memory_type": self.memory_type,
            "state": self.state,
            "cognitive_type": self.cognitive_type,
            "source_ids": self.source_ids or [],
            "entity_ids": self.entity_ids or [],
            "superseded_by": self.superseded_by,
            "last_recalled_at": self.last_recalled_at,
            "recall_strength": self.recall_strength,
            "pin": self.pin,
            "source": self.source,
            "tags": dict(self.tags or {}),
        }
        # D2a/D1 两字段**稀疏**序列化：默认值不落键——存量卡逐字节不变，
        # 只有真被隔离/打标的卡才多出这两个键；旧版本代码读到多出的键也
        # 安全（from_dict 对缺键取默认，未知键本就不读）。
        if self.quarantine:
            data["quarantine"] = True
        if self.sensitivity:
            data["sensitivity"] = self.sensitivity
        # P0-1/P0-2 同样稀疏：存量卡不加键，旧版本代码读到多出的键也不读
        if self.origin:
            data["origin"] = dict(self.origin)
        if self.idem:
            data["idem"] = self.idem
        if self.pin_source and self.pin_source != "legacy":
            data["pin_source"] = self.pin_source
        if with_embeddings:
            data["embeddings"] = {k: list(v) for k, v in self.embeddings.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        """从字典反序列化构建 ``MemoryRecord``。

        兼容三种历史形态：
            - v0.4: ``embeddings`` 多模型字典；
            - v0.3 及更早: 单个 ``embedding`` 列表（归入 ``__legacy__`` 槽位）；
            - 完全没有向量字段（由 VectorStore 从 .npz 另行注入）。
        缺失的可选字段使用默认值，保证向前兼容。
        """
        if "embeddings" in data and isinstance(data["embeddings"], dict):
            embeddings = {
                str(k): [float(x) for x in v]
                for k, v in data["embeddings"].items()
            }
        elif data.get("embedding"):
            embeddings = {
                LEGACY_MODEL_KEY: [float(x) for x in data["embedding"]]
            }
        else:
            embeddings = {}

        return cls(
            id=data["id"],
            content=data["content"],
            timestamp=float(data["timestamp"]),
            embeddings=embeddings,
            importance=float(data.get("importance", 0.5)),
            recall_count=int(data.get("recall_count", 0)),
            memory_type=data.get("memory_type", "general"),
            state=data.get("state", MemoryState.ACTIVE.value),
            cognitive_type=data.get("cognitive_type", CognitiveType.WORLD.value),
            source_ids=list(data.get("source_ids") or []),
            entity_ids=list(data.get("entity_ids") or []),
            superseded_by=data.get("superseded_by"),
            last_recalled_at=float(data.get("last_recalled_at", 0.0)),
            # 旧库/半截数据没有该字段 → 0.0（无强度，与新建记录同态）
            recall_strength=float(data.get("recall_strength") or 0.0),
            pin=bool(data.get("pin", False)),
            source=str(data.get("source") or ""),
            # 旧库没有这个字段 → 空 dict，过滤时不匹配任何 tag 条件（安全默认）
            tags=dict(data["tags"]) if isinstance(data.get("tags"), dict) else {},
            # D2a/D1：旧库/旧代码写的卡没有这两个键 → False/""（与新建默认同态）
            quarantine=bool(data.get("quarantine", False)),
            sensitivity=str(data.get("sensitivity") or ""),
            # P0-1/P0-2：旧库没有这两个键 → 空 dict / 空串（与新建默认同态）
            origin=(
                {str(k): str(v) for k, v in data["origin"].items()}
                if isinstance(data.get("origin"), dict) else {}
            ),
            idem=str(data.get("idem") or ""),
            pin_source=str(data.get("pin_source") or "legacy"),
        )
