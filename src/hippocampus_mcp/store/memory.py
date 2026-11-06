"""Memory data model.

Defines the single memory record ``MemoryRecord`` and the memory type
enum ``MemoryType``. This module depends on no external embedding model;
embedding vectors are supplied by the caller.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Migration slot marker for the old format (single vector).
# Early versions of memories.json stored one ``embedding`` per record with no
# record of which model produced it; on load they are all filed into this
# slot, awaiting explicit user claim (reembed --assume-model) or re-embedding
# by the current model.
LEGACY_MODEL_KEY = "__legacy__"

# Importance pin line: memories with importance >= this value are exempt from
# time decay (bigeye: importance>=7 does not decay). This is the **only**
# place the threshold is defined -- MemoryRecord.__post_init__ and
# VectorStore.update must both reference it, otherwise the create and update
# paths would produce inconsistent pins (defect D4).
PIN_IMPORTANCE = 0.7


def _as_f32_vector(value: Any) -> array:
    """Vectors are uniformly stored as ``array('f')`` (double-storage fix).

    Previously vectors were stored in ``record.embeddings`` as a **Python
    float list**: 768 dims ≈ 24.6KB per record; measured on a real library
    of 9859 records: **240.9MB** -- duplicate storage of the same content as
    the retrieval matrix (npz → lazy ``_active_matrix``).
    ``array('f')`` is 4B per dim ≈ 3.1KB per record (**−87%**), and all
    consumers stay compatible:

    * ``np.asarray(v)`` takes it directly (zero changes to the retrieval
      matrix / similarity math);
    * ``list(v)`` / ``v.tolist()`` still produce JSON as before (``to_dict``
      compatibility unchanged);
    * the npz / sqlite blob save paths all serialize via ``np.asarray``,
      no perceptible change.

    Accepts list / tuple / np.ndarray / array('f') (the 'f' type is returned
    as-is to avoid a copy).
    """
    if isinstance(value, array) and value.typecode == "f":
        return value
    return array("f", value)


def merge_id_lists(
    old: list[str] | None, new: list[str] | None
) -> list[str]:
    """Merge two id lists into an **ordered union** (dedup; old values
    first, new values appended in order).

    Used to accumulate ``source_ids`` / ``entity_ids`` during dedup merges:
    direct overwrite would leave the newly arrived L0 originals without a
    back-reference (broken audit chain) and would disconnect newly
    mentioned entities from this memory (defect D3).
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
    """Merge two structured tag dicts.

    On key conflict the **existing value is kept**: a dedup hit means the
    two contents are near-identical (>= dedup threshold), so tags almost
    always agree; if values genuinely differ, the record really spans two
    values (e.g. the content involves both ch43 and ch44), and picking
    either one would give "filter by 章" a half-right result -- keeping the
    existing value is at least **deterministic**, and the merged result is
    written into ingest's return, visible to the caller.
    """
    merged = dict(new or {})
    merged.update(old or {})
    return merged


class MemoryType(str, Enum):
    """Memory type enum.

    Inherits ``str`` so it works both as an enum and directly as a string
    in serialization and comparison.
    """

    EPISODIC = "episodic"      # episodic memory: concrete events / experiences
    SEMANTIC = "semantic"      # semantic memory: facts / knowledge
    PROCEDURAL = "procedural"  # procedural memory: skills / workflows
    GENERAL = "general"        # general memory: the default type


class MemoryState(str, Enum):
    """Memory lifecycle states (modeled on bigeye relation_store's fact state machine)."""

    ACTIVE = "active"          # currently valid
    SUPERSEDED = "superseded"  # superseded by an updated memory (kept, not deleted)
    ARCHIVED = "archived"      # long uncalled, archived (still in the library, excluded from default recall)


class CognitiveType(str, Enum):
    """Cognitive types: the epistemic origin of a memory, for anti-hallucination citation."""

    EXPERIENCE = "experience"  # personally lived / things done together
    WORLD = "world"            # objective facts (externally verifiable)
    OPINION = "opinion"        # subjective views / judgments


@dataclass
class MemoryRecord:
    """Data model for a single memory record (v0.4: vectors decoupled from models).

    Attributes:
        id: unique memory id.
        content: the memory's text content (the memory itself, independent
            of the embedding model).
        timestamp: creation / occurrence timestamp (unit: seconds).
        embeddings: model key -> the embedding vector that model produced.
            **Vectors are a rebuildable derivative**: switching embedding
            models requires no document re-import, just computing vectors
            for the current model (old model vectors may be kept for side
            comparison).
        importance: memory importance, usually in [0, 1], default 0.5.
        recall_count: number of times recalled / hit by retrieval, default 0.
        memory_type: memory type string, default "general".
        state: lifecycle state, default active.
        cognitive_type: cognitive type (experience/world/opinion), default world.
        source_ids: back-reference id list into the L0 raw layer (keeps the
            "check the original wording" ability).
        entity_ids: associated entity id list (used by the index layer).
        superseded_by: if superseded, the id of the superseding memory.
        last_recalled_at: timestamp of the most recent recall.
        pin: whether pinned (exempt from time decay). Automatically true
            when importance>=0.7.
        source: source label (file name / relative path / session id).
            In bulk-import scenarios this is "which md file this memory came
            from", letting recall hits point straight back at the original
            document (coverage report, High 3: previously only the L0 id was
            stored, the original file could not be traced, and per-file
            audit and incremental re-import were both impossible).
        tags: structured tags (zcode requirement A1). Retrieval in a
            strongly-constrained writing project is mostly **conditional
            retrieval**: by chapter number, by year/month, by single source
            file -- semantic matching alone is a gamble. tags turn such
            queries from "guessing" into "looking up".
            Division of labor with entities: tags hold **structured hard
            constraints** (章/卷/年/文件), while characters/proper nouns
            keep using the entity inverted index -- duplicating both would
            make them fight each other.
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
    # Retrieval strength: accumulated when successfully recalled (into the
    # result set) or reinforced by replay; scoring weights it by
    # +w_strength·log1p(strength) -- a minimal implementation of the
    # biological "retrieval as consolidation" (testing effect). For the
    # division of labor with recall_count see engine/manager.reinforce's
    # docstring: recall_count tracks the candidate set (used by the
    # diversity penalty), this field tracks the result set.
    recall_strength: float = 0.0
    pin: bool = False
    source: str = ""
    tags: dict[str, Any] = field(default_factory=dict)
    # D2a write-side quarantine: True = pending review (direct writes from a
    # non-home workspace via the user exemption channel). Excluded from the
    # recall pool (recall_flow) and from the user_scope slot
    # (user_scope.select); after drift_audit review, promoted via the
    # managed channel (single source of truth for the criterion:
    # domain/write_quarantine.should_quarantine). Default False = zero
    # change for existing libraries; serialization is **sparse** (False
    # emits no key), so loading old libraries needs zero migration.
    quarantine: bool = False
    # D1 privacy labeling: empty string = unlabeled (existing-library state);
    # "personal" / "sensitive:category1+category2". Labeling is a hint, not
    # enforcement (misses are the norm, see
    # domain/sensitive.classify_sensitivity docstring); consumers are the
    # D2b read-side masked-tier filtering and drift_audit review grouping.
    sensitivity: str = ""
    # P0-1 write credentials (Super-agent adaptation): {session, task, surface},
    # built by common/write_origin.build_origin at the tool boundary and
    # carried here verbatim. **Write-only, never filtered on** -- recall does
    # not read it (first accumulate the real library's value distribution,
    # then wire up retrieval filtering; changing the write and retrieval
    # paths at the same time is the lesson of phase G). Sparse serialization:
    # an empty dict emits no key, existing cards stay byte-identical.
    origin: dict[str, str] = field(default_factory=dict)
    # P0-2 idempotency key: the host's declaration that "this is the same
    # request" (retry / replay / re-run after resume). The source of truth
    # is store/idem.py's in-process inverted index; this field is what that
    # index is **rebuilt from after persisting to disk** -- the index is a
    # derivative, no separate sidecar file. Empty string = no idempotency
    # declared (default).
    idem: str = ""
    # Provenance of the pin: "auto" (importance threshold), "manual"
    # (caller supplied pin=True), "legacy" (record written before this
    # field existed), or "" (not pinned).  Declared LAST so positional
    # construction of existing fields is unchanged.  "legacy" is never
    # serialized, keeping old cards byte-identical on disk.
    pin_source: str = ""
    # V1 repair provenance: L0 raw ids this record CITED but which were lost
    # before ever persisting (the 2026-11-06 dangling audit: L1 rows landed
    # while their raw did not).  Set ONLY by scripts/repair_l0_dangling.py
    # when it strips dead references, so a record carrying lost_sources reads
    # as "was traceable, provenance lost" — distinct from records that never
    # had sources.  Sparse serialization (empty list never emits the key):
    # existing cards stay byte-identical.  Declared LAST.
    lost_sources: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # importance >= PIN_IMPORTANCE pins automatically (bigeye: importance>=7 does not decay)
        if self.importance >= PIN_IMPORTANCE:
            self.pin = True
        if self.pin_source != "legacy":
            if self.pin and not self.pin_source:
                self.pin_source = (
                    "auto" if self.importance >= PIN_IMPORTANCE else "manual")
            if not self.pin:
                self.pin_source = ""
        # H8 double-storage fix: every construction path (from_dict / direct
        # construction / tests) normalizes uniformly to array('f') --
        # converted only here and in set_vector; consumers (np math / json /
        # npz / sqlite blob) notice nothing.
        if self.embeddings:
            self.embeddings = {
                k: _as_f32_vector(v) for k, v in self.embeddings.items()
            }

    def sync_pin(self) -> None:
        """Recompute ``pin`` from the current ``importance``.

        ``__post_init__`` only fires on **construction**, while
        ``VectorStore.update`` goes through ``setattr`` and bypasses it --
        so "raising importance pins automatically" only held on the creation
        path; when the merge (dedup) path raised importance, pin stayed
        False (defect D4). Any place that changes importance should call
        this method (except when setting pin explicitly).
        """
        self.pin = self.importance >= PIN_IMPORTANCE
        if self.pin and self.pin_source in ("", "legacy"):
            self.pin_source = "auto"
        elif not self.pin:
            self.pin_source = ""

    def is_live(self) -> bool:
        """Whether it participates in default recall (archived does not)."""
        return self.state != MemoryState.ARCHIVED.value

    # ------------------------------------------------------------------
    # Vector read/write (keyed by model key, model-agnostic)
    # ------------------------------------------------------------------
    def vector_for(self, model_key: str) -> array | None:
        """Get the vector for the given model (``array('f')``; ``np.asarray`` takes it directly).

        Returns ``None`` if that model has none yet (triggering a recompute).
        """
        return self.embeddings.get(model_key)

    def has_vector(self, model_key: str) -> bool:
        """Whether a vector for the given model already exists."""
        return model_key in self.embeddings

    def set_vector(self, model_key: str, vector: Any) -> None:
        """Write / overwrite the vector for the given model (normalized uniformly to ``array('f')``)."""
        self.embeddings[model_key] = _as_f32_vector(vector)

    def drop_vectors(self, keep: set[str]) -> int:
        """Drop model vectors not in ``keep``, returning how many were dropped."""
        drop = [k for k in self.embeddings if k not in keep]
        for k in drop:
            del self.embeddings[k]
        return len(drop)

    def model_keys(self) -> list[str]:
        """Return the list of model keys this memory already has vectors for."""
        return list(self.embeddings.keys())

    def to_dict(self, with_embeddings: bool = False) -> dict[str, Any]:
        """Serialize the record into a JSON-friendly dict.

        Args:
            with_embeddings: whether to write the vectors into the JSON as
                well. Default ``False`` -- vectors are large (768-dim
                float64 is about 15KB per record after serialization); they
                are uniformly written by ``VectorStore`` into a binary .npz
                file.
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
        # D2a/D1: both fields serialize **sparsely** -- default values emit no
        # key, so existing cards stay byte-identical; only cards actually
        # quarantined/labeled gain these two keys. Older code reading the
        # extra keys is also safe (from_dict takes defaults for missing keys
        # and never reads unknown ones).
        if self.quarantine:
            data["quarantine"] = True
        if self.sensitivity:
            data["sensitivity"] = self.sensitivity
        # P0-1/P0-2 likewise sparse: existing cards gain no key, and old code reading the extra keys ignores them
        if self.origin:
            data["origin"] = dict(self.origin)
        if self.idem:
            data["idem"] = self.idem
        if self.pin_source and self.pin_source != "legacy":
            data["pin_source"] = self.pin_source
        if self.lost_sources:
            data["lost_sources"] = list(self.lost_sources)
        if with_embeddings:
            data["embeddings"] = {k: list(v) for k, v in self.embeddings.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        """Build a ``MemoryRecord`` from a deserialized dict.

        Compatible with three historical shapes:
            - v0.4: ``embeddings`` as a multi-model dict;
            - v0.3 and earlier: a single ``embedding`` list (filed into the
              ``__legacy__`` slot);
            - no vector field at all (injected separately by VectorStore
              from .npz).
        Missing optional fields take default values, preserving forward
        compatibility.
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
            # Old libraries / partial data lack this field → 0.0 (no strength, same state as a new record)
            recall_strength=float(data.get("recall_strength") or 0.0),
            pin=bool(data.get("pin", False)),
            source=str(data.get("source") or ""),
            # Old libraries lack this field → empty dict, matching no tag condition when filtering (safe default)
            tags=dict(data["tags"]) if isinstance(data.get("tags"), dict) else {},
            # D2a/D1: cards written by old libraries/old code lack these keys → False/"" (same state as creation defaults)
            quarantine=bool(data.get("quarantine", False)),
            sensitivity=str(data.get("sensitivity") or ""),
            # P0-1/P0-2: old libraries lack these keys → empty dict / empty string (same state as creation defaults)
            origin=(
                {str(k): str(v) for k, v in data["origin"].items()}
                if isinstance(data.get("origin"), dict) else {}
            ),
            idem=str(data.get("idem") or ""),
            pin_source=str(data.get("pin_source") or "legacy"),
            lost_sources=[str(s) for s in (data.get("lost_sources") or [])],
        )
