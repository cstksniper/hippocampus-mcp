# -*- coding: utf-8 -*-
"""Composition-root factories (the implementation body of server.py's
composition root block).

Implemented here: the cross-library use-case port shim ``_LIB_PORTS``, the
6 application use-case factories (``_write_service`` / ``_bootstrap_service``
/ ``_recall_service`` / ``_organize_service`` / ``_maintain_service`` /
``_library_admin_service``), and the brain constructor ``_build_brain``.

🔴 Every input of the composition root is **fetched live via
:func:`_srv`**:

* the factories go through ``s._brain_for`` / ``s._library_self`` -- after
  a test replaces ``server._brain_for``, the factory gets the replacement
  brain (the knowledge_tool_face trick must keep working all the way into
  the factories);
* every port of ``_LIB_PORTS`` is a **live reference** of the form
  ``lambda *a, **k: _srv().X(*a, **k)`` -- not an import-time snapshot; the
  server namespace is resolved only when the port is called;
* ``_build_brain`` resolves the class via ``s._brain_class()``
  (``globals().get("HippoBrain")`` semantics, so a ``server.HippoBrain =
  Fake`` replacement takes effect), falls back via ``s._prepare_torch()``,
  and fetches the config constants (``_LAMBDA_DECAY`` etc.) live one by one;
* ``probe=s.warmup_status`` is likewise fetched live at construction.

Stays in server.py: ``_brain`` / ``get_brain`` / ``_brain_for`` /
``_manager_for`` / ``_brain_class`` / the ``_registry`` trio /
``_PERSIST_DIR`` and all config constants (the patch surface + composition
root semantics). server.py re-exports this module's symbols verbatim by
import name -- the ``hasattr(S, "_write_service" / "_maintain_service" /
"_LIB_PORTS")`` checks of tests/test_m4_batch1_tool_face.py /
tests/test_m4_batch4_tool_face.py and the tool-body call sites change by
zero lines.

Home-library write guard: this module's composition-root factories are
where the write-class tools' **home-library write boundary** lands
(``_guard_write_boundary``, comparing the resolved target vs the
connection's default library -- writes are always pinned to the home
library, cross-library writes keep only the ``user`` managed channel); the
live-fetch pattern already provides every hook (``s.default_library_path``
stays in server, ``session_key`` goes into routing); keeping the single
"tool → factory" call path intact is all that is needed.
The composition root does **not import domain** (layer whitelist): the pure
functions and symbols needed for decisions are all fetched live through the
``server`` namespace -- the same trick as injecting the second brain for
cross-library reads.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from hippocampus_mcp.application import (BootstrapService, LibraryAdminService,
                                         LibraryPorts, MaintainService,
                                         OrganizeService, RecallService,
                                         WriteService)

if TYPE_CHECKING:  # for static checks only (annotations already stringified by the future import)
    from hippocampus_mcp.brain import HippoBrain

logger = logging.getLogger("hippocampus_mcp")


def _srv() -> Any:
    """The server module itself (the live-fetch entry point; see the
    live-fetch list in the module docstring)."""
    from hippocampus_mcp import server as _s

    return _s


#: Port shim for the cross-library use cases (stateless; one process-global
#: copy is enough). Every port is a **live reference** via :func:`_srv` --
#: the server namespace is resolved only when called, so test replacements
#: of ``server._brain_for`` / ``server._registry_get`` etc. still pass
#: through the ports (an import-time snapshot would silently void patches).
_LIB_PORTS = LibraryPorts(
    libraries=lambda *a, **k: _srv().libraries(*a, **k),
    default_path=lambda *a, **k: _srv().default_library_path(*a, **k),
    resolve_path=lambda *a, **k: _srv().resolve_library_path(*a, **k),
    binding_get=lambda *a, **k: _srv().get_library_binding(*a, **k),
    binding_set=lambda *a, **k: _srv().set_library_binding(*a, **k),
    brain_for=lambda *a, **k: _srv()._brain_for(*a, **k),
    registry_stats=lambda *a, **k: _srv()._registry_get().stats(*a, **k),
    ensure_dir=lambda *a, **k: _srv()._ensure_default_dir(*a, **k),
    layer_files=lambda *a, **k: _srv()._layer_files(*a, **k),
    prefetch=lambda *a, **k: _srv()._prefetch_library(*a, **k),
    # Task directory registration (managed channel: the only write entry
    # into the user library's sidecar). Implemented in server.py (it has to
    # resolve the user library under the reserved name ``user`` + write the
    # sidecar); still a live reference -- replacing
    # ``server._task_directory_register`` in tests takes effect immediately.
    task_directory_register=lambda *a, **k: _srv()._task_directory_register(*a, **k),
)


# ------------------------------------------------------------------
# application use-case factories (composition root): the single call path
# from tools to services
# ------------------------------------------------------------------
# Grounding of the "single call path" discipline: tools no longer call
# ``_manager_for`` / ``_brain_for`` directly, they get use-case services
# only through these factories. Each factory does exactly the composition
# root's three things: resolve the target library (_brain_for, registry
# cache), compute the library self-exposure (_library_self, TTL-cached
# scan), inject config values (_PERSIST_DIR, read once at module import).
# Built fresh per call -- brains are cached by path in the registry and the
# self-exposure goes through the cached scan, the same cost as the current
# tools' per-call _brain_for, no new state.
# (All three inputs are fetched live via _srv().)
def _sensitive_words() -> tuple[str, ...]:
    """Read the deployed sensitive-word list: one word per line; ``#``
    comment lines and blank lines are skipped.

    **Re-read on every call** (same semantics as load_config not caching:
    editing the word list takes effect immediately); a missing/unreadable
    file = empty word list + warning (the pattern-level built-in fallback
    is still there -- the second gate of the sensitive-word red line does
    not depend on the list existing). The application layer must not import
    config (layer dependency discipline), so the read point lands in the
    composition root -- this module is the composition root.
    """
    path = _srv().load_config().sensitive_words
    if not path:
        return ()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError as exc:
        logger.warning("sensitive-word list %s is unreadable (%s): this process runs pattern-level built-in fallback only.",
                       path, exc)
        return ()
    return tuple(ln.strip() for ln in lines
                 if ln.strip() and not ln.strip().startswith("#"))


class _UsageOff:
    """Stand-in config read when ``usage_recency`` is off: the UR numeric
    sub-knobs read zero. Returned by :func:`_usage_cfg` so the factories
    can inject masked values unconditionally (B-type allow-list promise:
    numeric envs alone must not enable the usage item)."""

    usage_recency_w = 0.0
    usage_recency_tau_days = 0.0
    usage_recency_cap_strength = 0


def _usage_cfg(s: Any) -> Any:
    """UR usage-recency config view, MASKED by the ``usage_recency`` bool.

    Read live (``s.load_config()``, same no-cache semantics as every other
    knob). With the bool false, the numeric sub-knobs resolve to zero on
    the returned stand-in -- setting only
    ``HIPPOCAMPUS_USAGE_RECENCY_W`` (etc.) must not enable the item.
    """
    cfg = s.load_config()
    return cfg if cfg.usage_recency else _UsageOff()


class _AdaptiveView:
    """Normalized adaptive config view for both factories (service + brain
    shell read ``adaptive_enabled`` + numerics off one shape).

    Since 2026-11-08 the default is ON (Config.adaptive_recall=True), so
    returning the raw ``Config`` no longer works — it never had an
    ``adaptive_enabled`` attribute (that name belongs to the service ctor).
    Explicit-off (env 0/false) maps to enabled=False AND max_k=0 so neither
    factory can route to the selector; explicit-on passes the numerics
    through unchanged."""

    def __init__(self, cfg: Any) -> None:
        enabled = bool(cfg.adaptive_recall)
        self.adaptive_enabled = enabled
        self.adaptive_max_k = cfg.adaptive_max_k if enabled else 0
        self.adaptive_expand_floor = cfg.adaptive_expand_floor if enabled else 0.6
        self.adaptive_tau_stop = cfg.adaptive_tau_stop if enabled else 0.1
        self.adaptive_lambda = cfg.adaptive_lambda if enabled else 0.7
        self.adaptive_beta = cfg.adaptive_beta if enabled else 0.15
        self.adaptive_gamma = cfg.adaptive_gamma if enabled else 0.1


def _adaptive_cfg(s: Any) -> Any:
    """Adaptive result-count config view, MASKED by ``adaptive_recall``.

    Same no-cache live read as ``_usage_cfg``. Explicit 0/false forces
    ``adaptive_enabled=False`` AND ``adaptive_max_k=0`` so neither the
    service nor the brain shell can route to the selector.
    """
    return _AdaptiveView(s.load_config())


def _int_or_zero(raw: str) -> int:
    """Parse a config string into a non-negative int, **0 when
    unparseable** (= no truncation).

    "A wrong setting equals no setting" is the default semantics of this
    project's knobs (same as ``long_content`` / ``onnx_no_arena``): an
    unintelligible value must never silently open the guard and cut long
    memories without anyone knowing -- that is far worse than not saving
    those few tokens.
    """
    try:
        v = int(str(raw or "").strip())
    except (TypeError, ValueError):
        return 0
    return v if v > 0 else 0


def _guard_write_boundary(library: str | None = None) -> None:
    """The **cross-library write guard** for write-class tools.

    Semantics in one sentence: **``library_use`` only affects where reads
    land; writes are always pinned to the home library**. This is a
    deliberate tightening of the old semantics ("writes follow the sticky
    library"); the corresponding behavior assertions were updated in sync.

    Every symbol used here is fetched live via :func:`_srv`:
    ``caller_tool_name`` / ``WRITE_CLASS_TOOLS`` / ``load_config`` /
    ``_library_self`` / ``default_library_path`` / ``libraries`` / the
    decision trio (``classify_write_permission`` / ``same_library`` /
    ``library_display``) -- the composition root does **not import domain**
    (layer whitelist), and test replacements of this namespace still pass
    through the guard (replacing ``server.default_library_path`` is exactly
    how the "home library" gets swapped).

    Raises:
        LibraryResolutionError: ambiguous resolution (raised upstream by
            ``_library_self``: with the target library undecidable there is
            nothing to decide permissions about).
        CrossLibraryWriteDenied: the target is neither the home library nor
            the user managed channel.
    """
    s = _srv()
    # depth=3: caller_tool_name itself → this guard → the use-case factory
    # → the tool body. Writing 2 would pick up the factory name
    # (``_write_service``), which is not in the table ⇒ the guard passes
    # silently.
    tool = s.caller_tool_name(3)
    if tool not in s.WRITE_CLASS_TOOLS:
        return  # read-class tools / non-tool calls: cross-library reads are open
    if s.load_config().cross_library_write:
        logger.warning(
            "Escape hatch HIPPOCAMPUS_CROSS_LIBRARY_WRITE=1 active: cross-library writes to %s "
            "are open in this process (home-library write boundary skipped) -- remove this env after the migration is done.", tool)
        return
    info = s._library_self(library)  # {id, name, path}
    home_path = s.default_library_path()
    perm = s.classify_write_permission(
        target_path=info.get("path", ""), home_path=home_path,
        target_name=info.get("name", ""))
    if perm == s.READ_ONLY:
        home_name = ""
        for cand in s.libraries(False):
            if s.same_library(cand.path, home_path):
                home_name = cand.name
                break
        raise s.CrossLibraryWriteDenied(s.boundary_error_message(
            tool=tool,
            target_display=s.library_display(info.get("name", ""),
                                             info.get("path", "")),
            home_display=s.library_display(home_name, home_path)))


def _write_service(library: str | None = None) -> WriteService:
    """Use-case service for the 9 tools of the Write surface
    (store/update/tags/forget/ingest/save)."""
    _guard_write_boundary(library)
    s = _srv()
    brain = s._brain_for(library)
    info = s._library_self(library)
    # D2a write-side isolation trio (pure criteria in domain/write_quarantine,
    # fetched live through the server namespace -- the S4.2 whitelist does
    # not allow a domain import, same trick as the write guard):
    # a direct write from a non-home workspace via the user exemption
    # channel → the review queue (decision ①a);
    # home library = the user library's daily host writing its own home
    # → no isolation (the full-tier semantics of decision ③);
    # escape hatch open = an explicit ops window → no isolation but still
    # audit-logged.
    home_path = s.default_library_path()
    escape_open = bool(s.load_config().cross_library_write)
    same_home = s.same_library(info.get("path", ""), home_path)
    target_name = info.get("name", "")
    # Narrowing (2026-10-31 decision, migration doc §六): "家" (home) is no
    # longer inferred from fallback -- under a user-level install, the
    # fallback same_home of unbound/ambiguous/corrupt-directory states does
    # **not count as home** (under the old criterion an unbound workspace
    # falling back to the user library was judged "家" → the hole of full
    # read/write and non-isolated writes).
    # Genuine home is one of exactly two shapes: no workspace declaration
    # (no_cwd/undeclared) and same_home (A1 daily host, explicit single-
    # library deployment); or bound with the entry explicitly declaring
    # full (elevation must be explicit, same criterion as the read-side
    # full tier -- full tier = authorized to treat the user library as
    # one's own home).
    brief = s._workspace_binding_brief()
    status = str(brief.get("status") or "undeclared")
    entry_access = str(brief.get("access") or "")
    genuine_home = (
        (same_home and status in ("no_cwd", "undeclared"))
        or (status == "bound" and entry_access == "full"))
    quarantine = s.should_quarantine(
        target_name=target_name, same_as_home=genuine_home,
        escape_open=escape_open)
    return WriteService(brain=brain,
                        library_self=info,
                        persist_dir=s._PERSIST_DIR,
                        sensitive_words=_sensitive_words(),
                        # long-content policy: read live (config edits take
                        # effect immediately, same as sensitive_words).
                        long_content_policy=s.load_config().long_content,
                        # the sentence splitter itself (including the "short
                        # sentence merged into the previous one" quirk), not
                        # a second implementation.
                        split_facts=getattr(brain, "_split_facts", None),
                        user_write_quarantine=quarantine,
                        is_home_write=genuine_home or escape_open,
                        access_log_dir=(info.get("path", "")
                                        if (target_name or "").strip() == "user"
                                        else ""))


def _bootstrap_service(library: str | None = None) -> BootstrapService:
    """Use-case service for the 5 tools of the Bootstrap surface
    (session_start/end, intents, feedback, stats).

    Write-class members (session_end / remember_intent /
    complete_intent / report_feedback) are gated via
    :func:`_guard_write_boundary`; read-class members (session_start /
    get_stats) are unrestricted across libraries.
    """
    _guard_write_boundary(library)
    s = _srv()
    return BootstrapService(brain=s._brain_for(library),
                            library_self=s._library_self(library),
                            persist_dir=s._PERSIST_DIR)


def _spec_parse(spec: str):
    """Parse the libraries= syntax (B1): the domain truth is fetched live
    through the server namespace -- the composition root does not import
    domain (S4.2 whitelist), the same trick as the write guard / isolation
    criteria."""
    return _srv().parse_library_spec(spec)


def _names_outside_directory(s: Any, names: list[str]) -> list[str]:
    """Soft directory check (B1/Q-D): a value not in the task directory →
    return the list for the envelope to disclose, do not block.

    The directory is navigation, not a gate -- old/unregistered libraries
    must not become unqueryable because of it (the realization of "not
    silent" is **visible**, not "unavailable"). If the directory cannot be
    read (user library unreachable / sidecar broken) → empty list with
    nothing flagged; the check itself is a freebie.
    """
    try:
        info = s._library.resolve_library("user", libs=s.libraries())
        data, issue = s._task_directory.load_directory(info.path)
        if issue:
            return []
        entries = data.get("entries") or []
        known: set[str] = set()
        for e in entries:
            if not isinstance(e, dict):
                continue
            for key in ("task", "name"):
                v = str(e.get(key) or "").strip().casefold()
                if v:
                    known.add(v)
        return [n for n in names if n.strip().casefold() not in known]
    except Exception:  # noqa: BLE001 -- the check is a freebie
        return []


def _recall_service(library: str | None = None,
                    scope: str = "project",
                    libraries_spec: str = "") -> RecallService:
    """Use-case service for the 11 tools of the Recall surface (recall/get/
    list/context/contradictions/drift/entity/map/topic).

    With ``scope="user"/"all"``, resolves the user-level library under the
    reserved name ``user`` and injects a second brain via the registry
    (an explicit retrieval intent ⇒ the first visit to the user library
    pays the full brain load cost itself. The registry residency cap is 3:
    home library + another library + user library is the normal cross-task
    read combination; the project+user pair of course co-resides).
    On resolution failure **raises LibraryResolutionError** (an explicit
    retrieval that cannot find the library must error out, not silently
    fall back to project -- same precedent as the ``library`` argument).
    Value validation happens inside the service (a domain pure function;
    the error body carries no self-exposure keys).

    B1 multi-library (decision ②a): ``libraries=`` is the explicit
    multi-library channel and is **mutually exclusive** with ``library=`` /
    ``scope=user|all`` (an explicit error is raised here); each value is
    resolved by three-state addressing (an unknown value errors with the
    value itself), and ``(library_self, brain)`` is injected per library.
    Soft directory check: a value not in the task directory stays usable,
    its name goes into ``unregistered_names`` for the envelope to disclose
    (the directory is navigation, not a gate -- old/unregistered libraries
    must not become unqueryable).

    Composition-root rule: ``libraries`` / ``_registry_get`` /
    ``resolve_library`` are all fetched live via :func:`_srv` or module
    attributes (test replacements of ``server.libraries`` /
    ``server._registry_get`` still pass through).

    When a single ``library=`` hits another library, a background warmup is
    kicked off (``maybe_warmup_library``) -- with explicit cross-library
    reads normalized, 25–33 seconds should not be paid by the first call.
    Failure does not block (the same semantics as the prefetch learned from
    ``library_use``), and it only triggers when the target is not the home
    library (the home library should already be resident).
    """
    s = _srv()
    extra: list[tuple[dict, Any]] = []
    unregistered: list[str] = []
    spec_names, spec_err = _spec_parse(libraries_spec)
    if spec_err:
        raise s.LibraryResolutionError(spec_err)
    if spec_names and (library or "").strip():
        raise s.LibraryResolutionError(
            "libraries= and library= are mutually exclusive: library= picks the main retrieval library, "
            "libraries= is the explicit multi-library fan-out; their combination has no predictable "
            "semantics. For multi-library pass only libraries= (the home library is the current default), "
            "for single library pass only library=.")
    if spec_names:
        for name in spec_names:
            path = s.resolve_library_path(name)
            if not s.same_library(path, s.default_library_path()):
                # Warmup is an accelerator, not a prerequisite: failure only
                # logs a warning; this call proceeds as usual (§4.1).
                try:
                    s.maybe_warmup_library(path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("cross-library read prefetch %s failed (this call is not blocked): %s",
                                   path, exc)
            info = s._library_self(name)
            extra.append((info, s._brain_for(name)))
        unregistered = _names_outside_directory(s, spec_names)
    if (library or "").strip():
        path = s.resolve_library_path(library)
        if not s.same_library(path, s.default_library_path()):
            # Warmup is an accelerator, not a prerequisite: failure only
            # logs a warning; this call proceeds as usual (§4.1).
            try:
                s.maybe_warmup_library(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("cross-library read prefetch %s failed (this call is not blocked): %s",
                               path, exc)
    user_brain = None
    user_self: dict[str, str] | None = None
    # D2b: the target of the tier decision is the **user library** -- the
    # scope=user/all stream targets the user library (the service's
    # library_self is still the project library), and the libraries=
    # periphery may also include it. Missing this makes the masked/none
    # tiers always empty (a wiring bug caught by E2E testing).
    user_tier_info: dict[str, str] | None = None
    if scope in ("user", "all") or spec_names:
        info = s._library.resolve_library("user", libs=s.libraries())
        user_self = {"id": info.library_id, "name": info.name,
                     "path": info.path}
        user_tier_info = user_self
        if scope in ("user", "all"):
            user_brain = s._registry_get().get(info.path)
    # D2b read-side tier resolver: computed per client (in the multi-library
    # daemon each connection has its own binding). Decision target = the
    # user library (the stream's target); in a non-user-library context
    # (a bare project stream with no user in the periphery) returns ""
    # (no tier).
    target_info = s._library_self(library)
    access_resolver = (lambda client="": s._user_access_tier(
        user_tier_info or target_info, client))
    return RecallService(brain=s._brain_for(library),
                         library_self=target_info,
                         persist_dir=s._PERSIST_DIR,
                         user_brain=user_brain,
                         user_library_self=user_self,
                         user_access_resolver=access_resolver,
                         # B1 multi-library: the periphery
                         # [(library_self, brain)] (already resolved) +
                         # the list of out-of-directory values (soft
                         # check, disclosed by the envelope).
                         extra_libraries=extra,
                         unregistered_names=unregistered,
                         # prospective-memory contextual trigger switch,
                         # read live and injected (application does not
                         # import config -- the same rule as
                         # long_content_policy; config edits take effect
                         # immediately).
                         prospective_trigger=s.load_config().prospective_trigger,
                         # per-record recall content truncation guard
                         # (0 = no truncation). Invalid/negative values
                         # are all treated as 0 (no truncation) -- a wrong
                         # setting equals no setting; never let the guard
                         # be silently opened. Parsing happens in
                         # BaseService.
                         recall_content_max_chars=_int_or_zero(
                             s.load_config().recall_content_max_chars),
                         # R1: shared scoring pipeline for production
                         # recall. Default on; HIPPOCAMPUS_RECALL_FUSION=0
                         # restores the legacy additive manager.recall.
                         recall_fusion_enabled=(
                             s.load_config().recall_fusion),
                         # R1-B4: MMR dimension-scale fix (semantic A,
                         # default on -- injected live like recall_fusion;
                         # both RecallDeps construction sites must stay in
                         # lockstep).
                         mmr_scale_fix=s.load_config().mmr_scale_fix,
                         # V2: query fanout (keyword variant + RRF merge).
                         # Default off (evaluation reproducibility);
                         # HIPPOCAMPUS_RECALL_QUERY_FANOUT=1 enables.
                         recall_query_fanout_enabled=(
                             s.load_config().recall_query_fanout),
                         # UR: usage-recency scoring. The numeric sub-knobs
                         # are MASKED by the usage_recency bool here --
                         # setting only HIPPOCAMPUS_USAGE_RECENCY_W (etc.)
                         # must NOT enable the item (B-type allow-list
                         # promise: only an explicit truthy bool opens it).
                         usage_recency_w=_usage_cfg(s).usage_recency_w,
                         usage_recency_tau_days=_usage_cfg(
                             s).usage_recency_tau_days,
                         usage_recency_cap_strength=_usage_cfg(
                             s).usage_recency_cap_strength,
                         # Adaptive result-count selection (2026-11-07):
                         # masked by the adaptive_recall bool, same B-type
                         # promise as UR -- numeric envs alone stay off.
                         adaptive_enabled=_adaptive_cfg(s).adaptive_enabled,
                         adaptive_expand_floor=_adaptive_cfg(
                             s).adaptive_expand_floor,
                         adaptive_tau_stop=_adaptive_cfg(s).adaptive_tau_stop,
                         adaptive_lambda=_adaptive_cfg(s).adaptive_lambda,
                         adaptive_beta=_adaptive_cfg(s).adaptive_beta,
                         adaptive_gamma=_adaptive_cfg(s).adaptive_gamma,
                         adaptive_max_k=_adaptive_cfg(s).adaptive_max_k)


def _organize_service(library: str | None = None) -> OrganizeService:
    """Use-case service for the 10 tools of the Organize surface
    (consolidate/relink/the lexicon trio/the map quartet/reset_context).

    Write-class members (consolidate / relink_entities /
    lexicon_add / map_set / reset_context) are gated via
    :func:`_guard_write_boundary`; read-class members (map_get / map_list /
    map_verify / lexicon_suggest / lexicon_alias_suggest) are unrestricted
    across libraries.
    """
    _guard_write_boundary(library)
    s = _srv()
    return OrganizeService(brain=s._brain_for(library),
                           library_self=s._library_self(library),
                           persist_dir=s._PERSIST_DIR,
                           sensitive_words=_sensitive_words(),
                           promote_min_sessions=s.load_config()
                           .promote_min_sessions)


def _maintain_service(library: str | None = None) -> MaintainService:
    """Use-case service for the 5 tools of the Maintain surface
    (export/miss_log/stats/save/load).

    What ``probe`` injects is :func:`warmup_status` (implemented in
    ``server_runtime.warmup``, fetched via import; the call shape
    ``warmup_status()`` is unchanged) -- the ``warmup`` section of
    ``brain_stats`` is facility status, not library facts; letting it in
    through a constructor argument is what keeps the dependency visible.

    Write-class members (save_brain / load_state / miss_log) are gated via
    :func:`_guard_write_boundary`; read-class members (export_memories /
    brain_stats) are unrestricted across libraries. Persistence-class
    operations are no less dangerous than content writes (design §2.5:
    explicit persistence against a half-written library state is just as
    dangerous), so they are pinned down together.
    """
    _guard_write_boundary(library)
    s = _srv()
    info = s._library_self(library)
    return MaintainService(brain=s._brain_for(library),
                           library_self=info,
                           persist_dir=s._PERSIST_DIR, probe=s.warmup_status,
                           # D2b: the read-side tier gate for user-library
                           # exports (consumed by export refusals).
                           user_access_resolver=lambda client="":
                           s._user_access_tier(info, client))


# Note: ``_library_admin_service`` **stays in server.py** --
# tests/test_m4_batch4_tool_face.py pins its function body in the server.py
# source via ast.get_source_segment (the structural contract of "no brain
# passed / no _brain_for hop": LibraryAdminService is not bound to a single
# brain). Project discipline is to adapt code to tests, never edit tests;
# the other five factories have no source-level pins and are collected into
# this module along with the composition root as normal.
def _build_brain(persist_dir: str) -> HippoBrain:
    """Build a HippoBrain (**one per library**; the registry guarantees the
    same path is only ever built once).

    The logic is bit-for-bit identical to the pre-multi-library
    ``get_brain()`` (compatibility requirement); the only difference is
    that ``persist_dir`` became a parameter.

    Class resolution (``s._brain_class()``, ``globals().get("HippoBrain")``
    semantics), the torch fallback, and all config constants are fetched
    live via :func:`_srv`.
    """
    s = _srv()
    s._prepare_torch()
    logger.info("initializing HippoBrain (persist_dir=%s)", persist_dir)
    brain = s._brain_class()(
        persist_dir=persist_dir,
        lambda_decay=s._LAMBDA_DECAY,
        embedder_model=s._EMB_MODEL,
        auto_save=s._AUTO_SAVE,
        mu_importance=s._MU_IMPORTANCE,
        w_sim=s._W_SIM,
        w_time=s._W_TIME,
        w_recall=s._W_RECALL,
        supersede_enabled=s._SUPERSEDE,
        auto_backfill=s._AUTO_BACKFILL,
        nav_penalty=s._NAV_PENALTY,
        recency_window_days=s.load_config().recency_window_days,
        w_strength=s.load_config().w_strength,
        mem_history_ring=s.load_config().mem_history_ring,
        # UR: usage-recency scoring, masked by the usage_recency bool (same
        # B-type promise as the RecallService factory above -- numeric envs
        # alone stay off). Must stay in lockstep with the service factory:
        # both RecallDeps construction sites (brain shell + RecallService)
        # need the same values or the two paths diverge when the knob is on.
        usage_recency_w=_usage_cfg(s).usage_recency_w,
        usage_recency_tau_days=_usage_cfg(s).usage_recency_tau_days,
        usage_recency_cap_strength=_usage_cfg(s).usage_recency_cap_strength,
        # Adaptive result-count selection: same masking + lockstep
        # discipline as the RecallService factory above (both RecallDeps
        # construction sites must match when the knob is on).
        adaptive_enabled=_adaptive_cfg(s).adaptive_enabled,
        adaptive_expand_floor=_adaptive_cfg(s).adaptive_expand_floor,
        adaptive_tau_stop=_adaptive_cfg(s).adaptive_tau_stop,
        adaptive_lambda=_adaptive_cfg(s).adaptive_lambda,
        adaptive_beta=_adaptive_cfg(s).adaptive_beta,
        adaptive_gamma=_adaptive_cfg(s).adaptive_gamma,
        adaptive_max_k=_adaptive_cfg(s).adaptive_max_k,
        # R1-B4: MMR dimension-scale fix (live read like recall_fusion;
        # both RecallDeps construction sites must stay in lockstep).
        mmr_scale_fix=s.load_config().mmr_scale_fix,
    )
    # The effective model must be visible: the trap with the wrong model
    # was that "nobody said which model was in use".
    _stats = brain.stats()
    logger.info(
        "effective embedding model: %s (vectors %s, missing %d); supersede=%s; lexicon %d entries",
        _stats["embed_model"], _stats["vectors_by_model"],
        _stats["vectors_missing"], _stats["supersede_enabled"],
        _stats["entity_lexicon_size"],
    )
    if _stats["vectors_missing"]:
        logger.warning(
            "%d records in the library lack vectors for model %s; retrieval will miss those memories. "
            "Run scripts/reembed.py --all --model %s offline first, then start the service.",
            _stats["vectors_missing"], _stats["embed_model"], _stats["embed_model"],
        )

    # Load CRL training weights (if configured)
    if s._CRL_MODEL_PATH and os.path.isfile(s._CRL_MODEL_PATH):
        if brain.manager.load_trained_weights(s._CRL_MODEL_PATH):
            logger.info("CRL training weights loaded: %s", s._CRL_MODEL_PATH)
        else:
            logger.warning("CRL weight loading failed: %s", s._CRL_MODEL_PATH)

    # Store load failures must be flagged explicitly: the corrupt file is
    # quarantined, but a human has to know
    issue = brain.manager.store.last_load_issue
    if issue:
        logger.error(
            "memory store failed to load (not silently cleared; corrupt file quarantined): %s", issue
        )
    return brain
