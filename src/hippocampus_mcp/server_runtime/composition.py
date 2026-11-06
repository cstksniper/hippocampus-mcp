# -*- coding: utf-8 -*-
"""组合根工厂（server.py 组合根区块的实现本体）。

这里实现：跨库用例端口填空 ``_LIB_PORTS``、6 个 application 用例工厂
（``_write_service`` / ``_bootstrap_service`` / ``_recall_service`` /
``_organize_service`` / ``_maintain_service`` / ``_library_admin_service``）、
brain 构造器 ``_build_brain``。

🔴 组合根的**全部输入经 :func:`_srv` 现取**：

* 工厂经 ``s._brain_for`` / ``s._library_self``——测试替换
  ``server._brain_for`` 后工厂拿到的是替 brain（knowledge_tool_face
  同款手法必须一路生效到工厂）；
* ``_LIB_PORTS`` 每条端口都是 ``lambda *a, **k: _srv().X(*a, **k)``
  的**活引用**——不是 import 时快照，端口被调用时才解析 server 命名空间；
* ``_build_brain`` 经 ``s._brain_class()`` 解析类（``globals().get(
  "HippoBrain")`` 语义，``server.HippoBrain = Fake`` 替换生效）、经
  ``s._prepare_torch()`` 兜底、配置常量（``_LAMBDA_DECAY`` 等）逐个现取；
* ``probe=s.warmup_status`` 同样在构造时现取。

留在 server.py：``_brain`` / ``get_brain`` / ``_brain_for`` /
``_manager_for`` / ``_brain_class`` / ``_registry`` 三件 / ``_PERSIST_DIR``
与全部配置常量（patch 面 + 组合根语义）。server.py 按 import 名
原样再导出本模块符号——tests/test_m4_batch1_tool_face.py /
tests/test_m4_batch4_tool_face.py 的 ``hasattr(S, "_write_service" /
"_maintain_service" / "_LIB_PORTS")`` 与工具体调用点零改动。

家库写守卫：本模块的组合根工厂是写类工具**家库写边界**的落点
（``_guard_write_boundary``，比对解析目标 vs 连接默认库——写永远钉在
家库，跨库写只留 ``user`` 受管通道）；现取模式已提供全部钩子
（``s.default_library_path`` 留 server、``session_key`` 进 routing），
「工具 → 工厂」单一调用路径保住即可。
组合根**不 import domain**（层级面白名单）：判定所需的纯函数与符号一律经
``server`` 命名空间现取，与跨库读第二 brain 的注入手法同款。
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from hippocampus_mcp.application import (BootstrapService, LibraryAdminService,
                                         LibraryPorts, MaintainService,
                                         OrganizeService, RecallService,
                                         WriteService)

if TYPE_CHECKING:  # 仅供静态检查（注解已被 future import 字符串化）
    from hippocampus_mcp.brain import HippoBrain

logger = logging.getLogger("hippocampus_mcp")


def _srv() -> Any:
    """server 模块本体（现取入口，见模块 docstring 的现取清单）。"""
    from hippocampus_mcp import server as _s

    return _s


#: 跨库用例的端口填空（无状态，进程全局一份即可）。
#: 每条端口都是经 :func:`_srv` 的**活引用**——被调用时才解析
#: server 命名空间，测试对 ``server._brain_for`` / ``server._registry_get``
#: 等的替换照样穿过端口（import 时快照会让 patch 静默失效）。
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
    # 任务目录登记（受管通道：user 库 sidecar 的唯一写入口）。
    # 实现在 server.py（要解析约定名 ``user`` 的用户库 + 写 sidecar），
    # 照样走活引用——测试替换 ``server._task_directory_register`` 即刻生效。
    task_directory_register=lambda *a, **k: _srv()._task_directory_register(*a, **k),
)


# ------------------------------------------------------------------
# application 用例工厂（组合根）：工具 → 服务的唯一调用路径
# ------------------------------------------------------------------
# 「单一调用路径」纪律的落地：工具不再直调
# ``_manager_for`` / ``_brain_for``，只经这些工厂拿用例服务。
# 工厂做且只做组合根三件事：解析目标库（_brain_for，注册表缓存）、
# 算库自曝（_library_self，TTL 缓存扫描）、注入配置值（_PERSIST_DIR，
# 模块导入时读一次）。每调用新建——brain 由注册表按路径缓存，库自曝
# 走缓存扫描，与现行工具逐个 _brain_for 同成本，无新增状态。
# （三个输入全部经 _srv() 现取。）
def _sensitive_words() -> tuple[str, ...]:
    """读部署敏感词表：一行一词，``#`` 注释行与空行跳过。

    **每次调用现读**（与 load_config 不缓存同语义：改词表立刻生效）；
    文件缺失/不可读 = 空词表 + warning（模式级内置兜底仍在，敏感词红线
    的第二道门不依赖词表存在）。application 层不许 import config
    （层级依赖纪律），所以读取点落组合根——本模块即组合根。
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


def _int_or_zero(raw: str) -> int:
    """把配置字符串解析成非负整数，**解析不出就给 0**（= 不截断）。

    「写错等同于没写」是本项目旋钮的默认语义（同 ``long_content`` /
    ``onnx_no_arena``）：一个看不懂的值绝不能把护栏悄悄打开，把长记忆
    切掉却没人知道——那比不省这点 token 糟得多。
    """
    try:
        v = int(str(raw or "").strip())
    except (TypeError, ValueError):
        return 0
    return v if v > 0 else 0


def _guard_write_boundary(library: str | None = None) -> None:
    """写类工具的**跨库写守卫**。

    语义一句话：**``library_use`` 只影响读落哪，写永远钉在家库**。
    这是对旧语义「写跟随粘性库」的有意收紧，相应行为断言已同步更新。

    用到的符号全部经 :func:`_srv` 现取：
    ``caller_tool_name`` / ``WRITE_CLASS_TOOLS`` / ``load_config`` /
    ``_library_self`` / ``default_library_path`` / ``libraries`` / 判定三件
    （``classify_write_permission`` / ``same_library`` / ``library_display``）
    ——组合根**不 import domain**（层级面白名单），测试对本命名空间的
    替换照样穿过守卫（替换 ``server.default_library_path`` 就等于换了「家库」）。

    Raises:
        LibraryResolutionError: 解析不明（``_library_self`` 上游抛：
            目标库都定不下来时谈不了权限）。
        CrossLibraryWriteDenied: 目标既非家库、也非 user 受管通道。
    """
    s = _srv()
    # depth=3：caller_tool_name 本身 → 本守卫 → 用例工厂 → 工具体。
    # 写成 2 会取到工厂名（``_write_service``），表里没有它 ⇒ 守卫静默放行。
    tool = s.caller_tool_name(3)
    if tool not in s.WRITE_CLASS_TOOLS:
        return  # 读类工具 / 非工具调用：跨库读开放
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
    """Write 面 9 工具的用例服务（store/update/tags/forget/ingest/save）。"""
    _guard_write_boundary(library)
    s = _srv()
    brain = s._brain_for(library)
    info = s._library_self(library)
    # D2a 写侧隔离三件（纯判据在 domain/write_quarantine，经 server 命名
    # 空间现取——S4.2 白名单不放 domain import，与写守卫同款手法）：
    # 非家工作区经 user 豁免通道的直写 → 待审队列（拍板①a）；
    # 家库 = user 库的日常宿主写自己家 → 不隔离（拍板③的 full 档语义）；
    # 逃生门开 = 显式运维窗口 → 不隔离但仍记审计。
    home_path = s.default_library_path()
    escape_open = bool(s.load_config().cross_library_write)
    same_home = s.same_library(info.get("path", ""), home_path)
    target_name = info.get("name", "")
    # 收窄（2026-10-31 拍板，迁移文档 §六）："家"不再由回落推断——用户级
    # 安装下 unbound/ambiguous/目录坏的回落 same_home **不算家**（旧判据下
    # 未绑定工作区回落 user 库被判"家"→ full 读写、写不隔离的洞）。
    # 真·家库形态二选一：无工作区声明（no_cwd/undeclared）且 same_home
    # （A1 日常宿主显式单库部署）；或 bound 后条目显式声明 full（提权要
    # 显式，与读侧 full 同判据——full 档 = 被授权把 user 库当自己家）。
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
                        # 超长内容策略：现读（改配置立刻生效，同 sensitive_words）。
                        long_content_policy=s.load_config().long_content,
                        # 分句器本体（含「短句并入上一条」怪癖），不另写一份。
                        split_facts=getattr(brain, "_split_facts", None),
                        user_write_quarantine=quarantine,
                        is_home_write=genuine_home or escape_open,
                        access_log_dir=(info.get("path", "")
                                        if (target_name or "").strip() == "user"
                                        else ""))


def _bootstrap_service(library: str | None = None) -> BootstrapService:
    """Bootstrap 面 5 工具的用例服务（session_start/end、意图、反馈、stats）。

    写类成员（session_end / remember_intent /
    complete_intent / report_feedback）经 :func:`_guard_write_boundary` 拦关，
    读类成员（session_start / get_stats）跨库不受限。
    """
    _guard_write_boundary(library)
    s = _srv()
    return BootstrapService(brain=s._brain_for(library),
                            library_self=s._library_self(library),
                            persist_dir=s._PERSIST_DIR)


def _spec_parse(spec: str):
    """libraries= 语法解析（B1）：domain 真相经 server 命名空间现取——
    组合根不 import domain（S4.2 白名单），与写守卫/隔离判据同款手法。"""
    return _srv().parse_library_spec(spec)


def _names_outside_directory(s: Any, names: list[str]) -> list[str]:
    """软目录校验（B1/Q-D）：值不在任务目录 → 返回名单供信封自曝，不拦截。

    目录是导航不是门禁——老库/未登记库不该因此查不了（「不静默」的
    实现是**可见**不是「不可用」）。目录读不到（user 库失联/sidecar 坏）
    → 空名单不标注，校验本身就是赠品。
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
    except Exception:  # noqa: BLE001 —— 校验是赠品
        return []


def _recall_service(library: str | None = None,
                    scope: str = "project",
                    libraries_spec: str = "") -> RecallService:
    """Recall 面 11 工具的用例服务（recall/get/list/context/contradictions/
    drift/entity/map/topic）。

    ``scope="user"/"all"`` 时解析约定名 ``user`` 的用户级库
    并经注册表注入第二 brain（显式检索意图 ⇒ 用户库首访自付完整
    brain 加载成本。注册表驻留上限 3：家库 + 他库 + 用户库是跨任务读的
    常态组合，项目+用户两库当然同驻）。
    解析失败**抛 LibraryResolutionError**（显式检索找不到库必须
    报错，不静默回落 project——同 ``library`` 参数先例）。值校验在服务内
    （domain 纯函数，错误体不带自曝键）。

    B1 多库（拍板②a）：``libraries=`` 显式多库通道，与 ``library=`` /
    ``scope=user|all`` **互斥**（这里显式报错）；每个值按三态寻址解析
    （未知值报错带值本身），逐库注入 ``(library_self, brain)``。
    软目录校验：值不在任务目录 → 照常可用，名字进 ``unregistered_names``
    由信封自曝（目录是导航不是门禁——老库/未登记库不该查不了）。

    组合根口径：``libraries`` / ``_registry_get`` / ``resolve_library``
    全部经 :func:`_srv` 或模块属性现取（测试替换 ``server.libraries``、
    ``server._registry_get`` 照样穿透）。

    单次 ``library=`` 命中他库时挂后台预热
    （``maybe_warmup_library``）——显式跨库读常态化后，25–33 秒不该由首次
    调用自付。失败不阻断（与预取学习自 ``library_use`` 的同款语义），且
    只在目标不是家库时触发（家库本就该已驻留）。
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
                # 预热是加速不是前提：失败只落 warning，本次调用照常（§4.1）。
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
            # 预热是加速不是前提：失败只落 warning，本次调用照常（§4.1）。
            try:
                s.maybe_warmup_library(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("cross-library read prefetch %s failed (this call is not blocked): %s",
                               path, exc)
    user_brain = None
    user_self: dict[str, str] | None = None
    # D2b：档位判定的对象是**用户库**——scope=user/all 的流打的是用户库
    # （服务的 library_self 仍是项目库），libraries= 的外围也可能含它。
    # 漏判会让 masked/none 档恒为空（E2E 实测抓到的接线 bug）。
    user_tier_info: dict[str, str] | None = None
    if scope in ("user", "all") or spec_names:
        info = s._library.resolve_library("user", libs=s.libraries())
        user_self = {"id": info.library_id, "name": info.name,
                     "path": info.path}
        user_tier_info = user_self
        if scope in ("user", "all"):
            user_brain = s._registry_get().get(info.path)
    # D2b 读侧档位 resolver：按 client 现算（多库 daemon 每连接各有绑定）。
    # 判定对象 = 用户库（流的目标）；非用户库语境（裸 project 流且外围无
    # user）返回 ""（无档位）。
    target_info = s._library_self(library)
    access_resolver = (lambda client="": s._user_access_tier(
        user_tier_info or target_info, client))
    return RecallService(brain=s._brain_for(library),
                         library_self=target_info,
                         persist_dir=s._PERSIST_DIR,
                         user_brain=user_brain,
                         user_library_self=user_self,
                         user_access_resolver=access_resolver,
                         # B1 多库：外围 [(library_self, brain)]（已解析）
                         # + 目录外值名单（软校验，信封自曝）。
                         extra_libraries=extra,
                         unregistered_names=unregistered,
                         # 前瞻记忆情境触发开关，现读注入
                         # （application 不 import config——同
                         # long_content_policy 口径；改配置立刻生效）。
                         prospective_trigger=s.load_config().prospective_trigger,
                         # recall 单条 content 截断护栏（0 = 不截断）。
                         # 非法值/负数一律按 0（不截断）——写错等同于没写，
                         # 绝不让护栏被悄悄打开。解析在 BaseService。
                         recall_content_max_chars=_int_or_zero(
                             s.load_config().recall_content_max_chars))


def _organize_service(library: str | None = None) -> OrganizeService:
    """Organize 面 10 工具的用例服务（consolidate/relink/lexicon 三件/
    map 四件/reset_context）。

    写类成员（consolidate / relink_entities /
    lexicon_add / map_set / reset_context）经 :func:`_guard_write_boundary`
    拦关，读类成员（map_get / map_list / map_verify / lexicon_suggest /
    lexicon_alias_suggest）跨库不受限。
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
    """Maintain 面 5 工具的用例服务（export/miss_log/stats/save/load）。

    ``probe`` 注入的是 :func:`warmup_status`（实现在
    ``server_runtime.warmup``，经 import 现取，调用形态 ``warmup_status()``
    不变）—— ``brain_stats`` 的 ``warmup`` 一节是设施状态不是库存事实，
    让它从构造参数进来，依赖才看得见。

    写类成员（save_brain / load_state / miss_log）经
    :func:`_guard_write_boundary` 拦关，读类（export_memories / brain_stats）
    跨库不受限。持久化类的危险程度不低于内容写（设计 §2.5：对半写状态的库
    做显式持久化同样危险），故一并钉死。
    """
    _guard_write_boundary(library)
    s = _srv()
    info = s._library_self(library)
    return MaintainService(brain=s._brain_for(library),
                           library_self=info,
                           persist_dir=s._PERSIST_DIR, probe=s.warmup_status,
                           # D2b：user 库导出的档位门（export 拒绝用）。
                           user_access_resolver=lambda client="":
                           s._user_access_tier(info, client))


# 注：``_library_admin_service`` **留在 server.py**——
# tests/test_m4_batch4_tool_face.py 以 ast.get_source_segment 钉住它在
# server.py 源码里的函数体（「不传 brain/不过 _brain_for」，LibraryAdminService
# 不绑定单一 brain 的结构契约）。项目纪律是改代码适配测试、不改测试；
# 其余五个工厂无源码级钉子，正常随组合根收在本模块。
def _build_brain(persist_dir: str) -> HippoBrain:
    """构造一个 HippoBrain（**每个库一份**，由注册表保证同路径只建一次）。

    逻辑与多库改造前的 ``get_brain()`` 逐位一致（兼容性要求），
    唯一差别是 ``persist_dir`` 改为入参。

    类解析（``s._brain_class()``，``globals().get("HippoBrain")``
    语义）、torch 兜底与全部配置常量经 :func:`_srv` 现取。
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
    )
    # 生效模型必须可见：导错模型的坑就是「谁都没说自己在用哪个模型」。
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

    # 加载 CRL 训练权重（如果配置了）
    if s._CRL_MODEL_PATH and os.path.isfile(s._CRL_MODEL_PATH):
        if brain.manager.load_trained_weights(s._CRL_MODEL_PATH):
            logger.info("CRL training weights loaded: %s", s._CRL_MODEL_PATH)
        else:
            logger.warning("CRL weight loading failed: %s", s._CRL_MODEL_PATH)

    # 存储加载异常必须显式告警：损坏文件已隔离，但人得知道
    issue = brain.manager.store.last_load_issue
    if issue:
        logger.error(
            "memory store failed to load (not silently cleared; corrupt file quarantined): %s", issue
        )
    return brain
