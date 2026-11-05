# -*- coding: utf-8 -*-
"""应用层：用例编排。

五个用例服务：Write + Bootstrap（14 工具）、Recall（11 工具）、
Organize（10 工具）、Maintain 5 + LibraryAdmin 5
（原 Maintain 11 因**是否绑定单一 brain**拆成两个类，见
``library_admin_service`` 模块 docstring）。三条纪律：

1. **不 import brain**：``brain/core.py`` 是大门面类，runtime
   由组合根（server.py 的 ``_write_service`` / ``_bootstrap_service``
   工厂）构造注入，服务内只以 ``Any`` 持有——import 图上这条边恒为零。
   brain 的方法体将来上收进各 Service 时，本包 API 不变，包外无感。
2. **不碰 I/O**：``open()`` / ``sqlite3`` / ``store._records`` 不出现在
   本包；存储访问经注入的 runtime 公开 API（端口纪律）。
3. **信封自产**：写类返回体的 ``library`` 首键（库自曝）由服务组装，
   工具只 ``json.dumps``——响应组装归用例层。

依赖方向：application → common /
domain / store / map / engine / retrieval / lnn / library，**不准 → brain**。
"""
from __future__ import annotations

from .base import BaseService
from .bootstrap_service import BootstrapService
from .library_admin_service import LibraryAdminService, LibraryPorts
from .maintain_service import MaintainService
from .organize_service import OrganizeService
from .recall_service import RecallService
from .write_service import WriteService

__all__ = ["BaseService", "BootstrapService", "LibraryAdminService",
           "LibraryPorts", "MaintainService", "OrganizeService",
           "RecallService", "WriteService"]
