# -*- coding: utf-8 -*-
"""server_runtime：server.py 的适配层协作者子包。

本包的模块是 server.py 非工具区块的整体拆分（预热 / 多库寻址 / 剖面门控 /
派发 / 入口序列）。纪律：

* **状态与 monkeypatch 面留在 server.py**，本包模块一律经 ``_srv()``
  延迟 import 现取——测试对 ``server._X`` 的替换/赋值全部照常生效；
* server.py 侧按 import 名原样再导出，既有调用点与测试引用零改动；
* 本包不参与层级方向静态检查（与入口层同待遇：其 import 边不被方向
  判定，也不产生新违规边——import 面与
  现 server.py 完全一致：application / common / library / config）。

模块清单（解析助手不在本包——落 ``common/params.py``，
属底层纯函数）：

* ``warmup``——预热区（后台预载会话/退出哨兵）；
* ``routing``——多库寻址（library_use 粘性 / 家库解析 / 并发防护）；
* ``composition``——组合根工厂（application 服务的装配点）；
* ``profiles``——剖面门控（common/full 工具投影与 _TOOL_GROUPS）；
* ``dispatch``——工具派发（工具体外的调度壳）；
* ``stdout_guard``——stdout 守卫（_ThreadAwareStdout，防 MCP 帧被污染）；
* ``entry``——入口序列（daemon/直连启动编排）。

import 面纪律（沿用上方三条之外的补充）：``dispatch``/``stdout_guard`` 只用
标准库；``profiles`` 只依赖 config；``routing``/``warmup`` 依赖 library+common；
``composition`` 依赖 application；``entry`` 只依赖本包——**不新增包外依赖边**，
搬进来的代码是什么 import 面，本包就保持什么面。
"""
