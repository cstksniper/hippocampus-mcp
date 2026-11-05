# 贡献指南

感谢你愿意花时间在这个项目上。这篇文档讲清楚**怎么把改动安全地送进来**，
以及几条不能破的约束——它们大多来自真实事故，不是风格偏好。

---

## 开发环境

```bash
git clone <你的 fork>
cd hippocampus-mcp

# 基础依赖
pip install -e .

# 想跑完整开发链路（含 NER / 评测）
pip install -e ".[dev,ner]"
```

Python 要求 **≥ 3.11**（本项目的实测基线；3.10 未验证，不承诺）。

**快速试跑**（不下载任何模型）：

```bash
PYTHONPATH=src python examples/quickstart.py
```

它用 `embedder_backend="hash"`，不需要 sentence-transformers、不需要 API Key。

---

## 改代码前必读：五条硬约束

这些不是建议，是踩过坑之后定下来的。违反它们的 PR 会被退回，
不是因为风格，而是因为**会造成静默错误**。

### 1. 嵌入模型只有一个真相源

模型名的解析只在 `embeddings.resolve_embedder_model()` 做，
优先级 **参数 > env > 默认**，别处不许抄 env 字面量。

原因：曾经有多个入口各自读 env，某个入口忘了设，
于是中文库悄悄跑在英文模型上，全程无报错。
现在 `brain_stats()["embedder_model_source"]` 会自曝来源，改相关代码时请保持它有效。

### 2. 换嵌入模型的唯一合法入口是 `scripts/reembed.py`

直接改配置换模型会导致库内向量与新模型不匹配。
`MemoryManager._sync_embedder_with_store()` 的处理是：
显式指定但与库内不符 → 报错；默认且与库内不符 → **跟随库内模型**。

### 3. 删除必须级联

L2 / L3 / 索引层都是 L1 的派生副本，**只删 L1 等于假删除**。
唯一入口是 `brain.forget(id, reason)`，它内部走 6 步级联。

### 4. 过滤必须在检索之前

结构化标签过滤要用 `search(allow_ids=)`，用 `-inf` 屏蔽白名单外的条目；
**BM25 那条路必须显式过滤**，否则过滤会被绕过。

### 5. 所有 MCP 工具必须带 `structured_output=False`

36 个工具无一例外，否则宿主会收到双份返回。

更多工程约定见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) 的「工程约束」一节。

---

## 测试

```bash
# 跑全部可公开套件（约 3.5 分钟，9~10 个套件 / 200+ 断言）
python scripts/run_all_tests.py

# 只跑名字含 forget 的
python scripts/run_all_tests.py -k forget

# 连 scripts/ 下的私有套件一起跑
python scripts/run_all_tests.py --include-scripts
```

`scripts/run_all_tests.py` **只用标准库**，不强制安装 pytest。
如果你装了 pytest，`pytest tests/` 同样可用（`pytest.ini` 已配）。

**新增测试的门槛**：能被 `run_all_tests.py` 自动发现、不依赖私有数据、
不依赖真实模型下载、不在仓库里写绝对路径。满足这四条的才放 `tests/`，
否则放 `scripts/`（私有层）。

---

## 提交与 PR

- 提交信息用**中文或英文均可**，但要说清**为什么**，不只是改了什么
- 一个 PR 做一件事。重构与功能改动分开提
- 改动 `src/` 必须附测试结果，说明跑了哪些套件
- 改动评测相关代码，必须说明数字是否变化

---

## ⚠️ 公开 / 私有边界

本项目从一份带私人数据的研究仓库导出，**边界是硬性的**：

| 层 | 内容 | 规则 |
|---|---|---|
| 公开层 | `src/`、`tests/`、`examples/`、`docs/` 白名单 | 可自由改动 |
| 私有层 | `data/`、`.env`、个人语料 | **不得提交**，靠 `HIPPOCAMPUS_PERSIST_DIR` 挂载 |
| 过程层 | `eval_v*/`、`*_snap_*/`、`_probe_out/` | 已在 `.gitignore`，不进库 |

**提交前请自查**：

```bash
python tools/export_public.py --audit
```

它会扫描全仓，报告密钥、私人数据与绝对路径的命中。
命中敏感内容的文件**会被拒绝导出**而不是静默复制。

---

## 报告问题

用 `.github/ISSUE_TEMPLATE/` 里的模板。
报 bug 时请附上 `brain_stats()` 的输出——它的自述字段（NER 后端、
嵌入模型来源、L0 悬空数、标签覆盖率）通常能直接定位问题。

安全问题、私人数据外泄、以及不适合公开讨论的事，
请直接发邮件到 [cstksniper@qq.com](mailto:cstksniper@qq.com)，不要先开 Issue。
