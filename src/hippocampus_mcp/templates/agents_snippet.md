# hippocampus 记忆纪律（可复制模板片段，2026-11-07）

> 用法：把下面整块粘进各项目的 `AGENTS.md`（或由
> `python -m hippocampus_mcp.cli install` 的后续版本自动写入）。与 README 的
> operating rules 同源；此处为"消灭手抄漂移"的单一模板，改纪律先改这里。

<!-- hippocampus:begin（幂等标记块，安装器按此识别并整块替换） -->

## 海马体记忆纪律

1. **开场**：主会话第一轮先调 `mcp__hippocampus__session_start`（带任务描述）。
2. **读侧**：涉及既往研究/历史决策/"之前说过" 的问题，先 `recall` 或
   `get_memory_context` 再回答；注意 `score` 是 RRF 排名分，不是余弦。
3. **写侧**：用户说"记住/存进去" → 立即 `ingest`；关键拍板/口径变更落盘前
   先 `recall` 查冲突，命中旧口径先 supersede。
4. **收尾**：任务完结调 `session_end`（summary + open_topics）；
   Stop hook 兜底提醒出现时，先收口再结束。
5. **自检**：见不到 `mcp__hippocampus__*` 工具时，按主仓 AGENTS.md
   「接入自检」分层排查（文件级 → 解释器级），`install` 可修复部署。

<!-- hippocampus:end -->
