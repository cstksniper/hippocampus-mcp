## 这个 PR 做了什么

<!-- 一句话说清改动。若涉及行为变化，说明前后差异。 -->

## 为什么

<!-- 动机。若是修 bug，说明根因而不只是现象。 -->

## 验证方式

<!-- 跑了哪些测试套件、结果如何。改 src/ 必填。 -->

```bash
python scripts/run_all_tests.py -k <关键字>
```

- [ ] `python scripts/run_all_tests.py` 全绿
- [ ] 改动涉及 `src/` 时，已在 PR 里写明跑过哪些套件
- [ ] 改动涉及评测时，已说明指标是否变化

## 边界自查

- [ ] 未引入私有数据、个人绝对路径或密钥
- [ ] `python tools/export_public.py --audit` 无新增命中
- [ ] 新增的公开层文件已加入 `tools/export_public.py` 白名单（若应公开）

## 相关 Issue

<!-- Fixes #123 -->
