# Benchmarks and evidence

> What HippoMCP currently measures, what that measurement does not prove, and how to verify the
> correctness claims that matter more than a single score.

**English** ｜ Chinese overview lives in [`README.zh-CN.md`](../README.zh-CN.md).

---

## 1. Current measured result

The latest measured baseline is:

```text
F1 = 0.244 ± 0.034
```

Conditions:

- Dataset: LoCoMo `locomo10`
- Questions: 50 QA
- Runs: 3
- Judge: `glm-5.3-flash`
- Judge interface: pluggable through `JUDGE_BASE_URL`, `JUDGE_API_KEY`, and `JUDGE_MODEL`

The judge environment itself varied by about 0.068, so small configuration differences are not
meaningful in this setup.

## 2. What this number does not prove

This is a small, self-defined measurement, not a leaderboard claim.

It should not be compared directly with published protocols that use approximately 1,540
questions, exclude the adversarial split, or use a different judge. Cross-project comparisons
require the same dataset, question count, protocol, and judge.

The historical driver that produced these measurements is not currently published. Treat the
number as a **measurement record with stated conditions**, not as a one-command reproduction
claim.

Earlier experimental configurations are intentionally omitted from this public summary. They
used judge environments that no longer exist and should not be cited as current performance.

## 3. What is more important than the score

The primary correctness claims are covered by executable tests:

- cascade-safe deletion;
- `superseded` fact handling;
- concurrent-save merging;
- workspace binding;
- cross-library isolation;
- user-library review boundaries;
- session lifecycle and compaction recovery.

Run the public suites:

```bash
python scripts/run_all_tests.py --list
python scripts/run_all_tests.py -k forget
python scripts/run_all_tests.py
```

`test_perf_write` contains absolute-time red lines that can fail under machine load. For
regression judgment, look at marginal-cost items and CI behavior, not one noisy wall-clock run.

## 4. Dataset availability

The public layer includes data loaders and metric implementations, but not the datasets
themselves.

| Dataset | Source | License | Bundled here? |
|---|---|---|---|
| LoCoMo | [snap-research/locomo](https://github.com/snap-research/locomo) | CC BY-NC 4.0 | No |
| LongMemEval | Hugging Face [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | MIT | No |

Download the dataset yourself and pass its path to the loader. Do not redistribute LoCoMo with
this repository.

## 5. Judge interface

Evaluation can use any OpenAI-compatible `/chat/completions` endpoint through:

```text
JUDGE_BASE_URL
JUDGE_API_KEY
JUDGE_MODEL
```

Changing the judge should not require changing core memory code. The core service does not
depend on these variables.

See [`scripts/judge_client.py`](../scripts/judge_client.py) for the intended boundary.

## 6. Known open areas

Known improvement areas include:

- multi-hop reasoning;
- open-domain retrieval;
- broader and more reproducible benchmark coverage;
- better public tooling for end-to-end reproduction.

These are active engineering questions, not solved capabilities. Do not present them as
strengths.

## 7. How to evaluate HippoMCP on your own workload

A useful evaluation should measure the behavior you actually care about:

1. Prepare a private or public corpus with known decisions and updates.
2. Define questions that require recall across sessions.
3. Include cases where a fact changes or must be forgotten.
4. Include cross-project isolation cases if you use multiple libraries.
5. Measure both answer quality and context cost.
6. Pin the dataset, protocol, and judge before comparing systems.
7. Keep the raw outputs so the result can be audited later.

This is usually more informative than comparing a small public score across different memory
systems.
