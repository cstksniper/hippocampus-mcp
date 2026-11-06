## What this PR does

<!-- One sentence describing the change. If behavior changes, state the before/after difference. -->

## Why

<!-- Motivation. For bug fixes, explain the root cause, not just the symptom. -->

## Verification

<!-- Which test suites were run and their results. Required when src/ changes. -->

```bash
python scripts/run_all_tests.py -k <keyword>
```

- [ ] `python scripts/run_all_tests.py` fully green
- [ ] If `src/` changed, state which suites were run in this PR
- [ ] If evaluation changed, state whether metrics changed

## Boundary self-check

- [ ] No private data, personal absolute paths, or secrets introduced
- [ ] `python tools/export_public.py --audit` reports no new hits
- [ ] New public-layer files added to the `tools/export_public.py` whitelist (if they should be public)

## Related issue

<!-- Fixes #123 -->
