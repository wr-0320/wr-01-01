# JMID v3 implementation audit

Status: **PASS — authorized frozen diagnostic may run**
Audit date: 2026-10-20

This audit covers implementation fidelity for
`docs/jmid_graph_uncertainty_protocol_v3.md`. It does not change that protocol,
inspect v3 research results, or authorize formal confirmation.

## Protocol-to-code checks

| Frozen requirement | Implementation evidence | Result |
|---|---|---|
| Exactly 105 perfect matchings | Deterministic recursion in `experiments/jmid_v3/graphs.py`; uniqueness and ordering test | PASS |
| Three nonzero mismatch levels | Edge-intersection test gives 12 graphs at 0.50, 32 at 0.75 and 60 at 1.00 for each truth | PASS |
| Eight labeled sources; lower endpoint is leader | Canonical edge orientation is used unchanged by the generator | PASS |
| Condition-D generator | Directed copy probability 0.85, content-triggered independent observation masks, no shared availability variable | PASS |
| Frozen split and seeds | Runner fixes seeds 4401–4410, 5,000 train, first 3,750 fit, last 1,250 validation, and 5,000 test | PASS |
| Information isolation | Selection and weighting functions accept only `ObservedDataset`; deployable fits receive episode IDs, labels and observed reports, never raw reports, copy flags or the true graph | PASS |
| Separate M3 and M4 selection | M3 validation NLL selects the M3/M23 graph; M4 validation NLL separately selects and weights M4 graphs | PASS |
| Frozen graph-average weights | Stable implementation of `exp[-1250*(NLL-NLL_min)]`, with no temperature, truncation or test reweighting | PASS |
| Paired evaluation | All six methods use the same test labels, report matrix and episode-ID digest within each seed | PASS |
| Mismatch aggregation order | Candidate metrics are averaged by distance within each seed before cross-seed summaries | PASS |
| Seed-level inference | Paired differences and two-sided 95% t intervals use ten seed differences and the df=9 critical value | PASS |
| Reproducible training-only choice | Graph scores, selections and weights are recomputed from the same saved split before test prediction and must match exactly | PASS |
| Evidence integrity | Full execution refuses a dirty checkout; outputs refuse overwrite; manifest records commit, protocol hash, seeds and every candidate graph | PASS |
| Complexity record | Per-seed wall time, fitted model count, Python peak allocation and process maximum RSS are recorded | PASS |

## Realized generator control check

A 30,000-episode implementation check with a fixed candidate graph produced:

- directed copy rate: `0.84885`;
- missing rate given raw report 1: `0.51791`;
- missing rate given raw report 0: `0.14693`;
- mean within-state paired-report correlation: `0.84757`.

These checks verify that copy dependence and content-triggered missingness are
present. They are implementation diagnostics, not v3 research evidence.

## Validation commands

```text
.venv/bin/python -m pytest tests/test_jmid_v3.py tests/test_jmid_v2.py tests/test_jmid_v1.py -q
13 passed

.venv/bin/python -m compileall -q experiments/jmid_v3
git diff --check
```

A ten-seed reduced-size smoke run produced all required file types and the
expected 60 method rows, 1,050 candidate-score rows and 30 within-seed mismatch
rows. It was explicitly marked `SMOKE_ONLY` and is not retained as evidence.

## Execution boundary

The full command may be run once from the clean implementation commit. Its
output must be archived without changing seeds, graph candidates, thresholds,
metrics or decision rules after results become visible. Passing v3 permits only
the design of a separately frozen formal protocol; it does not establish a
paper contribution.
