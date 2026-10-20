# JMID v3 frozen diagnostic result

Date: 2026-10-20
Code commit: `b8c729df361d68b9a7604e7a59f6aa1c8edb653f`
Frozen decision: **GO_PROTOCOL_DESIGN**
Research interpretation: **structure recoverable in this regime; graph averaging adds no demonstrated benefit**

This is the single authorized ten-seed exploratory run under
`docs/jmid_graph_uncertainty_protocol_v3.md`. It is not formal confirmation and
does not establish algorithm novelty.

## Integrity checks

- The run started from the clean implementation commit above.
- The manifest protocol hash matches the frozen protocol.
- Seeds were exactly 4401--4410, with 5,000 training and 5,000 paired test
  episodes per seed.
- All 105 graph candidates were scored for every seed.
- Training-only graph selection and graph weights were recomputed before test
  prediction and reproduced exactly for every seed.
- The output directory was created once and the runner refused overwrite.

## Main predictive result

| Method | Accuracy | Brier | NLL | ECE | Wrong confidence | High-confidence error |
|---|---:|---:|---:|---:|---:|---:|
| M2 | 0.8682 | 0.0971 | 0.3364 | 0.0569 | 0.7921 | 0.0414 |
| Learned M3 | 0.8504 | 0.1087 | 0.3577 | 0.0440 | 0.7608 | 0.0377 |
| Learned M23 | 0.8648 | 0.0985 | 0.3334 | 0.0516 | 0.7751 | 0.0404 |
| Oracle M4 | 0.8692 | 0.0868 | 0.2833 | 0.0106 | 0.6738 | 0.0279 |
| Learned M4 | 0.8692 | 0.0868 | 0.2833 | 0.0106 | 0.6738 | 0.0279 |
| Graph-Averaged M4 | 0.8692 | 0.0868 | 0.2833 | 0.0106 | 0.6738 | 0.0279 |

Learned M4 minus Learned M23 gave:

| Metric | Mean difference | Seed-level 95% t interval | Favorable seeds |
|---|---:|---:|---:|
| Accuracy | +0.00442 | [0.00105, 0.00779] | 10/10 higher |
| Brier | -0.01175 | [-0.01307, -0.01043] | 10/10 lower |
| NLL | -0.05005 | [-0.05611, -0.04398] | 10/10 lower |

The comparison with the stronger graph-free M2 baseline also remained favorable
for probability quality: Learned M4 reduced Brier by 0.01026 and NLL by 0.05311
on average, with both reductions present in 10/10 seeds. Its accuracy difference
versus M2 was +0.00104 with interval [-0.00141, 0.00349], so no stable accuracy
gain over M2 is claimed.

## What happened to graph uncertainty

Learned M4 recovered the true graph in 10/10 seeds. The smallest gap between
the true graph's validation NLL and the runner-up was 0.01809. Under the frozen
exponential weighting rule, the true graph received weight 1.000000 to reported
precision for every seed, and posterior expected graph error was zero.

Consequently:

- Learned M4 equaled Oracle M4;
- Graph-Averaged M4 equaled Learned M4;
- graph averaging did not hedge meaningful residual structure uncertainty;
- its 105 full-data component fits supplied no observed predictive benefit over
  selecting one graph.

This outcome supports recoverability only for this strong-copy, 3,750-episode
structure-fit regime. It does not show robustness to ambiguous, weak or
nonidentifiable graph structure.

Learned M3 recovered the true graph in 0/10 seeds. This does not contradict the
M4 result: M3 deliberately ignores the informative missingness component and
uses a different frozen validation score.

## Fixed-mismatch diagnostic

| Injected edge distance | Accuracy | Brier | NLL |
|---:|---:|---:|---:|
| 0.50 | 0.8661 | 0.0973 | 0.3244 |
| 0.75 | 0.8678 | 0.0977 | 0.3330 |
| 1.00 | 0.8679 | 0.0972 | 0.3373 |

The mismatch results confirm that supplied graph error degrades probability
quality, but the curve is not strictly monotone in every metric. It must not be
presented as a structure-learning error curve because candidates were grouped
by injected distance and averaged within each seed.

## Complexity and frozen decision

Mean runtime was 4.735 seconds per seed and the maximum recorded Python
allocation peak was 9,826,265 bytes. The archived `fitted_table_models` field is
318, which counts the primary analysis fits: 105 M3 validation fits, 105 M4
validation fits, 105 full-data M4 fits and three full-data baseline fits. The
frozen reproducibility check repeated the 105 M3 and 105 M4 validation fits, so
the actual total executed during the timed run was 528 fits per seed. This
post-run accounting correction does not affect predictions, intervals or the
frozen decision; the raw evidence file remains unchanged.

Both Learned M4 and Graph-Averaged M4 met every frozen continuation condition
against Learned M23. The machine decision is therefore `GO_PROTOCOL_DESIGN`.
Per protocol, this authorizes at most the design of a separately frozen formal
study. The post-v3 research decision is recorded separately because a passed
empirical gate does not by itself establish an independent algorithm
contribution.
