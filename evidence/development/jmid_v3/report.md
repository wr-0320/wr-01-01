# JMID v3 frozen exploratory diagnostic

Ten-seed diagnostic evidence; not formal confirmation.

## Predictive results

| Method | Accuracy | Brier | NLL | ECE | Wrong confidence | High-confidence error |
|---|---:|---:|---:|---:|---:|---:|
| M2_missing | 0.8682 | 0.0971 | 0.3364 | 0.0569 | 0.7921 | 0.0414 |
| Learned_M3 | 0.8504 | 0.1087 | 0.3577 | 0.0440 | 0.7608 | 0.0377 |
| Learned_M23 | 0.8648 | 0.0985 | 0.3334 | 0.0516 | 0.7751 | 0.0404 |
| Oracle_M4 | 0.8692 | 0.0868 | 0.2833 | 0.0106 | 0.6738 | 0.0279 |
| Learned_M4 | 0.8692 | 0.0868 | 0.2833 | 0.0106 | 0.6738 | 0.0279 |
| Graph_Averaged_M4 | 0.8692 | 0.0868 | 0.2833 | 0.0106 | 0.6738 | 0.0279 |

## Paired differences versus Learned M23

| Method | Metric | Mean difference | 95% CI | Lower seeds |
|---|---|---:|---:|---:|
| Oracle_M4 | accuracy | 0.004420 | [0.001053, 0.007787] | 0/10 |
| Oracle_M4 | brier | -0.011749 | [-0.013072, -0.010426] | 10/10 |
| Oracle_M4 | nll | -0.050045 | [-0.056108, -0.043982] | 10/10 |
| Learned_M4 | accuracy | 0.004420 | [0.001053, 0.007787] | 0/10 |
| Learned_M4 | brier | -0.011749 | [-0.013072, -0.010426] | 10/10 |
| Learned_M4 | nll | -0.050045 | [-0.056108, -0.043982] | 10/10 |
| Graph_Averaged_M4 | accuracy | 0.004420 | [0.001053, 0.007787] | 0/10 |
| Graph_Averaged_M4 | brier | -0.011749 | [-0.013072, -0.010426] | 10/10 |
| Graph_Averaged_M4 | nll | -0.050045 | [-0.056108, -0.043982] | 10/10 |

## Graph recovery

- Learned M3 exact recovery: 0.000.
- Learned M4 exact recovery: 1.000.
- Mean true-graph weight: 1.000000.
- Mean posterior expected graph error: 0.0000.

## Fixed mismatch diagnostic

- distance 0.5: Brier 0.0973; NLL 0.3244; accuracy 0.8661.
- distance 0.75: Brier 0.0977; NLL 0.3330; accuracy 0.8678.
- distance 1.0: Brier 0.0972; NLL 0.3373; accuracy 0.8679.

## Complexity

- Mean wall time per seed: 4.735 seconds.
- Fitted table models per seed: 318.
- Maximum recorded Python peak: 9826265 bytes.

## Frozen decision

**GO_PROTOCOL_DESIGN**

Deployable gate passed by: Learned_M4, Graph_Averaged_M4.

This exploratory outcome does not establish novelty or formal evidence.
