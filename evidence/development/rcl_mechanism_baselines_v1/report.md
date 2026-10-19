# Mechanism baseline development v1

This is three-seed development evidence, not formal confirmation.

| Method | Accuracy | Brier | NLL | ECE |
|---|---:|---:|---:|---:|
| single_mechanism | 77.57% | 0.2672 | 0.4235 | 0.0543 |
| global_dual_mixture | 81.52% | 0.1993 | 0.2873 | 0.0452 |
| student_t | 80.62% | 0.2149 | 0.3208 | 0.0420 |
| huber | 81.05% | 0.2129 | 0.3195 | 0.0420 |
| reliability_weighted | 77.48% | 0.2666 | 0.4220 | 0.0534 |

## Global dual mixture paired differences

Positive accuracy favors the dual mixture; negative Brier and NLL favor it.

| Comparator | Accuracy pp (95% CI) | Brier | NLL |
|---|---:|---:|---:|
| single_mechanism | +3.95 [-1.47, +9.37] | -0.0679 | -0.1362 |
| student_t | +0.90 [-2.28, +4.09] | -0.0156 | -0.0335 |
| huber | +0.48 [-2.39, +3.34] | -0.0136 | -0.0322 |
| reliability_weighted | +4.05 [-1.22, +9.32] | -0.0673 | -0.1347 |

Intervals use paired training-seed means. Crossing zero is not interpreted as equivalence.
