# Dual-mechanism mixture formal confirmation v1

This is the frozen ten-seed formal confirmation result.

| Method | Accuracy | Brier | NLL | ECE |
|---|---:|---:|---:|---:|
| single_mechanism | 77.67% | 0.2707 | 0.4662 | 0.0625 |
| global_dual_mixture | 82.20% | 0.2148 | 0.3367 | 0.0535 |
| student_t | 80.71% | 0.2229 | 0.3416 | 0.0443 |
| huber | 80.74% | 0.2196 | 0.3321 | 0.0449 |
| reliability_weighted | 77.73% | 0.2705 | 0.4659 | 0.0629 |

## Global dual mixture paired differences

Positive accuracy favors the dual mixture; negative Brier and NLL favor it.

| Comparator | Accuracy pp (95% CI) | Brier (95% CI) | NLL (95% CI) |
|---|---:|---:|---:|
| single_mechanism | +4.53 [+3.91, +5.15] | -0.0559 [-0.0847, -0.0271] | -0.1296 [-0.2154, -0.0437] |
| student_t | +1.49 [+0.57, +2.40] | -0.0081 [-0.0286, +0.0123] | -0.0049 [-0.0721, +0.0623] |
| huber | +1.46 [+0.82, +2.09] | -0.0048 [-0.0231, +0.0135] | +0.0045 [-0.0603, +0.0694] |
| reliability_weighted | +4.47 [+3.85, +5.09] | -0.0557 [-0.0846, -0.0269] | -0.1292 [-0.2156, -0.0428] |

Intervals use paired training-seed means. Crossing zero is not interpreted as equivalence.

## Pre-registered decision

Decision: **NOT_CONFIRMED**.

| Required condition | Passed |
|---|---:|
| nll_upper_below_zero | no |
| brier_upper_below_zero | no |
| accuracy_lower_above_minus_two_points | yes |

All three conditions are required; secondary results cannot rescue a failed primary rule.
