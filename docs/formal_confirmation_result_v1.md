# Dual-mechanism mixture formal confirmation result v1

## Status and provenance

The frozen formal experiment completed once on 2026-10-20 from execution commit
`f7544ee7aea9a37cefe8a481c2126c38c897d182`. It used the ten reserved training
seeds 13001--13010 and produced 35,000 model-episode evaluations. No seed was
replaced, no interim result was inspected, and no fitting or decision rule was
changed.

The pre-registered result is **NOT_CONFIRMED**.

## Integrity audit

- 35,000 episode rows are present.
- Every one of the 10 seed × 5 method × 7 family cells contains 100 rows.
- The 7,000 test scenarios each contain all five paired methods.
- All recorded Brier, NLL, and confidence values are finite.
- Ten parameter files are present.
- An independent recomputation from `paired_differences.csv` reproduces every
  condition and the final decision in `decision.json`.
- The formal report is explicitly marked as a ten-seed confirmation result.

## Overall results

| Method | Accuracy | Brier | NLL | ECE |
|---|---:|---:|---:|---:|
| Single mechanism | 77.67% | 0.2707 | 0.4662 | 0.0625 |
| Global dual mixture | 82.20% | 0.2148 | 0.3367 | 0.0535 |
| Student-t | 80.71% | 0.2229 | 0.3416 | 0.0443 |
| Huber | 80.74% | 0.2196 | 0.3321 | 0.0449 |
| Reliability weighted | 77.73% | 0.2705 | 0.4659 | 0.0629 |

Differences below are global dual mixture minus comparator. Positive accuracy
favors the mixture; negative Brier and NLL favor it.

| Comparator | Accuracy pp (95% CI) | Brier (95% CI) | NLL (95% CI) |
|---|---:|---:|---:|
| Single mechanism | +4.53 [+3.91, +5.15] | -0.0559 [-0.0847, -0.0271] | -0.1296 [-0.2154, -0.0437] |
| Student-t | +1.49 [+0.57, +2.40] | -0.0081 [-0.0286, +0.0123] | -0.0049 [-0.0721, +0.0623] |
| Huber | +1.46 [+0.82, +2.09] | -0.0048 [-0.0231, +0.0135] | +0.0045 [-0.0603, +0.0694] |
| Reliability weighted | +4.47 [+3.85, +5.09] | -0.0557 [-0.0846, -0.0269] | -0.1292 [-0.2156, -0.0428] |

## Pre-registered decision

The primary comparison was global dual mixture versus Huber. All three conditions
were required:

| Required condition | Observed interval bound | Passed |
|---|---:|---:|
| NLL upper bound below 0 | +0.0694 | No |
| Brier upper bound below 0 | +0.0135 | No |
| Accuracy lower bound above -2 pp | +0.82 pp | Yes |

The accuracy guard passed with a positive interval, but both probability-quality
conditions failed. Brier has a favorable point estimate with an interval crossing
zero. NLL has a slightly unfavorable point estimate with a wide interval crossing
zero. Secondary comparisons cannot rescue the failed primary intersection rule.

## Interpretation for the paper

The formal experiment does not support the planned claim that the global dual
mixture improves both NLL and Brier over Huber while preserving accuracy. The
result also does not establish equivalence: the probability-quality intervals are
wide enough to include effects in either direction.

The dual mixture does show higher formal accuracy than Huber in these scenarios,
and it clearly improves accuracy, Brier, and NLL over the matched single mechanism
and linear reliability weighting. Those comparisons should be reported as
secondary structural evidence. The paper must state that superiority over the
strong Huber probability model was not confirmed.

No additional seeds, post-result hyperparameter changes, or replacement primary
comparators should be introduced for protocol v1.
