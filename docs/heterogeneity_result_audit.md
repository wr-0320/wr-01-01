# Heterogeneity result audit

## Scope

This audit checks the three-seed controlled heterogeneity development result at
full stored precision. It does not add seeds or alter the frozen decision.

## Medium-profile zero accuracy difference

The source-level mixture and source-region RCL made exactly the same number of
correct fixed-position classifications in every development seed:

| Seed | Source accuracy | Source-region accuracy | Exact difference |
|---|---:|---:|---:|
| 9201 | 0.8242857142857143 | 0.8242857142857143 | 0 |
| 9202 | 0.8357142857142857 | 0.8357142857142857 | 0 |
| 9203 | 0.8171428571428572 | 0.8171428571428572 | 0 |

The paired seed differences are therefore the vector `[0, 0, 0]`. Its mean,
standard error, and t-interval width are all exactly zero. The reported
`0.00 [0.00, 0.00]` is not caused by display rounding.

## Probability predictions and parameters

The models did not converge to identical parameters or identical posterior
probabilities. Source-region minus source-level probability metrics were:

| Seed | Brier difference | NLL difference | ECE difference |
|---|---:|---:|---:|
| 9201 | +0.0001466264 | +0.0002300469 | -0.0009501675 |
| 9202 | +0.0000534945 | +0.0001515987 | -0.0001820821 |
| 9203 | -0.0005170398 | -0.0006841556 | -0.0000132259 |

Within each seed, both methods selected the same temperature from the frozen
grid: 1.1481536215, 1.0471285481, and 1.2589254118 respectively. Their reliability
matrices differ. The source model has a constant value across the three regions
within each source, whereas the source-region model estimates nine distinct
values. For example, seed 9201 estimates source 0 as
`[0.9860796, 0.9860796, 0.9860796]` in the source model and
`[0.9828080, 0.9871223, 0.9874335]` in the source-region model.

The exact accuracy match therefore means that the small posterior changes did
not cross an argmax decision boundary in any of the 2,100 paired evaluations per
method. It is evidence of identical classifications in these development draws,
not model identity, equivalence, or noninferiority.

## Implication

The stored results and interval calculation are internally consistent. No result
needs correction. The audit reinforces the frozen decision to stop searching for
a reliability-prior granularity advantage and to test the dual-mechanism mixture
against robust likelihood and fusion baselines.
