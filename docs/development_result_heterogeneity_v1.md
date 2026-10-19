# Controlled heterogeneity development result v1

## Status and question

This is three-seed development evidence, not a formal confirmation result. The
frozen question was whether source-region granularity in the reliability prior
adds value when reliability heterogeneity is increased while the overall
contamination rate and all other scenario conditions are held approximately
constant.

The compared models share the same source-region-conditioned reliable and
compromised emission estimates. They differ only in the granularity of the
mixture prior: global, source, or source-region. Consequently, this experiment
tests reliability-prior granularity. It does not test whether all forms of
source-region conditioning are unnecessary.

The protocol and decision rules were frozen in
`docs/heterogeneity_scan_protocol.md` before the full run. The run used seeds
9201--9203, seven intelligence families, 100 paired test episodes per family,
four heterogeneity profiles, and three models, for 25,200 model-episode
evaluations.

## Manipulation check

Every configured activation matrix has mean 0.60. The profiles only change the
probability that an otherwise eligible red manipulation activates. Hidden-state
sampling, observation noise, missingness, delay, deception families, fitting,
calibration, evaluation position, and scenario seeds are shared.

Values below are means over the three development seeds. `Cell SD` and `cell
range` are calculated from the nine observed source-region contamination rates
within each seed and split.

| Profile | Split | Configured activation SD | Overall contamination | Cell SD | Cell range |
|---|---|---:|---:|---:|---:|
| low | fit | 0.0000 | 0.0269 | 0.0032 | 0.0098 |
| medium | fit | 0.2449 | 0.0265 | 0.0122 | 0.0363 |
| high | fit | 0.3266 | 0.0264 | 0.0145 | 0.0429 |
| extreme | fit | 0.4198 | 0.0268 | 0.0189 | 0.0494 |
| low | test | 0.0000 | 0.0605 | 0.0155 | 0.0489 |
| medium | test | 0.2449 | 0.0597 | 0.0290 | 0.0920 |
| high | test | 0.3266 | 0.0610 | 0.0366 | 0.1100 |
| extreme | test | 0.4198 | 0.0614 | 0.0456 | 0.1215 |

The actual cell heterogeneity rises monotonically while overall contamination
stays within 0.17 percentage points across test profiles. The low profile still
has some empirical cell variation because attack eligibility and report coverage
vary by source and region; the controlled activation probability itself is
homogeneous.

## Results

Metrics are means over three independently fitted seeds. Accuracy is evaluated
at the fixed inference position. Brier and NLL are retained as primary
probability-quality measures; ECE is reported as a calibration diagnostic.

| Profile | Method | Accuracy | Brier | NLL | ECE |
|---|---|---:|---:|---:|---:|
| low | global mixture | 82.90% | 0.1878 | 0.2673 | 0.0317 |
| low | source mixture | 83.00% | 0.1877 | 0.2670 | 0.0306 |
| low | source-region RCL | 82.86% | 0.1875 | 0.2668 | 0.0312 |
| medium | global mixture | 82.43% | 0.1836 | 0.2622 | 0.0265 |
| medium | source mixture | 82.57% | 0.1829 | 0.2611 | 0.0250 |
| medium | source-region RCL | 82.57% | 0.1828 | 0.2610 | 0.0246 |
| high | global mixture | 83.33% | 0.1868 | 0.2676 | 0.0192 |
| high | source mixture | 83.33% | 0.1868 | 0.2676 | 0.0197 |
| high | source-region RCL | 83.43% | 0.1867 | 0.2676 | 0.0211 |
| extreme | global mixture | 83.14% | 0.1865 | 0.2688 | 0.0206 |
| extreme | source mixture | 83.14% | 0.1866 | 0.2689 | 0.0212 |
| extreme | source-region RCL | 82.95% | 0.1873 | 0.2707 | 0.0216 |

Source-region RCL minus each grouped-prior comparator:

| Profile | Comparator | Accuracy difference (pp, 95% CI) | Brier | NLL |
|---|---|---:|---:|---:|
| low | global | -0.05 [-0.25, +0.16] | -0.0003 | -0.0005 |
| low | source | -0.14 [-0.50, +0.21] | -0.0002 | -0.0003 |
| medium | global | +0.14 [-0.57, +0.85] | -0.0008 | -0.0012 |
| medium | source | +0.00 [+0.00, +0.00] | -0.0001 | -0.0001 |
| high | global | +0.10 [-0.93, +1.12] | -0.0002 | +0.0000 |
| high | source | +0.10 [-0.93, +1.12] | -0.0002 | +0.0000 |
| extreme | global | -0.19 [-0.60, +0.22] | +0.0007 | +0.0018 |
| extreme | source | -0.19 [-0.73, +0.35] | +0.0007 | +0.0018 |

The intervals are seed-level paired t intervals with only three development
seeds. An interval crossing zero is not evidence of equivalence or noninferiority.

## Predeclared decision

Decision: **use the global mixture as the main candidate and retain the
source-region prior as an optional parameterization**.

The source-region model did not beat both grouped-prior models in both the high
and extreme profiles, did not reach the predeclared two-percentage-point gain,
and did not improve both Brier and NLL in the high and extreme profiles. The
source model also failed to reach a two-point gain over the global model in the
medium profile. No model degraded by five points as heterogeneity increased.

This development result supports neither a source-region-prior advantage nor a
formal equivalence claim. It narrows the paper's defensible contribution toward
the dual-mechanism mixture likelihood. A later formal comparison should only be
run after that revised claim and its protocol are frozen. The existing
correlation-aware and intent-coupled NO-GO results remain unchanged.
