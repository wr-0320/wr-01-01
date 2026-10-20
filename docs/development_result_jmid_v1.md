# JMID v1 exploratory development result

Date: 2026-10-20  
Decision: **conditional continuation for falsification; no formal experiment**

## Scope

This is the first three-seed result under
`jmid_exploratory_protocol_v1.md`. It compares four small empirical generative
models on paired datasets. It does not establish novelty, significance,
equivalence or publication readiness.

## Manipulation check

In the decisive condition D, mean missingness was 0.2213, with rates 0.1228 for
(Z=0) and 0.3169 for (Z=1). Mean within-state report correlation was 0.4214,
and mean within-state missing-indicator correlation was 0.7669. The experiment
therefore instantiated both informative missingness and dependent evidence; the
correlation diagnostic did not count dependence induced solely by the state.

## Main result

| Condition | Method | Accuracy | Brier | NLL | ECE | Wrong confidence | High-confidence error |
|---|---|---:|---:|---:|---:|---:|---:|
| A | M1 | 0.8761 | 0.0859 | 0.2818 | 0.0089 | 0.7042 | 0.0280 |
| A | M2 | 0.8767 | 0.0859 | 0.2821 | 0.0097 | 0.7072 | 0.0270 |
| A | M3 | 0.8764 | 0.0859 | 0.2819 | 0.0102 | 0.7096 | 0.0262 |
| A | M4 | 0.8761 | 0.0864 | 0.2835 | 0.0138 | 0.7118 | 0.0244 |
| B | M1 | 0.8737 | 0.0875 | 0.2848 | 0.0250 | 0.7027 | 0.0269 |
| B | M2 | 0.9169 | 0.0652 | 0.2195 | 0.0053 | 0.7831 | 0.0152 |
| B | M3 | 0.8739 | 0.0875 | 0.2850 | 0.0267 | 0.7056 | 0.0247 |
| B | M4 | 0.9169 | 0.0653 | 0.2197 | 0.0071 | 0.7813 | 0.0152 |
| C | M1 | 0.8497 | 0.1074 | 0.3532 | 0.0382 | 0.7469 | 0.0403 |
| C | M2 | 0.8503 | 0.1074 | 0.3533 | 0.0385 | 0.7492 | 0.0399 |
| C | M3 | 0.8544 | 0.1037 | 0.3350 | 0.0123 | 0.7175 | 0.0184 |
| C | M4 | 0.8535 | 0.1042 | 0.3364 | 0.0135 | 0.7195 | 0.0189 |
| D | M1 | 0.8297 | 0.1141 | 0.3717 | 0.0572 | 0.7139 | 0.0549 |
| D | M2 | 0.8715 | 0.0980 | 0.3357 | 0.0455 | 0.8046 | 0.0359 |
| D | M3 | 0.8309 | 0.1111 | 0.3531 | 0.0254 | 0.6776 | 0.0149 |
| D | M4 | 0.8714 | 0.0933 | 0.3051 | 0.0111 | 0.7253 | 0.0221 |

In condition D, M4 minus M2 was -0.01 percentage point in accuracy, -0.0047
in Brier and -0.0306 in NLL. All three seeds had lower Brier and NLL than M2.
M4 also reduced wrong confidence and high-confidence errors relative to M2.

Against M3, M4 improved accuracy by about 4.05 percentage points, Brier by
0.0177 and NLL by 0.0480. M3 nevertheless had lower confidence on its remaining
errors and a lower high-confidence error rate. M4 therefore does not dominate
both single-factor models on every metric.

## Shift result

The models were trained at joint strength 0.2. At test strength 0.75, M4 had
mean Brier/NLL 0.1008/0.3324 versus M2's 0.1011/0.3341. At strength 1.0, the
values were 0.1066/0.3539 versus 0.1075/0.3583. These mean gains are small. One
of three seeds reversed the Brier direction at each endpoint, and one seed
reversed the NLL direction at strength 0.75.

## Decision

The result supports the existence of a calibration problem under the constructed
joint mechanism. It justifies one second exploratory falsification step, not a
formal confirmation experiment or a more complex algorithm.

The strongest positive evidence is the seed-consistent condition-D Brier/NLL
gain over M2 with essentially unchanged accuracy. The main limitations are:

- M4 is a saturated two-source categorical table and may exploit the exact
  provenance grouping built into this simulator;
- shift gains over M2 are very small and not directionally uniform by seed;
- the frozen four-method comparison did not include a parameter-balanced simple
  composition of M2 and M3;
- only one data-generating family was used.

Before increasing seeds or developing a new algorithm, the next study must use
an independently coded generator with a different missingness rule and add a
simple M2+M3 composition. If that control matches M4 or the gain disappears,
stop the joint-model route. Existing results must remain exploratory.

