# JMID v2 independent-generator falsification protocol

Status: **frozen before the v2 development run**  
Date: 2026-10-20

## Purpose

JMID v2 tests whether the v1 probability-quality gain survives a separately
implemented data mechanism and a fixed, parameter-comparable composition of the
two single-factor models. It is a falsification experiment, not a formal
confirmation or an algorithm-development stage.

## Independent generator

The v2 generator does not call the v1 generator and does not use its
state-dependent observation probabilities or shared availability outcome.
Instead it uses:

1. a noisy state-dependent binary report at a group leader;
2. a relay that either creates an independent report or copies the leader's
   already-noisy report, producing directed evidence dependence;
3. independent observation masks conditional on the uncensored report content;
4. positive-report censorship as the matched missingness mechanism.

The two-source provenance groups remain observable metadata in the matched
experiment. Latent uncensored reports and copy events are diagnostic-only.

## Frozen methods and composition

M2, M3 and M4 are the v1 missing-aware, dependence-aware and joint models. They
receive the same labeled training episodes, observed values, masks and declared
provenance groups.

The new **M23 fixed composition** fits M1, M2 and M3 on the same training set and
combines their posterior log odds by

\[
\operatorname{logit} p_{23}(Z=1\mid y)
=\operatorname{logit}p_2+\operatorname{logit}p_3-\operatorname{logit}p_1.
\]

Subtracting M1 removes the independent report evidence common to M2 and M3.
Every coefficient is fixed at one before testing; there is no validation or
test-set weight tuning. With binary states and three two-source groups, M23 fits
approximately 55 free categorical parameters across its component tables,
versus 49 for M4. M23 therefore has no smaller parameter budget.

## Matched conditions and budget

| Condition | Content censorship | Relay copying |
|---|---:|---:|
| A | 0.0 | 0.0 |
| B | 1.0 | 0.0 |
| C | 0.0 | 0.85 |
| D | 1.0 | 0.85 |

Seeds are `3401, 3402, 3403`. Every matched condition uses 4,000 training and
5,000 paired test episodes per seed. Laplace smoothing remains `alpha=1`.

## Frozen transfer tests

1. **Strength:** train at `(0.5, 0.5)` and test joint strengths
   `0.0, 0.25, 0.5, 0.75, 1.0`.
2. **Unseen missingness:** train under positive-report censorship and test,
   without refitting, under negative-report censorship and disagreement-triggered
   relay censorship.
3. **Unseen dependence structure:** train with groups `(0,1),(2,3),(4,5)` and
   test directed copying on `(1,2),(3,4),(5,0)`, while the fitted models retain
   the training provenance grouping.

The last test deliberately checks sensitivity to a wrong dependency graph. It
is not expected that every model remain optimal under arbitrary graph shift.

## Metrics and diagnostics

Accuracy, Brier, NLL, ten-bin ECE, mean confidence on errors and unconditional
high-confidence error rate are reported separately against every comparator.
Diagnostics include state-conditional missing rates, within-state uncensored
report correlation, the missing-rate gap between positive and negative reports,
and within-state missing-indicator correlation.

## Predeclared decision

**Continue** only if condition D gives M4 lower mean Brier and NLL than M2, M3
and M23, with the Brier and NLL directions repeated in all three seeds, and no
more than one percentage point accuracy loss against the best comparator. M4
must also beat M23 in mean Brier and NLL at strengths 0.75 and 1.0. Confidence
metrics are compared separately against every baseline and are never summarized
as universal dominance.

**Stop the current joint-table route** if M23 matches or beats M4 in condition D,
or if M4 loses its Brier/NLL advantage after replacing the v1 generator.

**Inconclusive; do not upgrade** if condition-D seed directions conflict, or if
only one probability metric improves. Unseen-mechanism and graph-shift results
define scope; negative transfer results cannot be repaired by post-hoc tuning in
v2.

