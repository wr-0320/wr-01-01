# Joint missingness--dependence exploratory protocol v1

Status: **frozen before the first development run**  
Date: 2026-10-20

## Question

Can a small joint model avoid posterior bias and false confidence when report
missingness is informative and reports share a latent information chain, in a
setting where models that address only one of those effects cannot?

This is an exploratory diagnostic. It is not a formal confirmation experiment,
a novelty claim, or an authorization to revive D-MIR.

## Isolation

All new code and results belong to `experiments/jmid_v1/` and
`outputs/development/jmid_v1/` in the independent `wr-01-01` repository. The
source repository `/workspace/wr-01`, archived RCL evidence and Route B theorem
documents are read-only.

## Data-generating model

Each episode has a binary state (Z\in\{0,1\}), six report sources and three
known two-source provenance groups. A group-level binary signal is informative
about (Z). A dependence-strength parameter controls whether each source uses
the group signal or an independent signal draw. Reports are noisy copies of
their selected signal.

A missingness-strength parameter interpolates between constant observation
probability and a state-dependent observation probability. At high strength,
reports are less available during anomalous activity. When both controls are
active, a provenance group may additionally share one availability outcome;
this preserves each source's state-conditional marginal observation rate while
creating the missingness--dependence interaction under test.
The same state labels, observed report values, missingness masks, provenance
groups and episode budget are supplied to all methods. Latent group signals are
used only for generator diagnostics.

## Frozen methods

- **M1 naive:** independent per-source Bernoulli likelihood; missing reports
  contribute no likelihood.
- **M2 missing-aware:** independent per-source categorical likelihood over
  missing, observed zero and observed one.
- **M3 dependence-aware:** a joint binary likelihood per provenance group,
  fitted from complete pairs; partially observed pairs are marginalized and
  missingness contributes no likelihood.
- **M4 joint:** a joint categorical likelihood over the two reports in each
  provenance group, including missing values.

All four are smoothed empirical generative models with Laplace parameter
`alpha=1`. M4 is intentionally a small saturated group model, not a proposed
publication algorithm.

## Conditions and budgets

| Condition | Missingness strength | Dependence strength |
|---|---:|---:|
| A | 0.0 | 0.0 |
| B | 1.0 | 0.0 |
| C | 0.0 | 0.9 |
| D | 1.0 | 0.9 |

Development seeds are `2401, 2402, 2403`. Each matched condition uses 4,000
training episodes and 5,000 test episodes per seed. A smoke run may reduce those
counts but cannot be interpreted as evidence.

The distribution-shift diagnostic trains once at missingness/dependence
strength `(0.2, 0.2)` and tests paired datasets at joint strengths
`0.0, 0.25, 0.5, 0.75, 1.0`, with 5,000 episodes per point and seed.

## Metrics

Accuracy, binary Brier score, NLL, ten-bin ECE, mean confidence on errors and
the unconditional rate of wrong predictions made with confidence at least 0.9
are recorded. Accuracy and probability quality must not be conflated.

Generator diagnostics record missing rate, state-conditional missing rates,
within-state report correlation on jointly observed reports, and within-state
missing-indicator correlation. Conditioning the diagnostic on state prevents
the common state signal from being mistaken for source dependence.

## Exploratory continuation rule

Continue studying the problem only if, in condition D, M4 has lower mean Brier
and NLL than both M2 and M3, reduces mean confidence on errors or the
high-confidence error rate, and loses no more than one accuracy percentage
point against the best of M2/M3. The direction should also persist at the two
strongest shift points.

Stop or redesign if M2 or M3 matches M4, if M4 gains only under one exact
parameter setting, or if the manipulation diagnostics fail. Three development
seeds cannot establish significance, equivalence, novelty or a formal result.

