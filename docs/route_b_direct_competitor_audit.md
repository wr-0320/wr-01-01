# Route B direct-competitor audit

Status: **preliminary map; full-text verification pending**  
Date: 2026-10-20

## Audit question

Does an existing method already connect structural identifiability assumptions
to finite-window evidence for attributing an online observation change among
situation-only, mechanism-only and simultaneous changes, with a statistically
defined refusal when those explanations overlap?

Finding a paper that only detects a changepoint, filters a switching state, or
rejects a low-confidence prediction does not answer this question. Finding a
method that provides the full chain under equal or weaker assumptions would
force Route B to narrow or stop.

## Preliminary comparison matrix

| Method family | Native latent object | Handles adversarial observations | Separates state change from observation-model change | Identifiability result | Evidence-based refusal | Route B implication |
|---|---|---:|---:|---:|---:|---|
| Classical BOCPD | run length and segment parameter | Usually no | No; it detects predictive-law change | Change detectability depends on segment models | Usually no | Required detection baseline, not the proposed solution |
| Robust BOCPD | run length with robust predictive update | Contamination/outliers, depending on model | Usually no | Robustness bounds may apply to posterior/run length | Usually no | Closest changepoint baseline; exact papers need full-text audit |
| Switching HMM/state-space model | joint physical state and regime, or one composite regime | Not inherently | Can represent both labels, but semantic factorization may be non-identifiable | Standard HMM identifiability under rank/emission assumptions | No explicit event-confusability refusal | Strongest matched latent-model baseline |
| Factorial HMM | several latent chains | Not inherently | Represents separate chains | Factor identifiability needs additional assumptions | No | Any D-MIR filter must outperform or theoretically differ from this baseline |
| IMM/MMAE | posterior over a known model bank | Fault models can be included | Attributes among enumerated models when model bank is valid | Usually model separability/observability assumptions | Low model probability, not necessarily calibrated refusal | Direct engineering baseline for known mechanisms |
| Fault detection and isolation | process state plus fault hypotheses/residuals | Faults rather than strategic attacks | Often explicitly separates process and sensor faults | Structured residual/isolability conditions | May declare unisolable faults | Potentially the closest conceptual competitor; must receive priority in full-text review |
| Sparse secure state estimation | physical state under sparse attacked sensors | Yes, under attack-budget assumptions | Primarily recovers state; attack support may also be estimated | Sparse observability/error-correction conditions | Often failure when attack budget is exceeded, not selective event attribution | Supplies likely non-novel parts of a sparse-source theorem |
| Byzantine quickest change detection | changepoint with compromised sensors | Yes | Usually protects detection of a specified process change rather than deciding whether the change is physical or adversarial | Delay/false-alarm guarantees under attacker models | May use robust aggregation | Direct competitor for distributed detection guarantees |
| Adversarial HMM manipulation | HMM posterior under corrupted observations | Yes | Studies how an attacker changes inference | Vulnerability/attack optimization rather than attribution in the known lead | No | Motivates the threat model; exact scope requires full-text check |
| Selective classification/reject option | prediction plus reject action | Not inherently | No | Risk-coverage or reject-cost guarantees | Yes | Supplies decision theory, not change attribution |
| Conformal/e-value sequential testing | sequential evidence under validity assumptions | Sometimes robust variants | Only if separate hypotheses are supplied | Type-I/error or anytime-valid guarantees | Can continue sampling/refrain from rejecting | Candidate tool for a refusal guarantee; not by itself an attribution model |

## Closest collision risks

### Sensor fault isolation

Fault detection and isolation already asks whether an innovation comes from
process evolution, a sensor fault or a model fault. If its structured residual
and isolability theory already covers unknown strategic emission switches and
ambiguous-event refusal, Route B's proposed theory may be a domain translation.
The review must therefore search `fault isolability`, `unknown input observer`,
`process fault versus sensor fault`, and `multiple-model fault diagnosis`, not
only deception and adversarial-learning keywords.

### Sparse secure estimation

The \(2k\)-style source-redundancy condition is established error-correction
logic. Route B cannot claim it as new. A possible increment would have to concern
online attribution among event mixtures, finite-window mechanism evidence and
refusal when the admissible event-law sets intersect.

### Factorial and switching latent models

A factorial HMM can encode \(z_t\) and \(m_t\) immediately. The new work must
show why posterior filtering of that model lacks a defensible semantic
attribution, and then add a checkable event-separation certificate or guarantee.
A different inference implementation alone is insufficient.

### Controlled sensing and active verification

Choosing an action to maximize KL, Chernoff information or expected information
gain between hypotheses has substantial prior art. Active verification should
not become the main claim unless the review identifies a strategic-deception
constraint or guarantee missing from controlled sensing and active hypothesis
testing.

## Paper-level falsification tests

For each direct competitor, the review must try to falsify Route B by answering:

1. Can its latent variables be mapped one-to-one to \(z_t\) and \(m_t\)?
2. Does it allow both chains to change at the same boundary?
3. Does it marginalize over multiple successor states, rather than assume one
   known post-change model?
4. Does it give event-level isolability conditions instead of only state or
   parameter identifiability?
5. Is its evidence finite-window and computable online?
6. Is refusal tied to a risk/error guarantee under overlapping event laws?
7. Does it use the same or weaker trusted-source, attack-budget and supervision
   assumptions?

If one method answers yes to all seven, Route B must identify a narrower theorem
or stop. Combining features from different papers is not automatically novel;
the combination must solve an interaction that none of them can handle.

## Required full-text clusters

The next database-enabled review must cover and citation-chain:

1. Bayesian and robust online changepoint detection;
2. switching/factorial HMM identifiability;
3. process-fault versus sensor-fault isolability;
4. unknown-input observers and multiple-model fault diagnosis;
5. sparse/Byzantine attack-resilient state estimation;
6. Byzantine quickest change detection;
7. active sequential hypothesis testing and controlled sensing;
8. selective prediction and anytime-valid sequential evidence.

For each cluster, record the exact theorem assumptions, whether the guarantee is
finite-sample or asymptotic, whether mechanism labels are observed in training,
and whether code is available.

## Current conclusion

The proposed connection

\[
\text{structural condition}\to\text{finite-window separation}
\to\text{online event attribution}\to\text{refusal}
\]

is not yet shown to be covered by one known method, but the current environment
does not permit the full-text search needed to call it a literature gap. Fault
isolation and sparse secure estimation are the highest-risk neighboring areas.
Until those are audited, the novelty status remains **UNRESOLVED**.

## Internal-prior-work check

The original project already proved the standard balanced-prior relation
between total variation of visible-history laws and minimum binary decision
error, and empirically catalogued exact trajectory aliasing. Route B may reuse
that result with attribution, but cannot present it as new theory. The open
step is to derive event-law separation from weaker operational structure rather
than assume or measure the separation after the fact.

The finite-alphabet candidate theorem additionally uses classical code-distance
and binary-channel arguments. The full-text audit must check whether its event-
class formulation adds anything beyond sparse fault isolability or secure state
estimation. Until then, the theorem is evidence of internal consistency rather
than evidence of novelty.

The joint-window theorem heightens this collision risk: its parity syndrome is
a structured residual, and its four event classes resemble process-change and
sensor-fault isolation. The direct audit must compare event isolability with
unknown fault time, sparse sensor faults, residual signatures and simultaneous
process/sensor changes. A different application vocabulary would not be a
theoretical increment.

