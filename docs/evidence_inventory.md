# Existing evidence, gaps, and reuse plan

| Study | Existing evidence | Status for RCL paper | Reuse |
|---|---|---|---|
| Fixed-prior RCL | 5 seeds, 7 families, 8 methods, 28,000 episodes; RCL 81.9%; matched single mechanism 77.8% | Core prior evidence, but supervision and structural attribution remain incomplete | Reuse protocol, simulator, fit/calibration rules, paired scenario construction, and reporting code |
| LM-RCL | 10 seeds, 49,000 episodes; 0%-label LM-RCL 81.9%, supervised RCL 82.2%; preregistered noninferiority supported | Supporting evidence about mechanism-label dependence, not the main method | Reuse as supervision-boundary evidence; do not pool estimates with the 5-seed study |
| Correlation-aware RCL | 3 development seeds, 8,400 episodes; +0.43 pp, CI crosses zero; NO-GO | Boundary evidence; no positive method claim | Retain unchanged; later scan dependency strength instead of relabeling this result |
| Intent-coupled RCL | 3 development seeds; +0.24 pp versus standard RCL, CI crosses zero; NO-GO | Boundary evidence for hypothesis-dependent contamination | Retain unchanged; use only to motivate limits or a predeclared later stress test |
| Global mixture | Missing before this extraction | Highest-priority structural baseline | Implemented in this repository with matched emissions and supervision |
| Source-level mixture | Missing before this extraction | Highest-priority structural baseline | Implemented in this repository with matched emissions and supervision |
| Controlled reliability heterogeneity | 3 development seeds, 4 matched profiles, 25,200 model-episode evaluations; actual cell-rate SD rose from 0.0155 to 0.0456 while overall test contamination stayed near 6% | Source-region prior showed no stable accuracy, Brier, or NLL advantage; predeclared decision favors global mixture as the main candidate | Retain as development evidence; do not claim equivalence and do not expand to formal seeds before freezing the revised mixture-likelihood claim |
| Student-t likelihood | 3 matched development seeds; 80.62% accuracy, 0.2149 Brier, 0.3208 NLL | Implemented direct robust-generative comparator; interval versus global mixture crosses zero | Keep as required secondary formal comparator |
| Reliability-weighted fusion | 3 matched development seeds; 77.48% accuracy, 0.2666 Brier, 0.4220 NLL | Clean matched linear weighting does not reproduce the dual-mixture result | Keep as a structural control; do not substitute the older learned attenuation study |
| Huber robust likelihood | 3 matched development seeds; 81.05% accuracy, 0.2129 Brier, 0.3195 NLL | Strongest development robust comparator; interval versus global mixture crosses zero | Frozen as the primary formal comparator |
| Formal dual mixture versus robust baselines | 10 reserved seeds, 35,000 paired model-episode evaluations; global mixture 82.20% accuracy, 0.2148 Brier, 0.3367 NLL; Huber 80.74%, 0.2196, 0.3321 | Pre-registered decision `NOT_CONFIRMED`: accuracy guard passed; NLL and Brier superiority conditions failed | Report without adding seeds or changing the primary comparator; retain raw formal rows and integrity manifest |
| Contamination-rate scan | Missing | Sensitivity gap | Single-factor scan after direct baselines |
| Reliability-error scan | Missing | Sensitivity gap | Bias and random perturbation after direct baselines |
| Correlation-strength scan | One fixed strength only | Sensitivity gap | Extend from independence to several frozen strengths |
| Missingness/delay and distribution shift | Family-level variation exists, controlled scans do not | Robustness gap | Add only after the core comparison is frozen |

Evidence snapshots are copied under `evidence/source_snapshot/`. They are historical
inputs and must never be overwritten by new runs. New development output goes under
`outputs/development/`; future formal output must use a new frozen directory.
