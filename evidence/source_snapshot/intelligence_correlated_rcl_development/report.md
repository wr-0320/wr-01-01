# Correlation-aware RCL development

A shared binary contamination event is introduced per region and report cycle. Emission distributions and unconditional cell rates match independent RCL exactly. The primary comparison shares the independent temperature; separately recalibrated correlation-aware inference is secondary. Source correlation is modeled, but temporal/cross-region coupling is not. Offline fitting labels indicate simulator intervention targets, not guaranteed report falsity.

| Method | Correct commitment | Brier | ECE | NLL | Regret | Latency | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| single_mechanism | 78.2% | 0.243 | 0.037 | 0.379 | 5.735 | 5.00 | 0.60 |
| independent_rcl | 82.1% | 0.195 | 0.032 | 0.276 | 5.002 | 5.00 | 0.60 |
| correlated_shared_temperature | 82.5% | 0.186 | 0.032 | 0.262 | 4.920 | 5.00 | 0.60 |
| correlated_calibrated | 82.5% | 0.196 | 0.046 | 0.273 | 4.920 | 5.00 | 0.60 |

Primary accuracy gain: +0.43 pp, 95% three-seed interval [-0.19, +1.04] pp.
Offline correlated-label NLL reduction: +0.5162 per report cycle.
Frozen development gate: **NO-GO**.

| Family | Independent | Correlated, same temperature | Correlated, recalibrated |
|---|---:|---:|---:|
| honest | 84.7% | 85.0% | 85.0% |
| concealment | 78.0% | 83.7% | 83.7% |
| decoy | 81.3% | 80.3% | 80.3% |
| multi_source | 82.3% | 81.7% | 81.7% |
| delayed | 83.7% | 84.3% | 84.3% |
| switch | 82.3% | 80.7% | 80.7% |
| adaptive | 82.3% | 82.0% | 82.0% |

These are three fresh development seeds (5101–5103), not a formal confirmation. No seen/unseen family design or algorithm parameter was changed after test inspection. Confidence intervals are unadjusted and secondary family comparisons are descriptive. Better offline joint label likelihood need not imply better hypothesis inference. A positive result would still require literature review before a novelty claim, and frozen held-out confirmation before paper-level performance claims.
