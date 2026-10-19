# Fixed-Prior RCL Formal Evaluation

Training seeds [3001, 3002, 3003, 3004, 3005]; 100 paired episodes per method × family × seed; 28,000 total episodes. Intervals use training-seed means, not individual episodes. All fitting and calibration precede evaluation for each seed. Smoke only: False.

| Method | Correct commitment | Brier | ECE | NLL | Regret | Latency | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| multi_source | 66.7% | 0.406 | 0.064 | 0.725 | 7.92 | 5.00 | 0.60 |
| ride_v1 | 67.9% | 0.439 | 0.082 | 0.846 | 8.45 | 4.84 | 1.43 |
| rcl_fixed | 81.9% | 0.201 | 0.043 | 0.291 | 5.04 | 5.00 | 0.60 |
| bayesian_fusion | 54.0% | 0.756 | 0.322 | 1.566 | 12.96 | 10.00 | 2.55 |
| pomcp_like | 54.0% | 0.756 | 0.323 | 1.566 | 12.96 | 10.00 | 2.55 |
| active_inference | 54.0% | 0.756 | 0.322 | 1.566 | 12.96 | 10.00 | 2.55 |
| rule_active | 56.4% | 0.780 | 0.360 | 1.636 | 11.15 | 6.05 | 1.74 |
| single_mechanism | 77.8% | 0.267 | 0.049 | 0.446 | 5.82 | 5.00 | 0.60 |

Primary paired accuracy difference: **+15.17 pp** (95% CI +10.63 to +19.71 pp).
Primary frozen support rule: **SUPPORTED**.

| Family | Multi-source | RCL-fixed | Difference | Single mechanism |
|---|---:|---:|---:|---:|
| honest | 77.8% | 81.4% | +3.6 pp | 81.4% |
| concealment | 63.4% | 79.0% | +15.6 pp | 65.4% |
| decoy | 76.0% | 83.0% | +7.0 pp | 77.6% |
| multi_source | 63.8% | 81.0% | +17.2 pp | 78.0% |
| delayed | 77.8% | 80.6% | +2.8 pp | 80.6% |
| switch | 64.4% | 84.0% | +19.6 pp | 84.0% |
| adaptive | 44.0% | 84.4% | +40.4 pp | 77.6% |

## Seen and unseen families

| Method | Seen | Unseen | Seen minus unseen |
|---|---:|---:|---:|
| multi_source | 73.8% | 57.4% | +16.3 pp |
| ride_v1 | 73.6% | 60.3% | +13.3 pp |
| rcl_fixed | 81.0% | 83.1% | -2.1 pp |
| bayesian_fusion | 56.8% | 50.3% | +6.4 pp |
| pomcp_like | 56.7% | 50.5% | +6.2 pp |
| active_inference | 56.8% | 50.3% | +6.4 pp |
| rule_active | 60.9% | 50.3% | +10.6 pp |
| single_mechanism | 76.2% | 79.9% | -3.6 pp |

## Interpretation and limits

Mixture versus matched single mechanism accuracy: +4.11 pp (95% CI +1.80 to +6.43 pp). This diagnostic separates mixture modeling from Gaussian emission modeling; it is secondary and has no multiplicity correction.
RCL-fixed uses training-only simulator mechanism labels to fit reliable/compromised emissions. It has additional mechanism supervision compared with the neural Multi-source baseline, so the experiment does not isolate architecture under equal supervision. Its compromised mechanism is hypothesis independent and report features are modeled as independent Gaussians with a validation-fitted temperature.
Only RCL-fixed and Multi-source share the fixed passive decision schedule. Other baselines retain their original active sensing and stopping rules; compare their cost and latency alongside accuracy. POMCP-like is the repository's one-step categorical rollout baseline, not a full POMCP solver.
Family designs were used during development; only the training, calibration, and episode draws are fresh. This is generalization to held-out scenario draws and families excluded from fitting, not proof of robustness to newly invented deception families.
RCL has no deception classification head; no deception-classification superiority is claimed. This result also does not support gains from learned reliability, which is absent.
All older formal and development outputs remain frozen. This study concerns an abstract adversarial intelligence simulator.

## Artifacts and final verification

Per-seed primary accuracy gains: +11.00 pp, +12.43 pp, +17.14 pp, +20.14 pp, +15.14 pp.

All 28,000 episodes and 280 method-family-seed cells passed completeness checks; 3,500 paired scenarios each contain all eight methods. RCL-fixed, single mechanism, and Multi-source have identical per-scenario decision latency and information cost. No episode timed out. The regression suite passed 187 tests before the frozen evaluation.

Figures: [overview](formal_overview.png), [family breakdown](deception_family_breakdown.png). Both are available as PDF, and `plot_results.py` regenerates them from the saved tables.
