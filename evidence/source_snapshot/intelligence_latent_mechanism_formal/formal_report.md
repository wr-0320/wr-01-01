# LM-RCL frozen formal confirmation

| Method | Mechanism labels | Accuracy | Brier | ECE | NLL | Regret | Latency | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| multi_source | no cell labels | 66.1% | 0.415 | 0.062 | 0.755 | 8.044 | 5.00 | 0.60 |
| unlabeled_single | 0% | 77.5% | 0.288 | 0.073 | 0.622 | 5.872 | 5.00 | 0.60 |
| supervised_rcl_100 | 100% | 82.2% | 0.195 | 0.038 | 0.278 | 4.974 | 5.00 | 0.60 |
| lm_rcl_0 | 0% | 81.9% | 0.201 | 0.038 | 0.287 | 5.036 | 5.00 | 0.60 |
| lm_rcl_10 | 10% | 82.2% | 0.199 | 0.036 | 0.283 | 4.982 | 5.00 | 0.60 |
| lm_rcl_25 | 25% | 82.0% | 0.198 | 0.038 | 0.284 | 5.020 | 5.00 | 0.60 |
| sparse_supervised_rcl_10 | 10% | 79.5% | 0.255 | 0.056 | 0.377 | 5.495 | 5.00 | 0.60 |

Primary formal result: **SUPPORTED**.
LM-RCL 0% minus supervised RCL accuracy: -0.33 pp, two-sided 95% paired-seed CI [-0.81, +0.15] pp; preregistered noninferiority margin −2.00 pp.
Key secondary LM-RCL 10% minus sparse-supervised 10%: +2.70 pp, 95% CI [+1.31, +4.09] pp.

The primary test concerns loss of task performance after removing cell-level mechanism labels. It does not show that hypothesis labels are unnecessary: all likelihood methods receive the true operational-hypothesis label during offline training. Multi-source retains its original episode-level deception supervision and is contextual rather than an equal-supervision control. Held-out mechanism-label metrics are diagnostic only. Novelty still requires literature comparison.
