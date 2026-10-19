# Controlled heterogeneity development v1

This is three-seed development evidence, not a formal confirmation.
All profiles preserve a configured mean manipulation activation of 0.60.

| Profile | Method | Accuracy | Brier | NLL | ECE |
|---|---|---:|---:|---:|---:|
| low | global_mixture | 82.90% | 0.1878 | 0.2673 | 0.0317 |
| low | source_mixture | 83.00% | 0.1877 | 0.2670 | 0.0306 |
| low | source_region_rcl | 82.86% | 0.1875 | 0.2668 | 0.0312 |
| medium | global_mixture | 82.43% | 0.1836 | 0.2622 | 0.0265 |
| medium | source_mixture | 82.57% | 0.1829 | 0.2611 | 0.0250 |
| medium | source_region_rcl | 82.57% | 0.1828 | 0.2610 | 0.0246 |
| high | global_mixture | 83.33% | 0.1868 | 0.2676 | 0.0192 |
| high | source_mixture | 83.33% | 0.1868 | 0.2676 | 0.0197 |
| high | source_region_rcl | 83.43% | 0.1867 | 0.2676 | 0.0211 |
| extreme | global_mixture | 83.14% | 0.1865 | 0.2688 | 0.0206 |
| extreme | source_mixture | 83.14% | 0.1866 | 0.2689 | 0.0212 |
| extreme | source_region_rcl | 82.95% | 0.1873 | 0.2707 | 0.0216 |

## Source-region paired differences

| Profile | Comparator | Accuracy pp | Brier | NLL |
|---|---|---:|---:|---:|
| low | global_mixture | -0.05 [-0.25,+0.16] | -0.0003 | -0.0005 |
| low | source_mixture | -0.14 [-0.50,+0.21] | -0.0002 | -0.0003 |
| medium | global_mixture | +0.14 [-0.57,+0.85] | -0.0008 | -0.0012 |
| medium | source_mixture | +0.00 [+0.00,+0.00] | -0.0001 | -0.0001 |
| high | global_mixture | +0.10 [-0.93,+1.12] | -0.0002 | +0.0000 |
| high | source_mixture | +0.10 [-0.93,+1.12] | -0.0002 | +0.0000 |
| extreme | global_mixture | -0.19 [-0.60,+0.22] | +0.0007 | +0.0018 |
| extreme | source_mixture | -0.19 [-0.73,+0.35] | +0.0007 | +0.0018 |

## Predeclared development decision

Decision: **global_mixture_main_source_region_optional**.

This decision follows `docs/heterogeneity_scan_protocol.md`. An interval crossing zero is not interpreted as equivalence. Existing correlation-aware and intent-coupled NO-GO results remain unchanged.
