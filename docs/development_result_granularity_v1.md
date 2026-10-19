# RCL granularity development v1

Status: **development evidence only; not a formal confirmation**.

The run used three new fitting seeds (9101--9103), seven report families, 100
paired scenarios per family and method, and four matched methods, for 8,400 test
episodes. Each method used the same mechanism-labeled fitting records, reliable
and compromised Gaussian emissions, validation cases, temperature-selection
rule, fixed commit step, and scenario seeds.

| Method | Accuracy | Brier | NLL | ECE | Regret |
|---|---:|---:|---:|---:|---:|
| Single mechanism | 77.57% | 0.2672 | 0.4235 | 0.0543 | 5.861 |
| Global mixture | 81.52% | 0.1993 | 0.2873 | 0.0452 | 5.110 |
| Source mixture | 81.48% | 0.1995 | 0.2876 | 0.0456 | 5.120 |
| Source-region RCL | 81.24% | 0.1994 | 0.2853 | 0.0430 | 5.165 |

Source-region RCL minus global mixture accuracy was −0.29 percentage points with
a three-seed 95% interval of [−1.22, +0.65] points. Source-region RCL minus the
source-level mixture was −0.24 points, interval [−1.32, +0.85]. Brier, NLL, and
ECE differences versus both grouped mixtures were also small and their intervals
crossed zero.

The mixture family as a whole improved clearly over the matched single mechanism
in probability quality: source-region RCL reduced Brier by 0.0678 and NLL by
0.1382, with both three-seed intervals excluding zero. Its accuracy gain over the
single mechanism was +3.67 points, but the three-seed interval [−1.48, +8.81]
crossed zero.

## Interpretation

This run supports the value of reliable/compromised mixture modeling, but it does
not support an additional task-performance claim for source-region reliability
over a global or source-level rate. The fitted source-region reliabilities were
also tightly clustered (approximately 0.949--0.964 in seed 9101), so the current
simulator and fitting protocol provide little heterogeneity for a granularity
advantage to exploit.

The result changes the likely paper claim. The next development decision should
test whether controlled source-region heterogeneity produces a predeclared and
interpretable crossover. If it does not, the main method should be framed as a
contaminated mixture likelihood, with source-region conditioning treated as an
optional parameterization rather than a validated core contribution.

Do not add seeds to turn this development comparison into a desired positive
result. Freeze the next heterogeneity/sensitivity protocol first, then reserve
fresh seeds for confirmation only if the development result identifies a stable,
scientifically meaningful regime.

