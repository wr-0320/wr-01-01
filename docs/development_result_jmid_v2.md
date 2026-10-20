# JMID v2 independent-generator falsification result

Date: 2026-10-20  
Decision: **core falsification passed; scope restricted; no formal confirmation**

## What changed from v1

The v2 generator was implemented separately. It produces dependence through
directed relay copying and missingness through censorship of the uncensored
report content. It does not use v1's state-conditional availability table or
shared availability draw.

The fixed M23 comparator combines M2 and M3 as
`logit(M2) + logit(M3) - logit(M1)`. The rule and unit coefficients were frozen
before the run. M23 fits about 55 effective categorical parameters, compared
with 49 for M4, so M4's result cannot be attributed to a larger nominal
parameter budget.

## Mechanism check

In condition D, the realized copy rate was 0.8510 and within-state uncensored
report correlation was 0.8511. Missingness was 0.5141 for positive raw reports
and 0.1481 for negative raw reports. Missing-mask correlation was only 0.0917,
consistent with masks drawn independently conditional on report content rather
than the shared availability variable used in v1.

## Matched-condition result

| Condition | Method | Accuracy | Brier | NLL | ECE | Wrong confidence | High-confidence error |
|---|---|---:|---:|---:|---:|---:|---:|
| A | M2 | 0.9075 | 0.0645 | 0.2148 | 0.0080 | 0.7072 | 0.0229 |
| A | M3 | 0.9074 | 0.0644 | 0.2148 | 0.0082 | 0.7084 | 0.0224 |
| A | M23 | 0.9081 | 0.0645 | 0.2152 | 0.0066 | 0.7115 | 0.0215 |
| A | M4 | 0.9069 | 0.0647 | 0.2157 | 0.0095 | 0.7104 | 0.0213 |
| B | M2 | 0.8948 | 0.0777 | 0.2589 | 0.0127 | 0.7434 | 0.0191 |
| B | M3 | 0.8633 | 0.0960 | 0.3111 | 0.0316 | 0.7248 | 0.0205 |
| B | M23 | 0.8954 | 0.0782 | 0.2601 | 0.0124 | 0.7473 | 0.0189 |
| B | M4 | 0.8942 | 0.0781 | 0.2596 | 0.0090 | 0.7368 | 0.0176 |
| C | M2 | 0.8586 | 0.1119 | 0.3983 | 0.0622 | 0.8240 | 0.0816 |
| C | M3 | 0.8623 | 0.1032 | 0.3355 | 0.0098 | 0.7521 | 0.0153 |
| C | M23 | 0.8623 | 0.1033 | 0.3358 | 0.0095 | 0.7525 | 0.0151 |
| C | M4 | 0.8619 | 0.1034 | 0.3359 | 0.0105 | 0.7532 | 0.0149 |
| D | M2 | 0.8529 | 0.1149 | 0.3946 | 0.0551 | 0.8076 | 0.0535 |
| D | M3 | 0.7609 | 0.1390 | 0.4353 | 0.1147 | 0.6859 | 0.0341 |
| D | M23 | 0.8323 | 0.1234 | 0.3988 | 0.0865 | 0.7553 | 0.0356 |
| D | M4 | **0.8635** | **0.1059** | **0.3421** | **0.0129** | 0.7660 | **0.0126** |

In condition D, M4 improved on M23 by 3.12 accuracy percentage points, 0.0175
Brier and 0.0567 NLL. M4 had lower Brier and NLL than each comparator in each of
the three seeds. Its accuracy exceeded every comparator, so the accuracy guard
passed.

Overconfidence remains metric-specific. M4 had the lowest high-confidence error
rate, but its mean confidence on errors was higher than M3 and M23. The result
does not support a claim that M4 dominates every baseline on all overconfidence
measures.

## Transfer tests

| Test | M23 Brier / NLL | M4 Brier / NLL | Result |
|---|---:|---:|---|
| strength 0.75 | 0.1073 / 0.3510 | **0.1045 / 0.3414** | M4 better in all three seeds |
| strength 1.0 | 0.1187 / 0.3865 | **0.1152 / 0.3724** | M4 better in all three seeds |
| negative-report censorship | 0.2319 / 0.7275 | **0.1673 / 0.5193** | M4 better in all three seeds |
| conflict censorship | 0.1266 / 0.4040 | **0.1115 / 0.3551** | M4 better in all three seeds |
| rotated dependency graph | **0.1247 / 0.4023** | 0.1332 / 0.4174 | M4 worse in all three seeds |

The unseen-censorship results do not mean M4 learned the new missingness law;
all models were evaluated without refitting and absolute performance changed.
They show only that M4 retained the best comparison within these two transfers.

The rotated-graph result is a clear failure boundary. M4 uses the declared
training groups and becomes misspecified when copying follows different pairs.
Its accuracy also fell to 0.8068, below M23's 0.8327 and M2's 0.8425.

## Decision

JMID v2 passes its predeclared core continuation rule: the v1 gain survived a
different generator, M23 did not match M4, condition-D seed directions were
consistent, and the strong intensity points repeated the result.

This permits continued study of the research problem only under an explicit
scope: the dependency structure is known or reliably estimated. It does not
authorize a formal confirmation experiment yet. The next decision should test
dependency-graph uncertainty directly and compare against structure-learning or
graph-averaged baselines. If a simple graph-robust baseline removes the gain,
the joint-table route should stop. Increasing seed count or model complexity
before resolving this boundary is not justified.

