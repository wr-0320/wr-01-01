# Controlled source-region heterogeneity scan v1

Status: **frozen development protocol; not formal evidence**.

## Causal question

When overall contamination opportunity is held constant, does increasing the
heterogeneity of source-region contamination cause a source-region mixture prior
to outperform global and source-level priors?

The scan changes only the probability that a red manipulation successfully
affects an eligible source-region report. Hidden-hypothesis schedules, red-policy
families, clean report generation, source noise, coverage, delay, missingness,
budgets, reliable/compromised emission fitting, validation rules, fixed decision
time, and paired scenario seeds remain unchanged.

## Frozen profiles

Every profile has mean activation probability 0.60 over the nine source-region
cells. The matrices are fixed before any scan result is inspected.

| Level | Activation matrix rows | Intended structure |
|---|---|---|
| low | `[.60,.60,.60]` repeated | no cell heterogeneity |
| medium | `.30`, `.60`, `.90` source rows | source-only heterogeneity |
| high | Latin permutations of `.20,.60,1.00` | strong within-source regional heterogeneity with equal row/column means |
| extreme | Latin permutations of `.02,.78,1.00` | stress-level within-source regional heterogeneity with equal row/column means |

Activation probability is not the ordinary sensor-noise reliability exposed in
the blue observation. It controls whether an otherwise eligible concealment,
decoy, or multi-source manipulation changes a report. All profiles consume the
same pre-drawn activation uniforms so clean stochastic variation remains paired.

## Methods and supervision

The scan compares global mixture, source mixture, and source-region RCL. Within
each profile they share the same fully mechanism-labeled fitting reports, reliable
and compromised Gaussian emissions, validation cases, temperature grid, commit
step, and test scenarios. Only the reliability-prior granularity differs.

Three development seeds are used. Confidence intervals use paired seed means and
are descriptive. An interval crossing zero does not establish equivalence or
noninferiority.

## Recorded manipulation checks

For fitting and test data, record by cell:

- configured activation probability;
- visible report count;
- activated contaminated report count;
- empirical contamination rate and reliability;
- profile mean, standard deviation, and range;
- overall empirical contamination prevalence.

The configured global mean must be identical. Realized overall rates may vary by
finite sampling and by which cells become eligible; those differences must be
reported and cannot be silently attributed to heterogeneity.

## Predeclared development decisions

These rules select the next research direction; they do not create formal claims.

1. Keep source-region conditioning as a core-contribution candidate only if its
   accuracy exceeds both grouped mixtures at both high and extreme levels, at
   least one of those two gains is at least +2 percentage points against each
   comparator, and neither Brier nor NLL is worse by more than 0.005.
2. Prefer a source-level main method if source mixture exceeds global mixture by
   at least +2 points at medium heterogeneity and rule 1 fails.
3. Shift the contribution to probability quality if rule 1 fails for accuracy but
   source-region RCL improves both Brier and NLL at high and extreme levels.
4. Otherwise treat source-region conditioning as an optional parameterization and
   retain the global contaminated-mixture likelihood as the main candidate.
5. If every method degrades sharply as heterogeneity rises, inspect manipulation
   checks, emission sample counts, and model mismatch before selecting a paper
   claim. No test seed may be added merely to obtain a preferred direction.

Correlation-aware and intent-coupled NO-GO results remain in the evidence record
and are not reinterpreted by this scan.
