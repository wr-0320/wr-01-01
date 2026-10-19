# Dual-mechanism mixture formal confirmation protocol v1

Status: **frozen and not yet run**.

## Confirmatory question

Under matched supervision and evaluation, does the global reliable/compromised
mixture improve hidden-situation probability quality over a Huber robust
likelihood without a material loss in classification accuracy?

## Frozen methods

The five methods are global dual mixture, Huber reliable likelihood, Student-t
reliable likelihood, single reliable Gaussian mechanism, and reliability-weighted
Gaussian fusion. Implementations and validation grids are frozen in
`experiments/run_rcl_mechanism_baselines.py`.

All methods receive the same training records and full cell mechanism labels.
They share reliable-emission estimates, validation cases, observation fields,
fixed inference step, and test scenarios. Methods differ in how they use the
available mechanism information. Hyperparameters and temperatures are selected
within each training seed using validation NLL only. Formal test rows may not be
used for tuning or method selection.

## Reserved evaluation

- Independent unit: training seed.
- Fresh seeds: 13001--13010 inclusive.
- Intelligence families: the frozen seven-family `RED_FAMILIES` set.
- Test cases: 100 paired episodes per family, method, and seed.
- Total: 35,000 model-episode evaluations.
- Fixed inference/commit step: 4.
- Likelihood fitting: 10 episodes per hypothesis and fitting family.
- Validation: 3 cases per fitting family.
- Output directory: `outputs/formal/rcl_mechanism_confirmation_v1`.

The run is all-or-nothing. There is no interim analysis, optional stopping, seed
replacement, or post-result extension. A failed or inconclusive result remains
the result of protocol v1.

## Metrics and estimands

For each seed, calculate the mean NLL, Brier score, accuracy, and ECE over its
paired test scenarios. Formal intervals are two-sided 95% Student-t intervals on
the ten paired seed differences, always reported as global dual mixture minus
comparator.

Primary comparison: global dual mixture versus Huber.

The confirmatory claim is supported only if all three conditions hold:

1. the upper 95% confidence bound for the NLL difference is below zero;
2. the upper 95% confidence bound for the Brier difference is below zero;
3. the lower 95% confidence bound for the accuracy difference exceeds -0.02.

This is an intersection rule, so every condition must pass. NLL and Brier test
superiority in probability quality; accuracy uses a predeclared two-percentage-
point noninferiority margin. A confidence interval crossing zero is never called
equivalence.

Student-t is the required secondary robust comparator. Single mechanism and
reliability-weighted fusion are structural controls. ECE and family-level
results are descriptive. No secondary result can rescue a failed primary rule.

## Reporting and stopping rules

Report every method and metric, all seed-level values, paired differences,
confidence intervals, selected hyperparameters, temperatures, and failures.
The development and heterogeneity NO-GO results remain linked in the paper's
evidence record. Do not add seeds, alter grids, change the accuracy margin, or
switch the primary comparator after inspecting formal output.

The reserved entry point is:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m experiments.run_rcl_mechanism_formal
```

This command has not been executed as part of freezing this protocol.
