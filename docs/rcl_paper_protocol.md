# RCL journal study: revised working protocol

## Research question

When a report may arise from either a reliable or contaminated generation
mechanism, does a dual-mechanism mixture likelihood improve hidden-situation
probability inference over single-mechanism and robust-likelihood alternatives?

## Main method candidate

The paper's main candidate is a global dual-mechanism mixture likelihood:

- separate reliable and compromised report-generation components;
- a fitted global prior over the latent report mechanism;
- marginalization over mechanism uncertainty;
- sequential posterior updating over the hidden situation.

Source-region reliability-prior granularity is an optional parameterization. The
controlled heterogeneity scan did not show a stable benefit from that finer
prior. The reliable and compromised emission distributions remain conditioned on
source and region, so this decision does not remove all source-region structure.

LM-RCL, the dual-head GRU, active sensing/stopping, correlation-aware RCL, and
intent-coupled RCL remain supporting or boundary studies. Their existing NO-GO
results stay in the evidence record.

## Matched mechanism comparison

The comparison contains:

1. single reliable Gaussian mechanism;
2. global reliable/compromised Gaussian mixture;
3. Student-t reliable likelihood;
4. Huber reliable pseudo-likelihood;
5. reliability-weighted Gaussian fusion.

All methods use the same training and validation records, full mechanism-label
availability, reliable-emission estimates, observation fields, fixed inference
time, and paired test scenarios. Robust model hyperparameters and every
temperature are selected by validation NLL only. Test data cannot affect model
selection.

This is a fully supervised structural comparison. It asks whether explicitly
marginalizing two generation mechanisms adds value when information availability
is matched. It does not establish performance without mechanism labels; the
separate LM-RCL study addresses that question.

## Metrics and statistics

NLL and Brier score are the confirmatory probability-quality metrics. Accuracy
provides a noninferiority guard and remains a key reported outcome. ECE and
family-level results are descriptive.

Test cases are paired by training seed, family, hidden hypothesis, and scenario
seed. The independent statistical unit is the training seed. Episodes are never
treated as independent training replicates. Formal intervals use paired seed
means.

## Evidence stages

The three-seed granularity, controlled heterogeneity, and robust-baseline runs are
development evidence. They selected the global mixture and Huber primary
comparator but cannot establish the final claim.

The formal method set, ten fresh seeds, run size, metrics, confidence intervals,
success rule, and stopping rule were frozen in
`docs/formal_confirmation_protocol_v1.md`. The experiment was executed once and
returned `NOT_CONFIRMED`; see `docs/formal_confirmation_result_v1.md`. Its formal
output remains separate from all development evidence and must not be rerun or
pooled with development outputs.
