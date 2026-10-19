# RCL journal study: frozen working protocol

## Research question

Does a source-region-conditioned mixture likelihood infer hidden situations from
contaminated multi-source observations more accurately and with better probability
quality than simpler robust fusion models under matched information and evaluation
conditions?

## Main method

The paper's main method is source-region RCL:

- reliable and compromised report-generation components;
- source-region-conditioned reliability;
- marginalization over the latent report mechanism;
- sequential posterior updating over the hidden situation.

LM-RCL, the dual-head GRU, active sensing/stopping, correlation-aware RCL, and
intent-coupled RCL are supporting or boundary studies, not contributions of this
paper. Existing NO-GO results remain in the evidence record.

## Matched first-round comparison

The first structural experiment compares:

1. single reliable mechanism;
2. global two-mechanism mixture;
3. source-level two-mechanism mixture;
4. source-region RCL.

All four variants use the same reliable and compromised emission estimates, the
same mechanism-labeled fitting records, the same hidden-situation labels, the same
validation cases, independent temperature calibration by the same frozen rule,
the same fixed commit step, and paired test scenarios. Only the mixture-prior
granularity changes among variants 2--4.

This is a fully supervised structural ablation. It does not establish performance
without mechanism labels. LM-RCL evidence addresses that separate question.

## Metrics and statistics

Primary metrics are situation accuracy and hypothesis Brier score. NLL and ECE
measure probability quality. Regret remains secondary because all methods in this
comparison share the fixed commitment protocol, latency, and information cost.

Test episodes are paired by training seed, family, hidden hypothesis, and scenario
seed. Intervals use training-seed means as the independent unit. Test episodes are
not treated as independent training replicates. Hyperparameters and temperatures
may use training and validation data only.

## Development and formal stages

The development runner exists to detect implementation errors and estimate whether
the global/source/source-region hierarchy is informative. Its results cannot be
promoted to formal evidence. A formal protocol, method set, seed set, test draws,
and decision rules will be frozen after development and before formal evaluation.

## Planned later baselines

After the granularity ablation: Student-t likelihood, reliability-weighted fusion,
and Huber or trimmed robust fusion. Sensitivity studies follow only after the
direct baselines are stable.

