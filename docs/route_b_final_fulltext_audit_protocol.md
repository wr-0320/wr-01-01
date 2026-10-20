# Route B final full-text competitor audit protocol

Status: **frozen; evidence collection blocked by the current network policy**  
Freeze date: 2026-10-20

## Decision question

This is the final audit of the passive Route B theory. It asks whether the
proposed separation of situation changes, observation-mechanism changes and
simultaneous changes supplies a theorem or decision rule that cannot already be
obtained from fault isolation, secure state estimation, attack identification
or set-membership diagnosis under equal or weaker assumptions.

No D-MIR implementation and no additional syndrome theorem are permitted while
this audit is open.

## Fixed scope and effort limit

The audit has four clusters:

1. structured residuals and fault isolability;
2. sparse secure state estimation and attack identification;
3. set-membership and multiple-model diagnosis with unknown onset;
4. causal or adaptive attacks, simultaneous process/sensor changes, and
   statistically justified refusal.

The evidence set is limited to 12--15 closest primary works. Review follows
backward and forward citations only until two consecutive passes add no work
that is closer on the comparison fields below. Survey papers and books may
locate primary results, but cannot replace the original theorem and proof.

## Evidence rule

A work counts as reviewed only when the auditor has read the original full text
and recorded:

- exact theorem, proposition or section identifiers;
- model and information structure;
- nominal dynamics and observation assumptions;
- unknown onset and simultaneous-change support;
- attacked-source support and whether it may vary over time;
- whether an attack may adapt to past observations or residuals;
- the isolation/recovery guarantee and its statistical unit;
- the treatment of overlapping or unisolable hypotheses.

Metadata, abstracts, search snippets and secondary summaries can nominate a
work. They cannot support a coverage or novelty verdict. An empty matrix cell
means unverified, not absent from the cited work.

## Comparison target

The current candidate increment consists of all of the following:

1. online attribution among situation-only, mechanism-only and simultaneous
   changes with an unknown boundary and nuisance pre-state;
2. an observation-adaptive, non-anticipating manipulation policy;
3. a finite-window probabilistic guarantee after nuisance marginalization;
4. a computable refusal condition tied to overlap of admissible event laws.

The audit must distinguish a genuinely new interaction from a direct
composition of established residual generation, sparse recovery, model-bank
isolation and reject-option rules.

## Predeclared decision

**GO** only if the completed theorem matrix shows that no closest primary work
covers the four-part target under equal or weaker assumptions, and the remaining
gap can be written as one precise theorem with operational assumptions and a
computable diagnostic.

**NO-GO** if an existing result covers the target under equal or weaker
assumptions, or if the only difference is the intelligence/deception vocabulary,
a four-event relabeling, or a routine combination of established components.
The passive D-MIR route is then archived. The next activity may be a separate
prior-art audit of active evidence verification; it is not automatically an
authorization to build that method.

**INCONCLUSIVE-BLOCKED** applies when required primary full texts cannot be
obtained. This status keeps implementation at NO-GO and does not count as a
literature verdict.

## Completion artifact

`route_b_fulltext_audit_matrix.csv` is the controlling evidence ledger. A final
decision must cite the relevant rows and exact theorem identifiers. The written
decision must also list the strongest result against continuation and the
strongest surviving difference, even if the final decision is NO-GO.

