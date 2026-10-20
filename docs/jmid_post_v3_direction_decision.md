# JMID direction decision after v3

Date: 2026-10-20
Decision: **retain the scientific problem; stop promoting graph selection or graph averaging as the algorithmic contribution; pivot to identifiability and partial identification**

## Decision

JMID remains a useful research problem. Across v1, the independent v2 generator
and the ten-seed v3 diagnostic, joint modeling improved probability quality
when MNAR censorship and duplicated evidence co-occurred. V3 further showed
that an M4 graph selected from allowed training observations can beat both M23
and M2 on Brier and NLL in the frozen regime.

The current method is not strong enough to justify a formal algorithm paper.
V3's finite graph search recovered the truth perfectly and the graph-average
weights collapsed to the same single graph. Thus the experiment did not show a
new way to remain robust under genuine graph uncertainty. It showed that a
standard constrained likelihood selection problem was easy with strong copy
dependence and 3,750 structure-fit examples.

The route is therefore narrowed as follows:

1. Keep Oracle M4 and Learned M4 as conditional empirical baselines.
2. Retire Graph-Averaged M4 as a proposed contribution; retain it only as the
   frozen v3 comparator and its recorded computational cost.
3. Do not start formal confirmation of the current M4 implementation.
4. Make the next research object the identification of the hidden-state
   posterior when both the MNAR law and evidence-dependency structure are
   uncertain.

This decision does not reverse the protocol's `GO_PROTOCOL_DESIGN` result. It
uses the discretion that the gate expressly preserves: passing permits protocol
design, while the observed mechanism and novelty audit decide whether that
investment is justified.

## Next research questions

The next stage should remain mathematical and diagnostic, without a new neural
model or a larger candidate graph space.

### A. Nonidentifiability

Construct the smallest pair of data-generating models with different hidden-
state posteriors but the same observed distribution over missingness patterns
and report values. The pair should vary the MNAR mechanism and dependency graph
without granting the estimator raw reports, copy indicators or the graph.

### B. Sufficient conditions

State observable or externally verifiable assumptions under which the graph,
the missingness law, or only the target posterior becomes identifiable. Separate
graph recovery from posterior recovery: the latter may be possible without the
former.

### C. Partial identification

When point identification fails, define an uncertainty class for missingness
and dependence and derive lower and upper bounds on
`P(Z=1 | observed reports and masks)`. Any claimed interval must state the
assumptions that guarantee coverage of the true posterior.

## Stop and continuation rules

- Continue if a nontrivial posterior identification condition, impossibility
  result, or computable sharp bound can be established beyond a direct
  restatement of standard missing-data or graphical-model results.
- Stop if the target is point identified only by assuming the correct graph and
  MNAR law, or if existing results already yield the same bounds under equal or
  weaker assumptions.
- Do not add another JMID predictor until the theoretical target and its
  information requirements are frozen.

No formal experiment, additional v3 seed, graph-weight tuning or new algorithm
is authorized by this decision.
