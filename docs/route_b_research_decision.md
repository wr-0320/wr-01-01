# Research decision: transition from RCL to Route B

Date: 2026-10-20  
Decision: **stop extending the RCL main method and begin a pre-implementation
study of dynamic mechanism attribution**.

## Evidence for the transition

The formal RCL protocol returned `NOT_CONFIRMED`: compared with Huber, the global
dual mixture passed the accuracy guard but did not establish the preregistered
Brier and NLL improvements. Source-region prior granularity, correlation-aware
RCL and intent-coupled RCL also failed to produce a stable new contribution.

This does not invalidate the archived RCL evidence. It narrows its role to:

- a fixed-mechanism baseline;
- evidence that mechanism mismatch matters;
- a source of paired scenario and reporting practices;
- a documented negative result that prevents repeating the same research path.

## New working question

Can an online multi-source inference system separate a hidden-situation change
from a strategic change in the report-generation mechanism, and can it recognize
when the reports do not identify which change occurred?

The provisional method name is D-MIR. No algorithm, performance claim or novelty
claim currently exists.

## Scope boundaries

The first study includes finite hidden situations, a finite mechanism bank,
online reports, abrupt or duration-dependent mechanism changes, paired
state/mechanism change experiments and an abstention output.

The first study excludes reinforcement learning, learned active verification,
large neural sequence models, unrestricted adaptive adversaries and claims of
universal deception detection. These can be considered only after the minimal
identification problem is resolved.

## Next authorized research work

1. finish the full-text competitor audit;
2. prove or source a sufficient identification result for a restricted
   multi-source model;
3. define a computable confusability diagnostic;
4. freeze the minimal simulator protocol;
5. decide whether the project passes the first three gates.

No route-B experiment code should be added before that decision.

The first event-level audit corrected an overreach in the initial
anchor/witness lemma: distinct composite-state emissions do not guarantee that
mixtures over change-event paths are distinct. See
`route_b_identifiability_audit.md`. The active target is now a structural lower
bound on finite-window event separation, not state-level injectivity alone.

