# Dynamic-lineage posterior certificate: frozen theory protocol

Status: **one bounded theorem audit; algorithm and simulation prohibited**

Date: 2026-10-20

## Sole target

Determine whether observable timing, partial provenance or trusted audit data
can support a finite-sample, nonvacuous bound on the error of a situation
posterior when the evidence-lineage structure is unknown.

The audit has one primary target. Anytime failure alarms are deferred. They may
not be presented as a second contribution if the posterior certificate fails.

## Required objects

1. a finite or explicitly controlled class of dynamic lineage paths;
2. an observable lineage signal and a simultaneous-coverage statement for a
   lineage confidence set;
3. a set-valued posterior over all retained lineage paths;
4. an error bound for a reported point posterior;
5. sufficient conditions under which the bound is narrower than the vacuous
   interval;
6. a counterexample showing when copying and common causes cannot be separated.

## Information boundary

Permitted lineage signals are report timestamps, transmission delays, partial
provenance markers and a prespecified amount of trusted audit data. Hidden edge
labels, the true graph and target labels unavailable at deployment may not be
used to form the certificate.

If report contents are reused to select a lineage set, the selection rule must
itself have a valid finite-sample coverage proof. Point estimation followed by
conditioning as if the selected graph were fixed is inadmissible.

## Frozen decisions

- **THEORY-FEASIBLE:** a finite-sample coverage theorem and a nonvacuity theorem
  hold under observable, auditable conditions.
- **CONDITIONAL:** the theorem is correct but follows directly from generic
  confidence-set inversion and robust posterior envelopes; no originality is
  claimed and no prototype is authorized.
- **STOP:** observationally equivalent lineage models force a vacuous posterior
  range under every operationally defensible information condition.
- **PROTOTYPE-ELIGIBLE:** requires THEORY-FEASIBLE plus a theorem-level increment
  not supplied by the six frozen competitors under equal or weaker assumptions.

The work package ends after one theorem draft and one decision. It must not grow
into a new literature matrix, alarm theorem, model implementation or simulator.
