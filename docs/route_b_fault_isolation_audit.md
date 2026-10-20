# Route B fault-isolation and contribution audit

Status: **decisive preliminary audit; full-text theorem comparison still
required**  
Date: 2026-10-20

## Preliminary verdict

The current joint-window syndrome theorem is structurally very close to
classical model-based fault detection and isolation. On the evidence available,
it is not yet sufficient to support an independent high-level theory paper.

The strongest collision is not BOCPD. It is parity-space residual generation
and fault isolability under sparse sensor faults.

## Direct concept mapping

| Route B object | Fault-isolation/security analogue | Novelty risk |
|---|---|---|
| Clean signature code \(\mathcal C\) | nominal behavior/model-consistency space | Established |
| Parity check \(H\) | parity-space or structured residual generator | Established |
| Syndrome \(Hx_t\) | residual/fault signature | Established |
| Unknown sparse mask | sparse sensor fault or attack vector | Established |
| \(d>2k\) | error-correction or sparse-observability condition | Established |
| At least \(\ell\) active times | persistent fault/detectability condition | Established form |
| Guard length and one boundary | dwell-time/change-window assumption | Established form |
| Situation transition inside the clean code | process/mode change consistent with the nominal model family | Established concept |
| Mechanism transition leaving the code | sensor/model fault producing a residual | Established concept |
| Four event classes | no fault, process change, sensor fault, simultaneous process/sensor change | Direct overlap |
| Nearest event set | minimum-distance residual classification/model-bank isolation | Established form |

Calling the process state a situation and the sensor fault strategic deception
does not create a theoretical distinction.

## What classical theory is likely to cover

The full-text audit must verify, rather than assume, the following likely
overlaps:

1. **Structured residuals:** residual generators designed to be insensitive to
   nominal state/process variation and sensitive to specified sensor-fault
   directions.
2. **Fault isolability:** rank, signature or subspace conditions ensuring that
   different faults or fault combinations produce distinguishable residuals.
3. **Unknown fault time:** sequential residual monitoring, change detection and
   multiple-model diagnosis routinely treat onset time as unknown.
4. **Sparse sensor attacks:** secure estimation uses redundancy and
   \(2k\)-style observability to recover state under unknown attacked sensors.
5. **Simultaneous changes:** multiple-fault and process/sensor-fault diagnosis
   studies when combined faults remain isolable.

The canonical sources already listed in the literature ledger—Chen and Patton,
Blanke et al., Ding, Fawzi et al., Pajic et al., and Pasqualetti et al.—must be
checked theorem by theorem. The present cloud environment cannot retrieve those
full texts, so this document does not claim an exhaustive literature result.

## Possible remaining theoretical increment

A defensible increment would need to involve an interaction that the closest
fault-isolation result does not cover. Candidates are:

1. event-law separation under an observation-adaptive strategic policy rather
   than a fixed or arbitrary open-loop fault signal;
2. joint probabilistic attribution of situation-only, mechanism-only and
   simultaneous changes after nuisance-variable marginalization;
3. a refusal rule tied to overlap of admissible event-law sets, with a finite-
   window error guarantee under causal attacks;
4. robustness when the continuing mechanism is already contaminated or the
   cross-source constraint is uncertain.

Items 1–3 are not automatically new as a combination. They must be compared
with robust fault isolation, active attack identification, Byzantine sequential
detection and set-membership diagnosis.

## What the current theorem does not yet add

- Its syndrome is a standard structured residual.
- Its \(2k\) condition is standard redundancy/error-correction logic.
- Its unknown boundary is handled by unioning guarded change paths, which is not
  by itself a new change-detection result.
- Its BSC transfer uses standard channel invertibility and minimum-distance
  decoding.
- Its reject option has not yet been connected to an anytime-valid or minimax
  sequential guarantee.

The four-class notation provides a clear problem organization, but notation and
application context are not enough for an independent theory contribution.

## Falsification checklist for the full-text audit

For every closest fault-isolation or secure-estimation theorem, record:

1. whether process/mode changes and sensor faults can occur simultaneously;
2. whether their onset times are unknown and asynchronous;
3. whether the affected sensor set is unknown and time-varying;
4. whether the fault/attacker adapts to past outputs;
5. whether conditions are deterministic rank/isolation conditions or
   probabilistic finite-window bounds;
6. whether ambiguous hypotheses can be declared unisolable or rejected;
7. whether the guarantee assumes a known exact nominal model;
8. whether arbitrary mixtures over nuisance states and paths are covered.

If one established method answers all eight under equal or weaker assumptions,
the current Route B theory should be treated as a domain reformulation and the
independent-paper route should stop or change substantially.

## Current paper-level decision

The theory package is internally coherent after the adaptive-attack correction,
but its independent contribution is **INSUFFICIENT / UNRESOLVED**. This means:

- do not implement D-MIR;
- do not describe the syndrome theorem as a paper contribution;
- complete the full-text fault-isolation comparison before adding more model
  variants;
- continue only if a precise adaptive/probabilistic/refusal theorem remains
  uncovered and can be proved under operational assumptions.

