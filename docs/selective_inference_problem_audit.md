# Deception-risk-aware selective inference problem audit

Status: **mathematically constrained; originality INCONCLUSIVE-BLOCKED; no implementation**

Date: 2026-10-20

## 1. Minimum problem

Let `Z` be a hidden situation and `O` the reports and masks available at one
decision point. A policy chooses a class prediction, trusted verification at
cost `c_v`, or abstention at cost `c_a`. Prediction errors cost `c_e`.

A predictor `h` and selector `s` have target coverage

\[
C_Q=Q(s(O)=1)
\]

and selective risk

\[
R_Q=Q(h(O)\ne Z\mid s(O)=1).
\]

The desired contribution cannot be “use a confidence threshold.” It must name
an uncertainty class `U`, available calibration information and a guarantee
such as `sup(Q in U) R_Q <= alpha` at nonzero coverage or bounded total cost.

## 2. Unrestricted deception is impossible

### Proposition 1

If the adversary may choose an arbitrary target joint law of `(Z,O)`, any
deterministic system that accepts at least one observation can be forced to have
selective risk one.

**Proof.** Choose an accepted observation `o`. Concentrate the target law on
`O=o` and assign `Z` a class different from `h(o)`. The system accepts every
draw and is wrong on every draw. Randomized prediction has an analogous lower
bound obtained by choosing the least-favored class. Thus a universal risk level
below one requires zero coverage, trusted verification, or a restricted threat
class.

Unknown “strategic deception” is therefore not itself a usable robustness
assumption.

## 3. Even bounded contamination limits selective guarantees

Let the clean law be `P`, let the accepted set have clean mass `c>0`, and let
the unconditional clean accepted-error mass be `e<=c`. Under Huber
contamination `Q=(1-epsilon)P+epsilon H`, an unrestricted contaminating law can
place all its mass on an accepted, misclassified pair. The exact worst-case
selective risk is

\[
\boxed{
R_Q^{\max}=
\frac{(1-\varepsilon)e+\varepsilon}
{(1-\varepsilon)c+\varepsilon}
}.
\]

This bound is attainable under the declared contamination class. For a perfect
clean classifier (`e=0`), `epsilon=0.1` and clean coverage `c=0.5`, the risk
floor is `2/11`, approximately 18.2%. Reducing coverage to `c=0.25` raises the
floor to `4/13`, approximately 30.8%, because the attacker concentrates on the
remaining accepted region.

As `epsilon` approaches one, every positive-clean-coverage guarantee approaches
risk one. An unknown, unbounded contamination rate therefore cannot support a
nontrivial selective-risk claim. This is a screening baseline, not a novelty
claim; robust statistics and distributionally robust prediction are direct
competitors.

## 4. Collision with existing method families

| Candidate component | Existing family likely to cover it | What would have to be different |
|---|---|---|
| Predict or abstain at a target coverage | Selective classification and reject options | A threat-specific guarantee unavailable to standard risk-coverage methods |
| Calibrate error or set coverage | Risk-controlling and conformal prediction | Validity under the declared strategic/dependent attack without assuming the answer through known weights |
| Reject unusual or attacked inputs | Adversarial detection and OOD uncertainty | A guarantee that survives adaptive, semantically valid intelligence manipulation |
| Request a trusted observation | Active sequential hypothesis testing | Unknown or partially identified observation models with a new reliability result |
| Defer to an analyst or trusted source | Learning to defer | A structured deception model and guarantee beyond standard expert deferral |
| Minimize prediction, verification and abstention cost | Bayes reject rules, value of information and POMDPs | A result not obtained by inserting a new simulator into an established decision model |

## 5. Smallest candidate gap

The narrowest potentially distinct question is:

> Can observable multi-source conflict, missingness and provenance information
> define a verifiable structured threat set whose worst-case selective risk can
> be controlled, while trusted verification is allocated only when it provably
> improves the risk--coverage--cost frontier?

Three facts must hold before this becomes a research contribution:

1. The threat set must be identifiable or externally auditable from information
   available before the decision. JMID-PI already showed that unconstrained
   MNAR/dependence uncertainty cannot provide this by itself.
2. The risk result must differ from applying robust conformal or
   distributionally robust selective prediction to a fixed uncertainty set.
3. The verification rule must add more than standard active testing, value of
   information or learning-to-defer under renamed actions.

At present none of these differences is established by full-text evidence.

## 6. Current decision

- **Feasibility:** unrestricted deception is a formal NO-GO; a bounded and
  auditable threat class is mandatory.
- **Novelty:** `INCONCLUSIVE-BLOCKED` until the 15 frozen primary works are read
  and their exact assumptions and guarantees are entered in the matrix.
- **Implementation:** NO-GO. No selector, conformal wrapper, verification policy
  or simulator should be written yet.
- **Next allowed action:** after the published network configuration is loaded,
  complete the frozen full-text matrix and make one GO/STOP decision. Do not
  enlarge the literature set or alter the mathematical target after seeing
  favorable papers.

The candidate direction remains worth auditing because it asks an operationally
important reliability question. It has not yet passed the originality gate.
