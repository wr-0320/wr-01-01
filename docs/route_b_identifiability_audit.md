# Route B event-level identifiability audit

Status: **theory audit; supersedes the event-level conclusion of the first
anchor/witness lemma draft**  
Date: 2026-10-20

## Audit conclusion

The first draft correctly showed that an anchor view and a mechanism-witness
view can make the composite state \((z,m)\) identifiable from its emission law.
It then moved too quickly from state-level injectivity to event-level
identifiability. That implication holds when each event has one known successor
state. It does not hold in general when an event marginalizes over several
successor states or paths.

The correct object is the event-conditioned observation-block law, not the
emission law of an individual composite state.

## Minimal counterexample

Let the pre-change state be \((z,m)=(0,0)\), let
\(\mathcal Z=\{0,1,2\}\), \(\mathcal M=\{0,1\}\), and let the first
post-boundary observation be binary.

- Under \(H_Z\), the situation moves to \(z=1\) or \(z=2\) with equal
  probability while \(m=0\). Their emissions are
  \(\operatorname{Bernoulli}(0.2)\) and
  \(\operatorname{Bernoulli}(0.8)\).
- Under \(H_M\), the situation remains \(z=0\), the mechanism moves to
  \(m=1\), and its emission is \(\operatorname{Bernoulli}(0.5)\).

All three successor-state emission distributions are different, but

\[
\tfrac12\operatorname{Bernoulli}(0.2)+
\tfrac12\operatorname{Bernoulli}(0.8)
=\operatorname{Bernoulli}(0.5).
\]

Consequently, \(H_Z\) and \(H_M\) have exactly the same one-step observation
law. If all three successor states return after one step to the same state and
share the same later transition law, their observation-block distributions are
equal for every \(L\ge1\). Distinct state emissions therefore do not prevent
event mixtures from colliding.

This example uses a transient change. A minimum dwell-time constraint excludes
this particular construction, but it does not by itself prove event
identifiability. The induced length-\(L\) path mixtures still have to be checked.

## Correct event-level criterion

For event class \(c\in\{0,Z,M,ZM\}\), let
\(\mathcal R_c^{(L)}\) be the reachable latent paths over an observation window
of length \(L\). A path \(\pi\) induces an observation-block measure
\(Q_\pi^{(L)}\). For fixed transition parameters,

\[
P_c^{(L)}=\sum_{\pi\in\mathcal R_c^{(L)}}
w_{c,\pi}Q_\pi^{(L)}.
\]

For fixed known weights, two events are identifiable over the window exactly
when their resulting measures differ:

\[
P_c^{(L)}\ne P_{c'}^{(L)}.
\]

When transition weights are uncertain within admissible sets, define the set of
possible event laws \(\mathcal P_c^{(L)}\). Uniform event identifiability is
equivalent to

\[
\mathcal P_c^{(L)}\cap\mathcal P_{c'}^{(L)}=\varnothing.
\]

If any distribution over the reachable paths is admissible, this becomes a
convex-hull condition:

\[
\operatorname{conv}\{Q_\pi^{(L)}:\pi\in\mathcal R_c^{(L)}\}
\cap
\operatorname{conv}\{Q_\pi^{(L)}:\pi\in\mathcal R_{c'}^{(L)}\}
=\varnothing.
\]

This criterion is exact but can be difficult to verify. A stronger, easier
sufficient condition is that all reachable block measures across the event
classes are linearly independent and that the event path sets are disjoint.
Equality of two event mixtures would then give a nontrivial linear dependence,
which is impossible.

Distinct block measures are not enough; linear or affine independence is the
relevant property when unknown mixtures are allowed.

## Quantitative finite-window evidence

Define the robust event separation

\[
\delta_L = \min_{c\ne c'}
\inf_{P\in\mathcal P_c^{(L)},Q\in\mathcal P_{c'}^{(L)}}
\operatorname{TV}(P,Q).
\]

- \(\delta_L=0\) means uniform attribution is impossible at window \(L\).
- \(\delta_L>0\) supplies a finite-window separation margin.
- With equal event priors, a binary comparison has minimax error at least
  \((1-\delta_L)/2\) for the least-separated pair.

Increasing \(L\) may separate persistent transition models even when their
one-step mixtures coincide. It need not do so: observationally equivalent path
mixtures can remain equal for all windows. A useful theorem must derive a
positive, computable lower bound on \(\delta_L\) from structural assumptions;
merely assuming \(\delta_L>0\) would restate the desired conclusion.

The source project already contains the balanced-prior identity
\(R^*=(1-\operatorname{TV})/2\) for aliased visible histories in
`/workspace/wr-01/outputs/stage15_robustness/theoretical_appendix.tex`.
Route B therefore cannot claim the total-variation testing bound as a new
result. Its possible theoretical increment must be a structural, event-level
lower bound on \(\delta_L\), especially under unknown affected sources and
mixed successor paths.

## What the anchor/witness lemma still proves

For emissions

\[
p(a,b\mid z,m)=a_z(a)b_{z,m}(b),
\]

injectivity of \(a_z\) in \(z\), invariance of \(a_z\) to \(m\), and
injectivity of \(b_{z,m}\) in \(m\) for fixed \(z\) imply injectivity of the
composite-state emission map. This is enough for four-event attribution only
under an additional deterministic-successor condition, or under an event-level
mixture-separation condition such as the convex-hull criterion above.

The lemma remains useful as a state-identification baseline. It is not an
event-identification theorem in the general stochastic transition model.

## Sparse attacks without a permanently trusted source

Consider a deterministic clean multi-source signature
\(s(z)\in\mathcal A^S\). A mechanism may arbitrarily alter at most \(k\)
source coordinates, and the affected source set is unknown.

The situation remains uniquely recoverable under every such attack exactly
when the clean signature code has minimum Hamming distance

\[
d_{\min}=\min_{z\ne z'}d_H(s(z),s(z'))>2k.
\]

This is the usual error-correction argument: radius-\(k\) corruption balls are
disjoint exactly under the \(2k\) separation. It replaces a permanently trusted
source with cross-source redundancy, but it is closely related to sparse secure
estimation and is unlikely to be novel by itself.

Two limitations matter for Route B:

1. From a known pre-change situation, a clean situation change can be separated
   from a mechanism-only \(k\)-sparse corruption when the new clean signature
   lies outside the radius-\(k\) ball of the old signature. Recovering the
   situation under simultaneous corruption needs the stronger \(2k\) condition.
2. Sparse corruption alone cannot identify that the mechanism changed. If an
   adversary may leave reports unchanged, a mechanism transition can have the
   same observation law as no mechanism transition. Detecting \(m\) requires an
   activation, divergence, duration or intervention assumption in addition to
   sparsity.

The stochastic analogue must replace Hamming separation with a source-level
testing or block-divergence condition and account for noise. Importing the
\(2k\)-sparse observability result without this event-attribution step would not
create a new contribution.

## Candidate weaker conditions

| Condition | What it can plausibly establish | Main limitation | Priority |
|---|---|---|---|
| Unknown \(k\)-sparse attack plus redundant source signatures | State recovery and separation of some state-only versus mechanism-only changes without a trusted source | Mechanism change is invisible if the attack can be a no-op; close to secure estimation | High |
| Cross-source algebraic or physical constraints | A residual or syndrome that reveals reports inconsistent with any clean situation | Detects inconsistency, not necessarily which mechanism caused it | High |
| Finite dwell time plus positive divergence rate | One-step ambiguity can resolve over a longer block; supports a delay/evidence relation | A positive divergence assumption can become tautological unless derived from model structure | Highest |
| Intermittent verified channel | Replaces a permanent anchor with occasional ground truth and may reset ambiguity | Verification timing and cost become part of the problem | Medium |
| Active verification action | Can choose an observation kernel that separates otherwise identical hypotheses | Moves the work toward controlled sensing/POMDPs and has substantial prior art | Defer unless passive conditions fail |

The most promising passive theory target is the combination of unknown sparse
manipulation, cross-source redundancy and a finite dwell-time condition that
produces a nonzero lower bound on \(\delta_L\). It must also impose a minimum
observable mechanism effect; otherwise mechanism-change detection remains
impossible.

A first binary finite-alphabet candidate theorem is now given in
`route_b_finite_alphabet_theorem.md`. It derives event-center distance
\(D=\min\{r,L(d-2k)\}\) and a uniform positive TV bound through an independent
binary symmetric channel. The result is a feasibility theorem under strong
conditions, not a novelty claim: it conditions on a candidate boundary and
identifiable prehistory, uses persistent situations and assumes separated old
and new mechanism-action classes.

## Revised theory target

The desired chain is now explicit:

\[
\text{structural redundancy + persistence + minimum mechanism effect}
\Longrightarrow
\delta_L>0
\Longrightarrow
\text{finite-window attribution/error bound}
\Longrightarrow
\text{statistically defined abstention}.
\]

The theory stage is a GO only if the first implication can be proved under
assumptions that are weaker and more operationally credible than a permanently
trusted source. The remaining implications can then use standard testing and
selective-decision tools, with their provenance stated explicitly.

