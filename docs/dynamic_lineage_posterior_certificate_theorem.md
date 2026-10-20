# Posterior certification under unknown evidence lineage

Status: **mathematically feasible, conditionally nonvacuous, originality unconfirmed, prototype NO-GO**

Date: 2026-10-20

## 1. Observation model

At a fixed decision horizon, let `Z` be a binary hidden situation, `X` the
observed report contents, and `A` the permitted lineage evidence: timestamps,
delays, partial provenance markers or trusted audit outcomes. Let
`g in G` index a finite admissible lineage path up to the horizon. A path may
contain a different dependency graph at each time, subject to prespecified
transition or dwell-time constraints.

For every `g`, the model supplies:

- a density `r_g(a)` for the lineage evidence;
- a joint model for `(Z,X)` and hence the posterior
  `pi_g(x,a) = P_g(Z=1 | X=x,A=a)`.

The model is well specified only if the true path `g_star` belongs to `G` and
the corresponding observable and situation likelihoods are correct. Parameter
uncertainty must be included in the index `g`; silently plugging in estimated
parameters is outside the theorem.

## 2. An exact finite-sample lineage confidence set

Assume `G` has at least two members and the observable laws admit the required
Radon--Nikodym derivatives. For each candidate `g`, choose fixed positive
weights `w(h|g)` over `h != g` that sum to one and define

\[
E_g(A)=\sum_{h\ne g}w(h\mid g)\frac{r_h(A)}{r_g(A)}.
\]

Under `P_g`, `E_g` is nonnegative and has expectation one. Define

\[
\mathcal C_\delta(A)=\{g:E_g(A)<1/\delta\}.
\]

### Lemma 1: finite-sample lineage coverage

For every `g_star in G`,

\[
P_{g_\star}\{g_\star\in\mathcal C_\delta(A)\}\ge 1-\delta.
\]

**Proof.** Under `P_{g_star}`, the likelihood-ratio identity gives
`E[E_{g_star}]=1`. Markov's inequality gives
`P(E_{g_star} >= 1/delta) <= delta`. No independence between graph selection
and later reporting is asserted or needed beyond the declared joint model.

This construction shows how the same lineage data can be used without an
informal post-selection argument. It does not make an incorrect likelihood
model valid, and it does not identify observationally equivalent graphs.

## 3. Primary posterior certificate

For observed `(x,a)`, define

\[
L_\delta(x,a)=\min_{g\in\mathcal C_\delta(a)}\pi_g(x,a),\qquad
U_\delta(x,a)=\max_{g\in\mathcal C_\delta(a)}\pi_g(x,a).
\]

Report the midpoint

\[
\widehat\pi_\delta(x,a)=\{L_\delta(x,a)+U_\delta(x,a)\}/2
\]

together with radius `R_delta=(U_delta-L_delta)/2`.

If `C_delta(a)` is empty, the procedure must emit model failure and the vacuous
interval `[0,1]`; it may not silently select a graph.

### Theorem 1: finite-sample posterior-error certificate

For every true lineage path `g_star in G`, with probability at least
`1-delta` over the lineage evidence,

\[
\pi_{g_\star}(X,A)\in[L_\delta(X,A),U_\delta(X,A)]
\]

and therefore

\[
|\widehat\pi_\delta(X,A)-\pi_{g_\star}(X,A)|\le R_\delta(X,A).
\]

**Proof.** On the event `g_star in C_delta(A)`, the true-model posterior is one
of the values over which the minimum and maximum are taken. Lemma 1 controls
the complement of that event.

The target is the posterior under the correctly specified member of the model
class. This is not a distribution-free guarantee against arbitrary deception.
The probability statement is marginal over the lineage evidence used to form
the confidence set; it is not a conditional-coverage claim for every fixed
realization of that evidence.

## 4. When the certificate is nonvacuous

Write the model-specific posterior log odds as

\[
\lambda_g(x,a)=\operatorname{logit}\{\pi_g(x,a)\}.
\]

This section assumes retained posteriors lie strictly between zero and one; a
deterministic zero or one posterior makes the log-odds sensitivity unbounded
unless all retained models agree on that boundary value.

Let `d(g,h)` be a lineage discrepancy visible to the model, such as the number
of uncertain evidence-equivalence groups. Assume the calibrated report model
satisfies the sensitivity condition

\[
|\lambda_g(x,a)-\lambda_h(x,a)|\le B(x,a)d(g,h)
\tag{S}
\]

for every retained pair. Condition (S) is auditable only when the per-group
Bayes factors are bounded or externally calibrated.

### Theorem 2: graph-radius to posterior-width propagation

If the diameter of `C_delta(a)` under `d` is at most `rho_delta(a)`, then

\[
U_\delta-L_\delta
\le
\tanh\left(\frac{B\rho_\delta}{4}\right),
\]

and

\[
|\widehat\pi_\delta-\pi_{g_\star}|
\le
\frac12\tanh\left(\frac{B\rho_\delta}{4}\right)
\]

on the coverage event.

**Proof.** Condition (S) bounds the retained log-odds range by
`B rho_delta`. Among two logistic probabilities whose logits differ by at most
`D`, their largest possible difference is `tanh(D/4)`. The midpoint radius is
half the interval width.

This result prevents a merely formal certificate: it is useful only when both
the lineage-set diameter and the posterior sensitivity remain controlled. If
either grows with the number of sources, the bound approaches its worst-case
value and the route repeats the failure mode observed in JMID-PI.

For scale, `B rho_delta <= 1` gives posterior radius at most
`tanh(1/4)/2`, approximately `0.1225`, rather than the vacuous radius `0.5`.
This numerical statement does not establish that a particular intelligence
system satisfies the sensitivity or graph-radius premise.

## 5. Observable conditions that shrink the lineage set

Suppose `A_1,...,A_n` are conditionally independent lineage-audit observations
on a stable segment. For every wrong `h`, assume

\[
D_{KL}(r_{g_\star}\|r_h)\ge \kappa>0
\]

and the one-step log likelihood ratio
`log(r_{g_star}(A_i)/r_h(A_i))` differs from its mean by at most `b` in absolute
value. Because the mixture defining `E_h` contains the true candidate with
weight `w(g_star|h)`, Hoeffding's inequality and a union bound imply that, with
probability at least `1-beta`, every such `h` is excluded whenever

\[
n\kappa-b\sqrt{2n\log\{(|\mathcal G|-1)/\beta\}}
+\min_{h\ne g_\star}\log w(g_\star\mid h)
\ge \log(1/\delta).
\tag{I}
\]

Condition (I) is a sufficient nonvacuity condition. It connects an observable
lineage signal to shrinkage of the posterior certificate. Positive separation
may come from timestamp order, transmission-delay laws, provenance hashes or
trusted audits. It cannot be asserted from matching report contents alone.

For dynamic graphs, the same argument applies within segments only if the
admissible path class and minimum dwell time are fixed in advance. Otherwise
the path count and its mixture weights can make (I) useless.

## 6. Failure and impossibility examples

### 6.1 Direct copy and common upstream can be observationally identical

Let all observed reports equal a latent binary report `R`. One model makes the
later reports direct copies of the first report; another makes all reports
children of an unobserved common upstream source. Without timestamps,
provenance or intervention, both generate exactly the same report vector.
Their lineage structures cannot be distinguished from contents alone.

If the unknown mechanisms also allow different values of
`P(R=Z)`, they can preserve the same marginal distribution of reports while
implying different `P(Z|X)`. The posterior is then unidentified without labels,
audits or externally calibrated reliability.

### 6.2 Point-graph fusion can become arbitrarily overconfident

Let `P(R=Z)=p>1/2` and observe `m` exact copies of `R=1`. The correct posterior
is `p`. Treating the copies as independent gives

\[
\widetilde\pi_m=
\frac{p^m}{p^m+(1-p)^m}\longrightarrow 1.
\]

The absolute overconfidence approaches `1-p`. A lineage point estimate that
incorrectly deletes the all-copy explanation provides no finite-sample
protection against this failure.

### 6.3 A complete graph set is valid but useless

If the retained set contains both the independent-source and all-copy models,
its posterior interval contains both `p` and `tilde pi_m`. As `m` grows, the
interval remains wide. Coverage alone therefore does not establish practical
value; condition (I) or another observable separation condition is essential.

## 7. Relationship to the six frozen competitors

Based on the externally supplied audit:

- dynamic truth discovery estimates dependence states but does not propagate a
  valid graph confidence set into a posterior-error radius;
- copy-aware truth discovery estimates static copying but its confidence score
  is not a posterior-error guarantee;
- known-topology data-incest removal corresponds to a singleton confidence set
  and does not control graph misspecification;
- MuSCI and online PAC-Bayes methods certify predictive objects without
  identifying evidence-copy equivalence classes;
- WATCH supplies an anytime alarm but does not provide this posterior envelope.

These distinctions remain provisional until the original papers are locally
verified. More importantly, Theorems 1 and 2 combine standard likelihood-ratio
confidence-set inversion with a robust posterior envelope and logistic
sensitivity. That composition is mathematically correct but is not yet a
demonstrated top-tier theoretical increment.

## 8. Decision

The main mathematical target is feasible:

- **finite-sample validity:** established under a correctly specified finite
  lineage class;
- **unknown structure:** explicitly covered through a set rather than a point
  graph;
- **nonvacuity:** obtained under positive observable lineage separation and a
  bounded graph-to-log-odds sensitivity;
- **failure boundary:** explicit observational-equivalence and overconfidence
  counterexamples are available.

The originality gate is not passed. The current certificate is a disciplined
composition of existing statistical tools. A final targeted audit asked whether
posterior-relevant graph equivalence classes could provide a tighter interval
without full graph recovery; `dynamic_lineage_target_equivalence_result.md`
shows that quotienting alone preserves the exact posterior image. Strict
tightening requires additional identifying information or a separately valid,
more efficient functional confidence procedure.

Accordingly, the outcome is **CONDITIONAL THEORY-FEASIBLE / PROTOTYPE NO-GO**.
Anytime alarms remain deferred. No model implementation or simulation is
authorized by this result.
