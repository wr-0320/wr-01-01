# Route B mathematical problem formulation

Status: **draft for theory review; no algorithm implemented**

## Variables and information boundary

At time \(t\), the defender receives reports
\(x_t=(x_{1,t},\ldots,x_{S,t})\) from \(S\) sources, with explicit missingness
and timestamps. The latent variables are:

- \(z_t\in\mathcal Z\): the operational situation of interest;
- \(m_t\in\mathcal M\): the report-generation mechanism or deception regime;
- \(r_t\): the run length of the current mechanism regime;
- \(c_t\in\{0,Z,M,ZM\}\): no change, situation-only change,
  mechanism-only change, or simultaneous change.

A minimal generative factorization is

\[
p(z_{1:T},m_{1:T},x_{1:T}) = p(z_1)p(m_1)
\prod_{t=2}^{T}p(z_t\mid z_{t-1})p(m_t\mid m_{t-1},r_{t-1})
\prod_{t=1}^{T}\prod_{s=1}^{S}g_s(x_{s,t}\mid z_t,m_t,q_{s,t}),
\]

where \(q_{s,t}\) contains observable context such as region, delay and sensing
mode. Cross-source dependence must be represented explicitly when a shared
deception plan creates it; the product emission is only the minimal starting
case.

The online outputs are

\[
p(z_t,m_t,r_t,c_t\mid x_{1:t}),
\]

the posterior probability of each change explanation, and a decision in
\(\mathcal Z\cup\{\text{insufficient evidence}\}\). Calling \(c_t\) a causal
explanation requires structural or intervention assumptions. Without them it is
only attribution within the specified model.

## Four competing event hypotheses

For a candidate boundary at \(t\), define:

- \(H_0\): neither latent variable changes;
- \(H_Z\): \(z\) changes and \(m\) remains fixed;
- \(H_M\): \(m\) changes and \(z\) remains fixed;
- \(H_{ZM}\): both change.

For a future block of length \(L\), let \(P_h^{(L)}\) be the distribution of
\(X_{t:t+L-1}\) induced by hypothesis \(h\), conditional on the pre-change
history and available source pattern.

Define block separation

\[
\Delta_L = \min_{h\ne h'}
\operatorname{TV}\!\left(P_h^{(L)},P_{h'}^{(L)}\right).
\]

KL, Jensen-Shannon or Chernoff information can supplement total variation for
computation, but they must not be interpreted as equivalent diagnostics without
the relevant bounds.

## A hard impossibility boundary

If \(P_{H_Z}^{(L)}=P_{H_M}^{(L)}\), the two changes are observationally
indistinguishable over that block. With equal prior probabilities, the minimum
possible binary attribution error is

\[
R_L^*=\frac{1-
\operatorname{TV}(P_{H_Z}^{(L)},P_{H_M}^{(L)})}{2}.
\]

Thus equality implies error \(1/2\); small total variation implies that any
method needs more evidence or must accept high attribution error. This is a
standard binary-testing consequence, not a new theorem. Its role is to prevent
the project from promising universal deception recognition.

A simple counterexample is

\[
g_{m_0}(\cdot\mid z_1)=g_{m_1}(\cdot\mid z_0).
\]

Then the one-step observations produced by a situation change
\((z_0,m_0)\to(z_1,m_0)\) equal those produced by a mechanism change
\((z_0,m_0)\to(z_0,m_1)\). If their transition-induced future block laws also
match, longer passive observation cannot resolve the ambiguity.

## Why a two-factor model alone is insufficient

The pair \(u_t=(z_t,m_t)\) can always be treated as one composite HMM state.
Accurate filtering of \(u_t\) does not identify which coordinate has operational
meaning. Permuting labels or absorbing a mechanism state into a situation state
can leave the observation law unchanged. A publishable method therefore needs
structural assumptions that make the factorization testable, not only a
factorial-HMM implementation.

Candidate assumptions to investigate are:

1. **Anchor view:** at least one report channel changes with \(z\) but is
   invariant to the admissible mechanism changes.
2. **Mechanism witness:** at least one observable feature or cross-source
   relation responds to \(m\) after conditioning on \(z\).
3. **Restricted attack surface:** a mechanism may alter only a bounded subset of
   sources, features or time intervals.
4. **Distinct dynamics:** the transition or duration laws of \(z\) and \(m\)
   belong to separately constrained families.
5. **Active verification:** a defender action can obtain a channel whose
   response differs under \(H_Z\) and \(H_M\).

These are candidate sufficient structures, not proved sufficient conditions in
their present form. The theory phase must determine a minimal set and prove the
corresponding identification result, preferably first for a finite-state,
finite-alphabet model.

## A restricted state-identification lemma

The following elementary result shows that the composite state can be
identifiable under an explicit information boundary. It is a design sanity
check, not an event-identification theorem and not a novel theoretical
contribution.

Assume two conditionally independent observed views \(A_t,B_t\). Their emission
law is

\[
p(a,b\mid z,m)=a_z(a)b_{z,m}(b).
\]

Suppose:

1. the anchor family is injective in situation:
   \(a_z\ne a_{z'}\) for every \(z\ne z'\), and is invariant to \(m\);
2. for each fixed \(z\), the witness family is injective in mechanism:
   \(b_{z,m}\ne b_{z,m'}\) for every \(m\ne m'\).

Then the mapping \((z,m)\mapsto p(A,B\mid z,m)\) is injective. If two pairs
have different \(z\), their \(A\) marginals differ. If they have the same
\(z\) but different \(m\), their conditional \(B\) laws differ. Therefore,
when each event has one known successor pair, the four events have distinct
distributions.

This is distributional identifiability, not perfect classification from one
sample. Overlapping distributions still impose nonzero finite-sample error.
More importantly, the conclusion does not extend automatically to stochastic
events that mix over several successor states or paths: different mixtures of
distinct emissions can be equal. The counterexample and corrected event-level
convex-hull criterion are in `route_b_identifiability_audit.md`.

The anchor assumption may also be too strong for strategic intelligence. A
worthwhile theorem must weaken it, for example to cross-source redundancy,
sparse attack surfaces, an intermittent verified channel or active verification,
and must state the required observation length.

## Confusability and abstention under model uncertainty

For fitted models, let \(\mathcal U_h(\epsilon)\) be a justified ambiguity set
around each event law \(P_h^{(L)}\), for example a total-variation ball whose
radius is estimated from held-out predictive error. Define robust separation

\[
\underline\Delta_L(\epsilon)=
\min_{h\ne h'}\inf_{P\in\mathcal U_h(\epsilon),
Q\in\mathcal U_{h'}(\epsilon)}\operatorname{TV}(P,Q).
\]

An operational `insufficient evidence` decision occurs when the minimum
worst-case attribution risk exceeds a frozen tolerance or when
\(\underline\Delta_L(\epsilon)\) falls below a frozen separation threshold.
This definition covers both intrinsic overlap and estimated model error. The
choice and coverage of \(\mathcal U_h\) require proof or empirical calibration;
a raw posterior maximum is not a model-mismatch guarantee.

## Candidate algorithmic object

The minimal D-MIR hypothesis is a finite-state online filter with four event
branches. Each update would:

1. propagate situation and mechanism transition hypotheses;
2. update their joint posterior with all available reports;
3. compute posterior odds for \(H_0,H_Z,H_M,H_{ZM}\);
4. estimate block-level confusability among the supported hypotheses;
5. update the situation posterior normally when attribution is supported, or
   emit `insufficient evidence` when expected risk or confusability exceeds a
   frozen threshold.

The method must expose both uncertainty about \(z_t\) and uncertainty about the
change explanation. Situation entropy alone is inadequate: a posterior may be
confident yet rely on the wrong mechanism model.

## Potential theoretical contribution

The project should continue only if it can add at least one result beyond
standard BOCPD or factorial-HMM filtering:

- a finite-window condition under which situation-only and mechanism-only
  changes are identifiable in a multi-source model;
- a lower bound showing when attribution is impossible or how evidence length
  scales with separation;
- an online rule coupling attribution and abstention, with a bound connecting
  false attribution, detection delay and block separation;
- a strategic-attack constraint that creates a demonstrably different problem
  from ordinary regime switching or sensor fault isolation.

If the only implementable method is a standard switching model with renamed
states, the theoretical route is NO-GO.

The active theorem target and counterexamples are maintained in
`route_b_identifiability_audit.md`. That audit supersedes any broader reading of
the anchor/witness lemma above.

The first complete finite-alphabet candidate theorem is in
`route_b_finite_alphabet_theorem.md`. It is deliberately conditional on a known
candidate boundary and identifiable prehistory and therefore does not yet solve
the full online problem.

`route_b_joint_window_theorem.md` removes those two conditions in a more
restricted cross-source model: the unknown boundary and pre-state are unioned
inside each event class, while a known parity constraint provides a noisy,
report-computable mechanism witness. This remains a fault-isolation-like
feasibility result rather than an established novel contribution.

`route_b_adaptive_attack_audit.md` distinguishes open-loop attack-path mixtures
from attacks adapted to past noisy observations. The latter invalidate the
fixed-mixture inverse-channel proof but preserve a pathwise decoder bound under
exogenous additive noise and almost-sure event constraints.

## Decisive experimental cases

The future simulator protocol must manipulate exactly one factor at a time
before considering mixtures:

| Case | Fixed factor | Changed factor | Primary failure being measured |
|---|---|---|---|
| Situation change | mechanism | situation | false deception alarm and delayed true transition |
| Mechanism change | situation | mechanism | false situation flip and missed deception change |
| Simultaneous change | neither | both | incorrect attribution and unjustified confidence |
| Constructed ambiguity | observational block law | semantic explanation | failure to abstain when distinction is impossible |

Every case needs matched trajectories across methods and a controlled
separation axis from indistinguishable to easy. Accuracy, Brier and NLL remain
useful, but the primary new metrics are event-attribution error, detection delay
at a fixed false-alarm rate, false situation-flip rate, selective risk versus
coverage, and calibration of the `insufficient evidence` decision.

