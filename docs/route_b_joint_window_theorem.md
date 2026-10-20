# Route B joint-window syndrome theorem

Status: **candidate theorem for unknown boundary and unknown pre-state; not a
novelty claim**  
Date: 2026-10-20

## Question

Can four change events remain distinguishable after the change time, previous
situation, successor situation and sparse manipulation path are all treated as
unknown nuisance variables?

The first finite-alphabet theorem conditioned on a candidate boundary and an
identified pre-state. This note removes both conditions in a minimal binary
model. It replaces the latent old-versus-new attack-path distance with an
observable cross-source consistency constraint.

## Cross-source observation model

Let the clean multi-source signatures form a known binary code

\[
\mathcal C=\{c_z:z\in\mathcal Z\}\subset\{0,1\}^S
\]

with minimum distance

\[
d=\min_{z\ne z'}d_H(c_z,c_{z'}).
\]

Assume first that \(\mathcal C\) is a linear code with parity-check matrix
\(H\), so every clean report satisfies

\[
Hc_z=0.
\]

For a report before ordinary measurement noise, \(y_t\), the ideal syndrome is

\[
s_t=Hy_t.
\]

If the actually observed report is \(x_t=y_t\oplus e_t\), the computable
syndrome is \(Hx_t=Hy_t\oplus He_t\). It is therefore a noisy observation of
the ideal syndrome, not an oracle mechanism label. The proof only needs
membership in a separated finite code. Linearity supplies a transparent
cross-source statistic and can later be replaced by distance to a nonlinear
constraint set.

## Joint window and event classes

Observe a window of \(T\) time steps. A non-null event has one unknown boundary

\[
\tau\in\{q,q+1,\ldots,T-q\},
\]

so at least \(q\ge1\) samples occur on both sides. The pre-boundary situation
\(z_0\) is unknown. Under a situation change, an unknown
\(z_1\ne z_0\) persists from \(\tau+1\) through the end of the window.

The continuing mechanism produces clean codewords before ordinary measurement
noise. A changed mechanism applies a mask \(a_t\) after \(\tau\), with

\[
\lVert a_t\rVert_0\le k.
\]

It is **observably active** at no fewer than
\(1\le\ell\le q\) post-boundary times:

\[
Ha_t\ne0
\quad\text{for at least \(\ell\) values of }t>\tau.
\]

This condition is stated through the constraint violation induced in reports.
It neither specifies which sources are altered nor uses a mechanism label.
Ordinary measurement noise can obscure the violation, and is handled later at
the distribution level. Under
\(d>2k\), every nonzero mask of weight at most \(k\) automatically has nonzero
syndrome: otherwise it would be a nonzero codeword lighter than \(d\).

Before measurement noise,

\[
y_t=c_{z_t}\oplus a_t,
\]

where \(a_t=0\) before the mechanism change and throughout events with no
mechanism change.

The four event classes are:

- \(H_0\): \(z_t\) is constant and the mechanism remains clean;
- \(H_Z\): \(z_0\to z_1\ne z_0\) at unknown \(\tau\), mechanism clean;
- \(H_M\): situation constant, mechanism becomes observably active at unknown
  \(\tau\);
- \(H_{ZM}\): situation and mechanism change at the same unknown \(\tau\).

Let \(\mathcal J_h^{(T)}\) contain every noiseless report block allowed by event
\(h\), over all admissible \(z_0,z_1,\tau\) and attack paths. This construction
places the unknown boundary, pre-state, successor state and strategy inside the
event set instead of assuming them known.

## Why the joint window is necessary

If only post-boundary reports are retained and \(z_0\) is unrestricted, a
constant trajectory in \(z_1\) under \(H_0\) is identical to the post-change
part of \(z_0\to z_1\) under \(H_Z\). Therefore

\[
\mathcal P_0\cap\mathcal P_Z\ne\varnothing.
\]

Unknown pre-state is not itself fatal. It becomes fatal when the observation
window contains no guaranteed evidence from both sides of the boundary. The
guard length \(q\) is the structural condition that preserves that evidence.

The smallest example uses one source, two signatures \(c_0=0,c_1=1\), no
attack and a one-sample window. Under \(H_0\), choose the unknown constant state
\(z=1\). Under \(H_Z\), change from \(0\) to \(1\) immediately before the
sample. Both event laws are the point mass at observation 1. No estimator can
recover the event label from that window.

## Lemma 1: observable mechanism witness

Assume \(d>2k\). If \(0<\lVert a_t\rVert_0\le k\), then

\[
H(c_z\oplus a_t)=Ha_t\ne0
\]

for every situation \(z\). Hence an attacked report cannot equal any clean
signature, and its Hamming distance from the clean code is at least one.

### Proof

If \(Ha_t=0\), then \(a_t\) is a codeword. Its weight is positive and no more
than \(k<d\), contradicting the code minimum distance. Since XOR by a codeword
does not change the syndrome, \(H(c_z\oplus a_t)=Ha_t\). If the attacked report
equaled another codeword, their XOR would again be a nonzero codeword of weight
at most \(k\). \(\square\)

The syndrome is an observable witness of constraint violation, not a mechanism
identity. Several mechanisms can produce the same nonzero syndrome.

## Lemma 2: joint-window event-center separation

Assume \(d>2k\), \(q\ge1\) and \(1\le\ell\le q\). Every center in one event class is
separated from every center in another event class by at least

\[
\boxed{D_{\mathrm{joint}}=
\min\{\ell,\ q(d-2k)\}>0.}
\]

### Proof

First compare an event with a mechanism change to an event without one while
holding the state-change indicator fixed: \(H_0\) versus \(H_M\), or \(H_Z\)
versus \(H_{ZM}\). The no-mechanism event is a sequence of clean codewords. At
each of at least \(\ell\) active times, Lemma 1 places the mechanism-change
report outside the clean code and at distance at least one from every clean
signature. The block distance is therefore at least \(\ell\), even if the two
events choose different states and boundaries.

Now compare events with different state-change indicators: \(H_0\) versus
\(H_Z\), \(H_M\) versus \(H_{ZM}\), or \(H_Z\) versus \(H_M\). A one-change
state path with at least \(q\) samples in each segment differs from every
constant state path at no fewer than \(q\) time steps. At each such time, the
two clean signatures differ in at least \(d\) coordinates, while the two
reports contain at most \(k\) altered coordinates each. Their distance is at
least \(d-2k\), giving block distance at least \(q(d-2k)\).

The remaining comparison, \(H_0\) versus \(H_{ZM}\), satisfies both types of
difference and hence at least one of the two bounds. Taking their minimum covers
all six event pairs. \(\square\)

The proof does not estimate \(\tau\) or \(z_0\). It shows that marginalizing or
randomizing over all their admissible values cannot make the noiseless event
supports overlap.

## Theorem 1: joint-window distribution separation

The information structures I0 and I1 are defined and audited in
`route_b_adaptive_attack_audit.md`.

Let each of the \(ST\) center bits be subject to exogenous independent binary
noise with \(0\le\eta<1/2\). Under I0, the complete center path is selected
independently of this noise; \(\mathcal P_h^{(T)}\) then contains
\(K_\eta\mu\) for every mixture \(\mu\) over
\(\mathcal J_h^{(T)}\). Under the assumptions of Lemma 2,

\[
\delta_T^{\mathrm{joint}}=
\min_{h\ne h'}
\inf_{P\in\mathcal P_h^{(T)},\,Q\in\mathcal P_{h'}^{(T)}}
\operatorname{TV}(P,Q)>0.
\]

In particular,

\[
\delta_T^{\mathrm{joint}}
\ge
\max\left\{
(1-2\eta)^{ST},
1-2\Pr\!\left[B\ge
\left\lceil D_{\mathrm{joint}}/2\right\rceil\right]
\right\},
\]

where \(B\sim\operatorname{Binomial}(ST,\eta)\).

### Proof

Lemma 2 makes the four noiseless event supports disjoint with pairwise distance
at least \(D_{\mathrm{joint}}\), including after all nuisance variables are
unioned within their event classes. Arbitrary distributions supported on two
different event sets therefore have TV distance one before measurement noise.
The invertible-channel and nearest-event-set arguments from
`route_b_finite_alphabet_theorem.md` apply with \(L\) replaced by \(T\) and
\(D\) by \(D_{\mathrm{joint}}\). \(\square\)

Under I1, an attacker may choose each mask from past noisy observations. The
final center path is then dependent on earlier noise, so the inverse-channel
term \((1-2\eta)^{ST}\) is not established. The pathwise decoder proof still
gives

\[
\delta_{T,\mathrm{I1}}^{\mathrm{joint}}
\ge 1-2\Pr\!\left[B\ge
\left\lceil D_{\mathrm{joint}}/2\right\rceil\right].
\]

This is a positive certificate only when the binomial tail is below \(1/2\).
The proof and filtration assumptions are audited in
`route_b_adaptive_attack_audit.md`.

Thus the theorem handles arbitrary open-loop mixtures over unknown boundaries,
states and sparse attack paths. Its distance-sensitive part additionally
handles causal observation-adaptive paths whose structural constraints hold
almost surely. It does not claim that either bound is large enough for useful
finite-sample inference.

## Assumption-deletion counterexamples

### 1. No two-sided guard window

If \(q=0\), place the situation change before the first retained sample. An
\(H_Z\) block that is constantly \(c_{z_1}\) is identical to an \(H_0\) block
whose unknown state is \(z_1\). Thus unknown boundary and unknown pre-state make
\(H_0\) and \(H_Z\) overlap.

### 2. Constraint-preserving manipulation

If an allowed nonzero attack mask is itself a codeword, then its syndrome is
zero. Adding it maps one clean signature to another. A mechanism change can
therefore be observationally identical to a situation change. This occurs when
the attack budget reaches the relevant codeword weight or when the chosen
cross-source constraint does not cover the manipulated feature.

For example, with repetition code \(\{000,111\}\), the mask \(111\) has zero
syndrome and maps situation 000 to the report 111. A mechanism-only change using
that mask is identical to a clean situation change from 000 to 111.

### 3. No observable activity requirement

If the changed mechanism may produce zero syndrome throughout the window, it
may choose the clean path. Then \(H_0\) overlaps \(H_M\), and \(H_Z\) overlaps
\(H_{ZM}\). Naming a latent mechanism transition does not make it observable.

### 4. Insufficient source redundancy

If \(d\le2k\), two distinct clean signatures can be made equal by two
admissible sparse masks. The state-change and no-state-change event supports can
then overlap even when a two-sided window is available.

### 5. Multiple or asynchronous changes

If arbitrary numbers of state and mechanism changes are allowed in one window,
a change-away-and-back path can share the same retained endpoints as a null
path, while different state and mechanism boundaries invalidate the four
synchronous event definitions. Additional dwell-time and event-count structure
is required; increasing \(T\) alone does not resolve this ambiguity.

### 6. Unknown or misspecified cross-source constraint

If \(H\) is learned from contaminated data and a strategic mechanism can shape
that learning, a zero syndrome no longer certifies consistency with the true
clean process. The theorem treats \(H\) as known structural knowledge. It does
not provide robustness to constraint misspecification.

## What has and has not been achieved

The theorem removes two assumptions from the first finite-alphabet result:

- the change boundary is unknown within a guarded joint window;
- the pre-change situation is unknown and is marginalized with the other
  nuisance variables.

It also replaces separation between latent attack-path classes with a
cross-source syndrome-activation condition whose noisy version is computable
from reports. No source is declared permanently trusted; consistency is
supplied by redundancy across all sources.

The remaining assumptions are substantial:

- the clean cross-source code and parity relation are known;
- the continuing mechanism is constraint-respecting;
- a changed mechanism violates the constraint at least \(\ell\) times;
- there is at most one synchronous boundary with a two-sided guard;
- reports use binary XOR corruption and independent bit noise.
- the automatically positive inverse-channel bound uses noise-independent path
  selection; causal adaptation retains only the audited decoder bound.

Coding theory, parity-based fault detection and sparse secure estimation already
contain closely related ingredients. The candidate increment, if any, would
have to be the four-event joint-window formulation and its connection to online
attribution and refusal. That increment remains unverified against direct prior
work.

## Decision

This theorem rules out the simplest impossibility claim: unknown boundary and
unknown pre-state do not inevitably destroy event identifiability when a known
cross-source constraint, sparse attacks, persistent segments and observable
mechanism activity are available.

It does not yet justify D-MIR development. The next theory audit must determine
whether the syndrome assumption is merely a renamed fault-isolation residual,
and whether useful separation survives uncertain or learned constraints,
asynchronous boundaries, or a continuing mechanism that is already
contaminated.

