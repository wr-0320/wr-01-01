# Route B adaptive-attack and noise audit

Status: **proof-scope correction**  
Date: 2026-10-20

## Audit finding

The earlier finite-alphabet documents used two different proof devices as if
they had the same information requirements:

1. an inverse-channel argument applied to \(K_\eta\mu\), where a complete
   noiseless center path is mixed before independent BSC noise;
2. a nearest-event-set argument based on the realized number of bit flips.

They behave differently under an observation-adaptive attacker. If the attack
at time \(t\) depends on past noisy reports, the final center path is generally
statistically dependent on past noise. Its joint observation law need not be
representable as \(K_\eta\mu\) for a noise-independent distribution \(\mu\).
The inverse-channel proof therefore does not apply as written.

The nearest-set proof is pathwise and survives causal adaptation under the
conditions below. This audit narrows the theorem instead of assuming that
`non-anticipating` means noise-independent.

## Information structures

Let \(E_t\in\{0,1\}^S\) be ordinary measurement noise,
\(Y_t=c_{Z_t}\oplus A_t\) the pre-noise report and
\(X_t=Y_t\oplus E_t\) the observed report. Let \(\mathcal F_{t-1}\) contain
past observations, attacker randomization and permitted public context.

### I0: open-loop or noise-independent path selection

The complete attack path is drawn independently of \(E_{1:T}\). Conditional on
the center path, observations pass through a fixed independent BSC. The law has
the form \(K_\eta\mu\).

Both the inverse-channel bound and the nearest-set bound are valid.

### I1: causal observation-adaptive attack

At time \(t\), \(A_t\) is \(\mathcal F_{t-1}\)-measurable, while \(E_t\) is
independent of \(\mathcal F_{t-1}\) and follows the fixed BSC law. The attacker
may react to all earlier noisy reports and may randomize.

The final center path depends on \(E_{1:T-1}\), so the fixed-mixture
representation generally fails. The pathwise decoder bound remains valid if
the sparsity, persistence, guard-window and syndrome-activity constraints hold
almost surely for every realized path.

A two-step example makes the failure concrete. Let one source have
\(Y_1=0\), \(X_1=Y_1\oplus E_1\), and let the adaptive rule choose
\(Y_2=X_1\). The marginal center-path distribution assigns probabilities
\(1-\eta\) and \(\eta\) to 00 and 01. The actual law has

\[
\Pr(X_1=0,X_2=0)=(1-\eta)^2.
\]

If that marginal center distribution were instead passed independently through
\(K_\eta\), the same probability would be

\[
(1-\eta)\big((1-\eta)^2+\eta^2\big),
\]

which differs for \(0<\eta<1/2\). Marginalizing the adaptive center path first
therefore does not restore the fixed-mixture representation.

### I2: current-noise-aware path selection with exogenous noise

The attacker observes \(E_t\) before choosing \(A_t\), but the noise process
itself remains exogenous and i.i.d. The standard controlled-channel
factorization fails. Nevertheless, if the final pre-noise center is still
required to belong to the correct event set on every path, the nearest-set
argument remains pathwise valid: \(X=Y\oplus E\) and
\(d_H(X,Y)=\lVert E\rVert_0\).

This information pattern is physically unusual and is not needed for the Route
B model. It shows that seeing noise and controlling its law are different.

### I3: channel-controlling or selectively censoring attacker

The attacker changes the distribution of \(E_t\), chooses which noisy reports
are retained, induces missingness based on the realized noise, or violates the
event-center constraints on some histories.

Neither previous bound is valid without a new channel and selection model.

## Proposition: causal-adaptive decoder bound

Let the four noiseless event-center sets have minimum pairwise Hamming distance
\(D>0\). Suppose:

1. every realized center path belongs to its declared event set;
2. the measurement-noise bits are i.i.d. Bernoulli\((\eta)\), independent of
   the past when generated;
3. the attacker may use any I1 causal strategy and private randomization.

Let \(n\) be the number of observed bits and

\[
\alpha_D=\Pr\left\{
\operatorname{Binomial}(n,\eta)
\ge\left\lceil D/2\right\rceil
\right\}.
\]

For every admissible strategy, nearest-event-set decoding has error at most
\(\alpha_D\). Consequently, for any two event-induced laws \(P,Q\),

\[
\operatorname{TV}(P,Q)\ge 1-2\alpha_D.
\]

This gives a strictly positive uniform certificate when
\(\alpha_D<1/2\).

### Proof

For every realized attack, state and noise history, let \(Y\) be the resulting
center path and \(X=Y\oplus E\). Causal dependence between \(Y\) and past
components of \(E\) does not change the pathwise identity

\[
d_H(X,Y)=\lVert E\rVert_0.
\]

If \(\lVert E\rVert_0<D/2\), the observation is closer to its realized center
than to every center in another event set. Therefore an error implies
\(\lVert E\rVert_0\ge\lceil D/2\rceil\). The exogenous noise count is binomial,
so the error probability is at most \(\alpha_D\), uniformly over adaptive
strategies.

For a pair of events, let \(A\) be the region decoded as the first event. Then
\(P(A)\ge1-\alpha_D\) and \(Q(A)\le\alpha_D\), giving the TV bound. \(\square\)

## What no longer follows

Under I1 adaptation, the documents must not claim

\[
\delta\ge(1-2\eta)^n
\]

from the fixed-channel inverse norm. That proof requires I0 path selection.
When \(\alpha_D\ge1/2\), the audited adaptive proof supplies no positive
quantitative lower bound. Finite-alphabet compactness and causal-channel
injectivity may still imply qualitative separation, but that requires a
separate proof and is not asserted here.

The activity requirement must also hold almost surely. Requiring only an
expected number of syndrome-active times permits positive probability of a
clean path, which can reintroduce overlap with \(H_0\) or \(H_Z\).

## Correction to the earlier future-noise statement

The first finite-alphabet note stated that seeing future noise invalidates both
the inverse and binomial arguments. That was too broad. Future-noise dependence
invalidates the fixed-mixture inverse proof, but the nearest-set argument still
holds whenever the exogenous additive-noise identity and event-center membership
hold pathwise. The decoder bound fails when the attacker controls the noise or
the observation-selection process, not merely because the chosen center is
statistically dependent on noise.

## Implication for Route B

Causal strategic adaptation does not destroy the distance-sensitive guarantee,
but it removes the only previously stated bound that was automatically positive
for every \(\eta<1/2\). A useful adaptive result therefore needs one of:

- enough event distance for \(\alpha_D<1/2\);
- a valid controlled-channel separation bound;
- a different anytime-valid evidence process with assumptions stated at the
  filtration level.

This is a proof correction and a sharper research boundary. It is not a new
algorithmic contribution.

