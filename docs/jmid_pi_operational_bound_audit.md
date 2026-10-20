# JMID-PI operational-bound feasibility audit

Status: **RESTRICT TO SENSITIVITY ANALYSIS; stop the strong-theory route**

Date: 2026-10-20

## Question and answer

This audit asks one final question:

> Can finite censorship, dependence and environment-shift constraints imply a
> nontrivial target posterior bound without specifying the target observable
> likelihood ratio directly?

The mathematical answer is **yes, conditionally**. A strict bound follows when
all three constraints are finite and a labeled source environment anchors each
source's class-conditional report signal. The operational answer is weaker:
the required constants cannot be verified from unlabeled target reports alone,
and the resulting uncertainty factor grows exponentially with source count.

Under the predeclared decision rule, JMID-PI therefore contracts to conditional
sensitivity analysis. It does not continue as the main theoretical innovation
route, and no new inference algorithm or formal experiment is authorized.

## Setup

Let `R=(R_1,...,R_S)` be the uncensored target reports and let `O=o` contain a
mask plus all visible values. Denote the labeled source-environment marginal
report laws by

\[
f^{s}_{zi}(a)=P_s(R_i=a\mid Z=z)
\]

and their likelihood ratios by

\[
\lambda_i^s(a)=f^s_{1i}(a)/f^s_{0i}(a).
\]

All probabilities used in ratios are assumed positive on the audited support.
The compatible set `C(o)` contains every raw report vector consistent with the
visible part of `o`.

Three operational sensitivity parameters are imposed.

### C. Finite censorship contrast

Conditional on `(Z,R)`, source masks factor across sources. For each source and
each local observed/missing outcome compatible with its raw content, the ratio
of its observation-kernel probabilities between any two audited state/content
conditions is bounded by `Gamma_C >= 1`. In particular,

\[
\Gamma_C^{-S}\le
\frac{K_1(o\mid r)}{K_0(o\mid r)}
\le\Gamma_C^S,
\qquad r\in C(o).
\]

This is stronger than merely bounding the overall missing rate. It requires an
audit that sometimes reveals censored content or uses controlled probes.

### D. Finite source dependence

For each target state, define the joint/product density ratio

\[
\delta_z^t(r)=
\frac{f_z^t(r)}{\prod_i f_{zi}^t(r_i)}.
\]

Assume

\[
\Gamma_D^{-1}\le\delta_z^t(r)\le\Gamma_D.
\]

This permits dependence but rules out arbitrary duplication or concentration.
It implies

\[
\Gamma_D^{-2}\le\delta_1^t(r)/\delta_0^t(r)\le\Gamma_D^2.
\]

### T. Finite class-conditional transport

For every state, source and supported raw value, assume

\[
\Gamma_T^{-1}\le
\frac{f^t_{zi}(a)}{f^s_{zi}(a)}
\le\Gamma_T.
\]

Consequently, the target single-source likelihood ratio differs from its
source counterpart by at most `Gamma_T^2` in either direction.

This condition transports labeled source information. Bounds only on changes
in the unlabelled marginal report frequency would not anchor the target class
likelihood ratio.

## Operational likelihood-ratio theorem

Define

\[
A_S=\Gamma_C^S\Gamma_D^2\Gamma_T^{2S}.
\]

### Theorem 1

Under C, D and T, the target observable likelihood ratio satisfies

\[
\boxed{
\frac{1}{A_S}
\min_{r\in C(o)}\prod_{i=1}^{S}\lambda_i^s(r_i)
\le
\frac{P_t(O=o\mid Z=1)}{P_t(O=o\mid Z=0)}
\le
A_S
\max_{r\in C(o)}\prod_{i=1}^{S}\lambda_i^s(r_i)
}.
\]

Applying the monotone map

\[
g_\pi(L)=\frac{\pi L}{1-\pi+\pi L}
\]

gives a strict target posterior interval for any known `0 < pi < 1`.

### Proof

For a compatible raw vector,

\[
\frac{f_1^t(r)}{f_0^t(r)}
=
\frac{\delta_1^t(r)}{\delta_0^t(r)}
\prod_i
\frac{f_{1i}^t(r_i)}{f_{0i}^t(r_i)}.
\]

Conditions D and T bound this expression by `Gamma_D^2 Gamma_T^(2S)`
times the source marginal likelihood-ratio product, in both directions.
Condition C supplies the remaining `Gamma_C^S` factor for the observation
kernel. Finally,

\[
\frac{P_t(O=o\mid Z=1)}{P_t(O=o\mid Z=0)}
=
\frac{\sum_{r\in C(o)}f_0^t(r)K_0(o\mid r)
\left[\frac{f_1^t(r)K_1(o\mid r)}{f_0^t(r)K_0(o\mid r)}\right]}
{\sum_{r\in C(o)}f_0^t(r)K_0(o\mid r)}.
\]

The last expression is a convex combination of the bracketed ratios, so it
lies between their minimum and maximum. This proves the bound.

The theorem provides a strict guarantee. Its endpoints are not claimed to be
globally sharp because normalization, shared marginals and graph constraints
can prevent every local extreme from being attained simultaneously.

## Which constraints are necessary?

| Constraint set | Can it bound the target observable LR? | Reason |
|---|---|---|
| C alone | No | Even with state-invariant complete observation, class report probabilities can approach zero and one, making the LR arbitrarily large or small. |
| D alone | No | `Gamma_D=1` only gives conditional independence; state-specific source marginals remain unrestricted. |
| T on unlabeled marginals or mechanisms alone | No | It does not attach report values to target states. |
| C + D | No | They control observation and dependence distortion but provide no class-signal anchor. |
| Source class signal + T only | No in the general MNAR/dependent class | Unbounded target selection or dependence can still overturn the transported marginal signal. |
| Source class signal + C + D + T | Yes | Theorem 1 gives a finite positive LR interval. |

Directly bounding `P_t(O=o|Z=z)/P_s(O=o|Z=z)` would also give an interval, but
that is merely an observable likelihood-ratio sensitivity assumption. Theorem 1
shows how more interpretable component restrictions imply such an interval.

## Exact finite example

For two binary sources, suppose the first visible report is 1 and the second is
missing. The compatible raw vectors are `(1,0)` and `(1,1)`. Let the labeled
source likelihood ratios be

\[
\lambda_1^s(1)=3/2,\quad
\lambda_2^s(0)=4/5,\quad
\lambda_2^s(1)=5/4.
\]

Set

\[
\Gamma_C=10/9,\qquad
\Gamma_D=10/9,\qquad
\Gamma_T=21/20.
\]

Then `A_2=2401/1296`, and Theorem 1 gives

\[
L_t(o)\in
\left[\frac{7776}{12005},\frac{12005}{3456}\right]
\approx[0.648,3.474].
\]

With target prior `pi=1/5`, the posterior interval is

\[
P_t(Z=1\mid o)\in
\left[\frac{1944}{13949},\frac{12005}{25829}\right]
\approx[0.139,0.465].
\]

This is strictly narrower than `[0,1]`, so the construction passes mathematical
nontriviality. It remains wide relative to many operational decisions.

## Scaling and verifiability audit

The logarithm of the uncertainty multiplier is

\[
\log A_S=S\log\Gamma_C+2\log\Gamma_D+2S\log\Gamma_T.
\]

Uncertainty therefore compounds exponentially with the number of sources. If
all three sensitivity constants equal 1.1 and `S=8`, then `A_S=1.1^26`, about
11.9. Even before accounting for unobserved compatible contents, a neutral
source likelihood ratio can expand to approximately `[0.084,11.9]`; with prior
one half this becomes a posterior interval near `[0.077,0.923]`.

The constants also require external information:

- `Gamma_C` needs revealed censored reports, controlled probes or a trusted
  audit channel;
- `Gamma_D` needs sufficiently rich state-stratified joint-report calibration;
- `Gamma_T` needs target labels, a validation sample or a defensible transport
  guarantee.

Routine unlabeled target observations cannot verify all three. Treating their
values as analyst-selected parameters makes the result sensitivity analysis,
not point identification from battlefield reports.

## Final route decision

The narrow feasibility question has been answered and is now closed.

- A computable strict posterior bound can be derived from operationally named
  constraints.
- The bound relies on external calibration and compounds rapidly with source
  count.
- The proof is multiplicative density-ratio algebra and currently has no
  demonstrated theoretical increment over robust Bayes, distribution-shift or
  MNAR sensitivity analysis.
- No claim of sharpness or literature novelty is made.

JMID-PI is retained as a conditional sensitivity-analysis component, not as the
main theoretical paper direction. Further JMID-PI theorem expansion, algorithm
development and formal experiments stop here. The next candidate research
direction is deception-risk-aware selective inference, which requires its own
bounded literature and problem audit before any implementation.
