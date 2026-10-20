# JMID-PI minimal identifiability result

Status: **baseline theory completed; JMID-specific theoretical increment not yet established**

Date: 2026-10-20

## 1. The information boundary comes first

Let `O=(M,Y_obs)` denote everything observable at deployment, including which
sources are missing. The estimand is

\[
\psi(o)=P(Z=1\mid O=o).
\]

### Proposition 1: target-supervised posterior identification

If the target population joint law of `(Z,O)` is observed and `P(O=o)>0`, then

\[
\psi(o)=\frac{P(Z=1,O=o)}{P(O=o)}
\]

is identified, regardless of whether the MNAR mechanism or dependency graph is
identified.

This is a population statement. Finite samples still create estimation error,
and structured models may improve efficiency or calibration. They do not turn
this supervised estimand into a structural nonidentifiability problem.

JMID v3 belongs to this regime: every training episode has a state label and is
drawn from the same frozen generator as its test data. Therefore a new JMID-PI
claim must concern an unlabeled target environment or a specified transport
problem.

## 2. Exact nonidentifiability witness

Consider binary `Z` with prior `P(Z=1)=1/2` and three reports. Let all three raw
reports equal one binary value `R`, and let all three masks equal `M`. The
observable outcome is therefore one of `***`, `000`, or `111`.

Two different latent dependency graphs can generate the duplicated reports:

- Model A: chain `1 -> 2 -> 3`;
- Model B: fork `1 -> 2` and `1 -> 3`.

Their raw-report and observation probabilities are:

| Model | Z | P(R=1|Z) | P(M=1|Z,R=0) | P(M=1|Z,R=1) |
|---|---:|---:|---:|---:|
| A | 0 | 1/5 | 3/4 | 1/2 |
| A | 1 | 4/5 | 1/2 | 3/4 |
| B | 0 | 3/10 | 5/7 | 2/3 |
| B | 1 | 7/10 | 2/3 | 5/7 |

In every row, observation depends on raw content, so both constructions are
MNAR. Direct multiplication gives:

| Model | Z | P(***) | P(000) | P(111) |
|---|---:|---:|---:|---:|
| A | 0 | 3/10 | 3/5 | 1/10 |
| A | 1 | 3/10 | 1/10 | 3/5 |
| B | 0 | 3/10 | 1/2 | 1/5 |
| B | 1 | 3/10 | 1/5 | 1/2 |

After marginalizing `Z`, both models induce exactly

\[
P(O=***)=\frac{3}{10},\qquad
P(O=000)=P(O=111)=\frac{7}{20}.
\]

Their posteriors nevertheless differ:

\[
P_A(Z=1\mid111)=\frac67,qquad
P_B(Z=1\mid111)=\frac57,
\]

and

\[
P_A(Z=1\mid000)=\frac17,qquad
P_B(Z=1\mid000)=\frac27.
\]

### Proposition 2: failure of target-unlabeled point identification

Under the model class containing A and B, the marginal target law of `O` and
the known prior `P(Z=1)=1/2` do not identify `psi(111)` or `psi(000)`.

**Proof.** A and B satisfy the same observable law and prior, but the displayed
posterior values differ. Any functional of only those observed inputs has the
same value under A and B and therefore cannot equal both true posteriors.

The same construction proves nonidentifiability in a transport regime without
a declared source-to-target invariant: attach the same labeled source law to
both candidates and use A or B only as the target law.

### Boundary of the witness

The graph ambiguity uses deterministic duplication. The induced distribution
is not faithful to a unique graph and lacks strict positivity over all report
patterns. The witness is still valid for the broad model class, but it does not
prove a new impossibility result under positivity, faithfulness or bounded-copy
assumptions. Moreover, posterior nonidentifiability already arises from the
unlabeled mixture; the example does not yet isolate a uniquely joint MNAR--graph
phenomenon.

## 3. Identification without full graph recovery

Write

\[
q_z(o)=P(O=o\mid Z=z),\qquad \pi=P(Z=1).
\]

If `pi`, `q_0(o)` and `q_1(o)` are identified for an outcome of positive mass,
then

\[
\psi(o)=\frac{\pi q_1(o)}{(1-\pi)q_0(o)+\pi q_1(o)}.
\]

Thus full dependency-graph recovery is sufficient in some parametric models
but is not necessary. Any calibration design or invariant that identifies the
class-conditional observable likelihood ratio

\[
L(o)=q_1(o)/q_0(o)
\]

also identifies the posterior through

\[
\psi(o)=\frac{\pi L(o)}{1-\pi+\pi L(o)}.
\]

This is the correct object for the next structural analysis: operational MNAR
and provenance assumptions must imply information about `L(o)`, rather than
requiring every graph edge to be recovered.

## 4. Partial identification

Assume a finite observed alphabet, known prior `pi`, and known target marginal
probabilities `p_o=P(O=o)>0`.

### 4.1 No likelihood restrictions

Let `u_o=P(Z=1,O=o)`. Marginal compatibility and the known prior imply

\[
\max(0,p_o+\pi-1)\le u_o\le\min(p_o,\pi).
\]

Therefore

\[
\frac{\max(0,p_o+\pi-1)}{p_o}
\le\psi(o)\le
\frac{\min(p_o,\pi)}{p_o}.
\]

For `pi=1/2` and `p_111=7/20`, this interval is `[0,1]`. Knowing the prior alone
does not produce a useful conclusion in the counterexample.

### 4.2 Likelihood-ratio envelopes

Suppose externally justified constants satisfy

\[
0\le \ell_o\le L(o)\le u_o^{L}<\infty.
\]

Define

\[
a_o=p_o\frac{\pi\ell_o}{1-\pi+\pi\ell_o},\qquad
b_o=p_o\frac{\pi u_o^{L}}{1-\pi+\pi u_o^{L}}.
\]

The uncertainty class is feasible exactly when

\[
\sum_o a_o\le\pi\le\sum_o b_o.
\]

### Theorem 3: sharp finite-alphabet posterior bounds

For any outcome `j`, the sharp identified interval is

\[
\boxed{
\frac{\max\{a_j,\;\pi-\sum_{o\ne j}b_o\}}{p_j}
\le \psi(j)\le
\frac{\min\{b_j,\;\pi-\sum_{o\ne j}a_o\}}{p_j}
}.
\]

**Proof.** The likelihood-ratio constraints are equivalent to the interval
restrictions `a_o <= P(Z=1,O=o) <= b_o`. The known prior adds the single equality
`sum_o P(Z=1,O=o)=pi`. Holding the mass for outcome `j` fixed, the other interval
masses can attain every value between the sum of their lower endpoints and the
sum of their upper endpoints. Intersecting that attainable range with the local
interval `[a_j,b_j]` gives the displayed endpoints. Each endpoint is attainable
by distributing the remaining prior mass across the other intervals, so the
bounds are sharp.

For example, with `pi=1/2`, observed masses `(7/20,7/20,3/10)`, and likelihood-
ratio intervals `[1,9]`, `[1/9,1/4]`, and `[2/3,1]`, the local interval for the
first outcome is `[1/2,9/10]`. The global prior constraint sharpens it to
`[4/5,9/10]`.

## 5. Research decision

The counterexample, supervised-identification proposition and finite-alphabet
bound are exact and machine checked. They are also baseline probability and
mixture-identification results. They do not yet establish a JMID-specific
theoretical contribution.

The only justified continuation is one bounded structural question:

> Can operational restrictions on content-dependent censorship and uncertain
> evidence provenance imply nonvacuous, testable bounds on the observable
> likelihood ratio `L(o)` without identifying the full graph and MNAR law?

Possible outcomes are fixed in `jmid_pi_theory_protocol.md`:

- a derived, nonvacuous envelope supports continued partial-identification work;
- an envelope requiring external sensitivity constants restricts the result to
  conditional sensitivity analysis;
- no improvement beyond generic mixture bounds stops JMID-PI and triggers the
  previously identified selective-inference alternative.

No algorithm implementation or formal experiment is authorized by the present
result.
