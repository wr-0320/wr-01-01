# Can target-relevant lineage classes tighten posterior certificates?

Status: **negative core result; target-equivalence route STOP for algorithmic novelty**

Date: 2026-10-20

## 1. Setup

Fix observed information `o=(x,a)` and let `C(o)` be any nonempty set of
lineage models retained by a valid confidence procedure. Each model `g` implies
a situation posterior

\[
T_o(g)=P_g(Z=1\mid O=o).
\]

Define target-relevant equivalence at `o` by

\[
g\sim_o h\quad\Longleftrightarrow\quad T_o(g)=T_o(h).
\]

A stronger global equivalence can require equality for every possible `o`; the
argument below is unchanged. Let `q_o(g)` denote the equivalence class and let
`\bar T_o(q_o(g))=T_o(g)`.

The complete-graph posterior envelope is

\[
I_C(o)=
\left[
\inf_{g\in C(o)}T_o(g),
\sup_{g\in C(o)}T_o(g)
\right].
\]

## 2. Quotienting alone cannot tighten the envelope

### Theorem 1: posterior-image invariance

For every retained graph set `C(o)`,

\[
I_C(o)=
\left[
\inf_{k\in q_o(C(o))}\bar T_o(k),
\sup_{k\in q_o(C(o))}\bar T_o(k)
\right].
\]

**Proof.** The quotient map deletes duplicate elements of the image
`{T_o(g):g in C(o)}` but neither adds nor removes a distinct posterior value.
Its infimum and supremum are therefore unchanged.

This theorem gives an exact negative answer to the proposed shortcut. Recovering
only a posterior-relevant equivalence class may reduce computation and avoid
unnecessary edge recovery, but simply replacing graphs by their classes cannot
make a sharp graph envelope narrower.

## 3. What is required for strict tightening

Suppose a separate procedure constructs a confidence set `K_delta(o)` directly
for the true class `q_o(g_star)` and satisfies

\[
P_{g_\star}\{q_O(g_\star)\in K_\delta(O)\}\ge 1-\delta.
\]

Then

\[
I_K(o)=
\left[
\inf_{k\in K_\delta(o)}\bar T_o(k),
\sup_{k\in K_\delta(o)}\bar T_o(k)
\right]
\]

is a valid posterior certificate by the same set-inclusion argument used in the
earlier graph-envelope theorem.

### Proposition 1: necessary form of a strict improvement

`I_K(o)` is strictly narrower than `I_C(o)` only if the direct class procedure
excludes at least one posterior-extreme class retained by `q_o(C(o))`.

That exclusion needs its own valid statistical evidence. It can arise from:

- additional timestamp, provenance or trusted-audit information;
- a more efficient test of the target functional than the particular graph
  confidence procedure;
- structural restrictions that rule out some posterior values.

It does not arise from relabeling or quotienting the same feasible models. With
the same sharp observable model class, the identified posterior image is already
the smallest set that can be justified without further restrictions.

## 4. An observational-equivalence lower bound

### Theorem 2: indistinguishable models force certificate width

Let two admissible models `g_0` and `g_1` induce the same distribution for all
available observations `O`. Suppose their target posteriors differ by at least
`Delta` almost surely:

\[
|T_O(g_1)-T_O(g_0)|\ge\Delta>0.
\]

For any possibly randomized interval `I(O)` satisfying marginal coverage at
least `1-delta` under each model, the common observable law obeys

\[
P\{\operatorname{width}(I(O))\ge\Delta\}\ge 1-2\delta.
\]

**Proof.** Under the common observable law, the two coverage events each have
probability at least `1-delta`. Their intersection has probability at least
`1-2delta`. On that intersection the interval contains both target posterior
values, so its width is at least their separation.

The conclusion also applies across time when the two dynamic lineage models
induce the same joint law for the entire observed sequence. More timestamps do
not help unless their distributions differ across the candidate mechanisms.

## 5. Consequences for evidence lineage

### 5.1 Internal topology can be irrelevant without improving the bound

If partial provenance identifies which reports descend from the same root but
not the internal retelling tree, all trees inside a root group can imply the
same posterior. Quotienting removes those nuisance trees and may greatly reduce
computation. The posterior envelope was already unchanged across them, so its
statistical width does not improve.

### 5.2 Effective evidence can be inferred without full graph recovery

In a model where the posterior factors through an effective-evidence functional

\[
S(g,o)=\sum_{c\in\text{independent root groups}}\log BF_c(o),
\qquad
T_o(g)=\operatorname{logistic}\{\operatorname{logit}(\pi_0)+S(g,o)\},
\]

one may construct a confidence set directly for `S`. This can avoid recovering
edge directions within each group. It is ordinary inference for a lower-
dimensional functional unless a new, lineage-specific identification or
optimality theorem is supplied.

### 5.3 Copying versus independent agreement remains decisive

An all-copy explanation and an independent-roots explanation can imply sharply
different values of `S` and hence different posteriors. If their observable
timing, provenance and audit distributions are identical, Theorem 2 prevents a
uniformly narrow certificate. Calling both structures members of an “effective
evidence” family does not remove this ambiguity.

### 5.4 A lineage-specific instance of the lower bound

Let `Z` have prior one half. Under `g_0`, a root report `R` has accuracy `p_0`
and every later report directly copies it. Under `g_1`, an unobserved common
upstream report has accuracy `p_1` and all observed reports copy that upstream
value. In both models the observed vector is all zero or all one with equal
probability, so its distribution is identical. When the vector is all one, the
posteriors are `p_0` and `p_1`; when it is all zero, they are `1-p_0` and
`1-p_1`. The posterior gap is therefore `|p_1-p_0|` for every observation.

Theorem 2 implies that any interval with coverage `1-delta` under both models
has width at least `|p_1-p_0|` with probability at least `1-2delta`. Timestamps,
provenance or trusted reliability audits must distinguish the mechanisms before
this gap can be reduced.

## 6. Comparison with the previous certificate

The earlier certificate used a graph confidence set and bounded posterior width
through its graph diameter. The proposed equivalence-class approach changes
three things only under additional work:

| Change | Mathematical effect | Novelty implication |
|---|---|---|
| Merge graphs with identical posteriors | Reduces representation size | No tightening by Theorem 1 |
| Infer the posterior functional directly | May avoid full graph recovery | Standard functional confidence-set idea unless a new identification result is proved |
| Add provenance, timing or audit restrictions | Can exclude extreme classes | Tightening comes from the added information and must be justified independently |

Thus a strict comparison against a deliberately loose graph confidence set is
not enough. The relevant comparator is the sharp posterior image under the same
observable assumptions.

## 7. Decision against the frozen criteria

- **Theoretical validity:** passes. A valid class confidence set gives a valid
  posterior interval.
- **Strict tightness:** fails as an intrinsic property. Quotienting alone leaves
  the exact envelope unchanged.
- **Assumption strength:** additional auditable signals can help, but they are
  the source of identification rather than the equivalence relation itself.
- **Originality:** fails at the current level. Direct inference of a posterior-
  relevant functional is a standard projection/partial-identification move.
- **Practical value:** computational compression is plausible, but it does not
  establish a new reliability guarantee.

The breakthrough criterion is therefore not met. The target-equivalence route
is **STOPPED as the core of a new algorithmic contribution**. The prior
finite-sample posterior certificate remains a correct conditional result, and
Theorem 2 is a useful impossibility boundary, but neither authorizes a prototype.

Any future revival would require genuinely new observable structure and a sharp
or minimax-optimal posterior bound that cannot be obtained by applying standard
functional confidence sets to that structure. Online alarms, simulations and
model implementation remain out of scope.
