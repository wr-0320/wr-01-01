# Deception-risk-aware selective inference audit protocol

Status: **frozen research-value audit; implementation prohibited**

Date: 2026-10-20

## Candidate question

In a target environment with incomplete, dependent and potentially manipulated
multi-source reports, can an inference system choose among prediction, trusted
verification and abstention while providing a risk guarantee that existing
selective, conformal, robust and active-testing methods do not already provide
under equal or weaker assumptions?

This audit concerns the novelty and feasibility of that question. It does not
authorize a confidence threshold, neural selector, conformal wrapper, POMDP or
new simulator.

## Minimum mathematical object

- hidden situation `Z` in a finite set;
- deployment observation `O`, including report values and missingness masks;
- action `a` in `{predict class, verify, abstain}`;
- error cost `c_e`, trusted-verification cost `c_v` and abstention cost `c_a`;
- a declared target-law or adversary class `U`;
- accepted-prediction coverage, selective error risk, verification rate and
  total decision cost.

For a predictor-selector pair `(h,s)`, the primary reliability target is

\[
\sup_{Q\in\mathcal U}
Q(h(O)\ne Z\mid s(O)=1)\le\alpha,
\]

subject to a nonzero coverage or cost requirement. A guarantee without an
explicit `U`, calibration information boundary and probability statement is
not admissible.

## Frozen literature set

The audit is restricted to the 15 primary works listed in
`selective_inference_competitor_matrix.csv`, divided across:

1. selective classification and reject options;
2. conformal or risk-controlling prediction under shift or attack;
3. adversarial robustness and attack detection;
4. sequential testing, verification and deferral.

Exact assumptions, theorem numbers and failure conditions require the original
full text. Abstracts, snippets, secondary surveys and repository README files
cannot close a matrix row. A bibliographic replacement is permitted only when
a listed record is wrong or inaccessible and must be logged before reading the
replacement's result.

## Questions for every work

1. What is random and what can the adversary choose?
2. Is exchangeability, iid calibration, a fixed contamination budget or a
   known shift weight required?
3. Is the guarantee marginal, conditional, class-conditional, PAC or
   asymptotic?
4. Does it control selective error, set coverage, total cost or only a surrogate?
5. Can it abstain, request information or defer to an expert?
6. Does it allow dependent evidence, informative missingness or adaptive attack?
7. What trusted labels, audit samples or verification channel are available?
8. Does its theorem already imply the proposed result after a direct change of
   notation?

## Predeclared decisions

### GO to a minimum prototype

All of the following are required:

- every closest-competitor row is supported by full text;
- one operational threat class permits nonzero coverage and a verifiable risk
  guarantee;
- the guarantee is not a direct application of selective classification,
  robust conformal prediction, distributionally robust risk control, active
  hypothesis testing or learning-to-defer;
- the claimed difference can be stated as a theorem, impossibility boundary or
  falsifiable algorithmic advantage before code is written.

### STOP as a strong algorithm direction

Stop if the proposed method is only an existing confidence/reject rule plus a
warfare simulator, or if an existing method provides the same guarantee under
equal or weaker information and threat assumptions.

### IMPOSSIBLE UNDER CURRENT INFORMATION

Narrow or abandon the target if unrestricted deception makes every nonzero-
coverage guarantee fail and no operationally defensible restriction can be
observed or externally audited.

### INCONCLUSIVE-BLOCKED

If closest full texts remain unavailable, preserve the matrix and do not begin
an implementation, experiment or novelty claim.
