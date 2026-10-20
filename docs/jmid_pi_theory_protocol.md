# JMID-PI theory work package

Status: **scope frozen; no algorithm or formal experiment authorized**

Date: 2026-10-20

## Research target

JMID-PI studies the target-environment posterior

\[
\psi_t(o)=P_t(Z=1\mid O=o),
\]

where `Z` is the hidden situation and `O=(M,Y_obs)` contains the complete
missingness mask and every observed report value. Raw censored reports,
provenance links and the target dependency graph are unavailable.

The target is posterior identification, not recovery of every latent graph edge
or every parameter of the missingness process.

## Information regimes

The theory must state which of the following regimes it uses.

1. **Target-supervised:** the population law of `(Z,O)` is observed in the
   target environment. This matches the asymptotic information boundary of the
   v3 training protocol.
2. **Target-unlabeled:** only the target law of `O` is observed; the state prior
   may be known or bounded externally.
3. **Transport:** labeled `(Z,O)` data are available in a source environment,
   while only `O` is observed in the target. Any invariant mechanism or bounded
   shift connecting the two environments must be stated explicitly.

Nonidentifiability from the marginal law of `O` is relevant only to regimes 2
and 3. It cannot be used to claim that v3's target-supervised posterior is
unidentifiable.

## First finite-alphabet scope

- `Z` is binary.
- Reports and masks induce a finite observed alphabet.
- At least three labeled sources are used when two different dependency
  skeletons are required.
- Missingness may depend on the hidden state and raw report content.
- No estimator receives censored values, copy indicators or the graph.
- The first partial-identification result may assume a known target state prior,
  but that assumption must remain visible in every bound.

## Required deliverables

1. An exact pair of latent constructions with the same target observable law
   and different values of `psi_t(o)`.
2. A proof explaining why target labels eliminate that particular ambiguity.
3. Sufficient conditions that identify the posterior without requiring full
   graph identification.
4. A finite-alphabet identified set with an explicit feasibility condition and
   a proof of sharpness.
5. A scope audit stating which results are baseline probability algebra and
   which unresolved step could contain a JMID-specific theoretical increment.

Exact arithmetic checks may accompany the proof. They are verification tools,
not a new inference algorithm.

## Fixed decision rule

- **Continue theory:** only if operational restrictions on MNAR selection and
  evidence dependence imply a nonvacuous likelihood-ratio or posterior bound,
  or yield a stronger impossibility result under positivity and a meaningful
  dependency assumption.
- **Restrict:** if useful bounds require external calibration or explicit
  sensitivity constants; the result must be presented as conditional
  sensitivity analysis.
- **Stop JMID-PI:** if the only valid results are generic unlabeled-mixture,
  missing-data or graphical-model facts under unchanged assumptions.

No new predictor, larger simulator, v3 rerun or formal confirmation begins
under this work package.
