# Unknown dynamic evidence lineage: direct-competitor audit protocol

Status: **frozen comparison set; theory screening only; prototype prohibited**

Date: 2026-10-20

## Candidate question

Under unknown and time-varying report lineage, can a system estimate the
effective amount of independent evidence and propagate lineage uncertainty into
both a posterior-error certificate and an anytime-valid failure alarm?

This is a screening question. It does not authorize a lineage learner, fusion
model, simulator or experiment.

## Frozen closest-work set

The comparison is limited to the six rows in
`dynamic_lineage_competitor_matrix.csv`:

1. dynamic and dependent truth discovery;
2. copy-aware truth discovery;
3. removal of repeated information when the flow topology is known;
4. multi-source conformal inference under site shift;
5. anytime-valid online failure detection;
6. online martingale PAC-Bayes risk certification.

The set may be corrected if a supplied bibliographic record is wrong. It must
not be enlarged after seeing whether a paper supports the candidate claim.

## Capability definitions

### Unknown-lineage identification

Counts only when the method infers copying, retelling, common upstream causes,
or evidence equivalence classes from observations. Deduplication on a known
graph is conditional support. Temporal autocorrelation alone does not count.

### Posterior or predictive error control

Counts only for a stated probabilistic result: posterior approximation error,
coverage, or a high-probability target-risk bound. A score named confidence or
reliability, an optimization convergence statement, and an empirical accuracy
gain do not count.

### Online failure alarm

Counts only for a sequential alarm or stopping time with an explicit false-
alarm guarantee, such as

\[
\Pr_{H_0}(\tau<\infty)\leq\alpha.
\]

Streaming updates or an uncalibrated shift score do not count.

## Evidence rule

The user supplied a theorem-level audit summary for all six works. Its claims
are preserved in the matrix as `externally_reported`. The current runtime has
not retrieved the paper bodies, so those claims are not marked
`locally_fulltext_verified`. A title page, abstract, search result or secondary
summary cannot upgrade the status.

For local verification, each row must record the exact section, theorem,
proposition or explicit absence supporting all three capability judgments.

## Predeclared decision

- **THEORY-SCREENING GO:** at least one observable lineage signal is available,
  no frozen competitor already supplies the joint guarantee under equal or
  weaker assumptions, and a nontrivial error-propagation target can be stated.
- **CONDITIONAL ONLY:** dependency can be estimated but no new reliability
  guarantee is available.
- **STOP:** a frozen competitor directly implies the same joint result, or
  copying and common cause remain observationally indistinguishable without an
  auditable anchor, timing signal or intervention.
- **INCONCLUSIVE-BLOCKED:** the external theorem audit cannot be reproduced from
  lawful full text. This does not count as evidence of novelty.

Only `THEORY-SCREENING GO` permits a separate minimal-theorem protocol. None of
these outcomes directly permits prototype development.
