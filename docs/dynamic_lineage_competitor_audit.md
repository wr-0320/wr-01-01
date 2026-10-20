# Direct-competitor audit: unknown dynamic evidence lineage

Status: **candidate retained; originality not locally confirmed; prototype NO-GO**

Date: 2026-10-20

## Decision

The supplied theorem audit reports no single frozen competitor that jointly
provides:

1. inference of unknown, time-varying evidence lineage;
2. propagation of lineage error into a posterior or predictive-risk guarantee;
3. an anytime-valid alarm when that guarantee ceases to be supported.

That intersection is narrower than generic truth discovery, duplicate removal,
multi-source calibration or online shift detection. It is a plausible theory
candidate, but the absence of a single end-to-end method is not by itself an
original contribution. Combining three known modules is also insufficient.

## Claim chart

| Work | Unknown dynamic lineage | Error control | Anytime alarm | Boundary relevant to the candidate |
|---|---|---|---|---|
| Zhang et al., dynamic dependent truth discovery | Partial | No reported guarantee | No | Dynamic dependence scores do not certify the resulting posterior |
| Dong et al., copy-aware truth discovery | Static, assumption-dependent | No | No | Value confidence is not a calibrated posterior guarantee |
| Brehard--Krishnamurthy, data-incest removal | Known graph only | Conditional exact recovery | No | Does not address unknown or misspecified lineage |
| Liu et al., MuSCI | No report-lineage inference | Target-site marginal coverage | No | Multi-source means sites, not copied evidence chains |
| Prinster et al., WATCH | No | Sequential-test guarantee only | Yes | Detects failure without attributing or bounding unknown-lineage overconfidence |
| Huang et al., OMPB | No | Online predictive-risk certificate | No reported alarm guarantee | Temporal dependence does not identify source-copy lineage |

All entries above are currently `externally_reported`: the detailed audit was
provided in the research record, while the present runtime still lacks access
to the paper bodies. They must not be cited as locally reproduced theorem
findings until the evidence column in the matrix is completed.

## What remains potentially distinct

The defensible candidate is not “detect duplicates” or “combine truth discovery
with conformal prediction.” It is the following implication, if it can be made
nontrivial:

> Observable timing, provenance, or trusted-audit information identifies an
> effective-evidence equivalence structure up to a quantified error; that error
> yields an explicit bound on posterior overconfidence; a sequential diagnostic
> raises an alarm with controlled false-alarm probability when the bound's
> observable premises fail.

The first implication must not assume the true graph. The second must control a
posterior or decision-relevant quantity rather than a lineage-recovery score.
The third must be a stopping-time guarantee rather than a plotted drift signal.

## Current boundary

- **Closest existing capability:** known-topology deduplication can remove
  repeated information exactly under its graph conditions.
- **Closest dependency learner:** dynamic truth discovery reportedly estimates
  dependence states but supplies no posterior reliability certificate.
- **Closest certificate:** MuSCI and OMPB reportedly control prediction objects,
  but do not model a report-copy graph.
- **Closest alarm:** WATCH reportedly supplies anytime false-alarm control, but
  does not diagnose or bound unknown-lineage double counting.

The candidate therefore passes only a **provisional theory-screening gate**. It
does not pass the originality gate, because the full-text findings have not been
locally reproduced and no new implication has yet been proved.

## Next permitted work

The external audit has been accepted as a provisional evidence basis for one
bounded mathematical-feasibility check, recorded in
`dynamic_lineage_posterior_certificate_theorem.md`. This does not upgrade the
literature rows to locally verified evidence or confirm novelty.

Do not implement a lineage learner, online detector, fusion model or simulator.
Originality remains `INCONCLUSIVE-BLOCKED` until the six paper bodies are
verified, while the theorem package separately decides mathematical
feasibility.
