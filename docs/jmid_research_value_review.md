# JMID research-value review before v3

Status: **pre-v3 direction audit; novelty unresolved**  
Date: 2026-10-20

## Decision

JMID is worth one bounded v3 diagnostic because v1 and v2 show a reproducible
probability-quality effect and v2 exposes a precise failure variable: dependency
graph error. It is not yet justified as an algorithm paper direction.

The current M4 is a smoothed conditional contingency table over grouped report
outcomes in the alphabet `{missing, 0, 1}`. Learned-graph M4 is finite graphical-
model selection, and Graph-Averaged M4 is predictive model averaging. Those
building blocks are established method families. Their application to strategic
intelligence does not by itself supply novelty.

V3 therefore has diagnostic value: it can determine whether the v2 gain exists
outside the oracle-graph setting. Passing v3 would preserve the research
question, not validate a paper contribution. Failing v3 would stop the complex
joint-table route and make identifiability the preferred next question.

## What the evidence supports

| Claim | Current evidence | Allowed interpretation |
|---|---|---|
| Informative missingness alone can bias a model that ignores missingness | V1 condition B and v2 condition B | Supported within two synthetic generators |
| Dependent reports can cause poor probability quality when treated independently | V1/v2 condition C and high-confidence-error diagnostics | Supported within the tested pair-copy mechanisms |
| Joint modeling can outperform single-factor and fixed-composition baselines when the graph is correct | V2 condition D; all three seeds agree for Brier and NLL | Supported as an exploratory synthetic result |
| M4 is robust to dependency-graph error | Rotated-graph v2 test reverses the comparison | Rejected for fixed-graph M4 |
| The graph can be recovered from allowed observations under informative missingness | Not tested | Unsupported |
| Graph averaging preserves calibrated inference | V3 protocol only | Unsupported |
| JMID is novel relative to missing-data and graphical-model methods | No completed full-text audit | Unsupported |
| The effect transfers to real or semi-real intelligence data | No such dataset | Unsupported |

Three seeds establish neither a formal interval nor broad robustness. The v2
relative reductions are meaningful development signals, but they cannot answer
novelty, identifiability or external validity.

## Method-family comparison

| Established family | Informative missingness | Evidence dependence | Structure uncertainty | Typical output or guarantee | Collision with JMID | Remaining question that could matter |
|---|---|---|---|---|---|---|
| Selection, pattern-mixture and shared-parameter missing-data models | Native target | May include correlated outcomes or latent random effects | Usually assumes a specified missingness/outcome structure | Likelihood or sensitivity analysis under assumptions | M2/M4 encode a small discrete selection model | Does strategic content censorship plus unknown provenance create a new identifiable or partially identifiable target? |
| Bayesian networks with missing/latent variables and structural EM | Handles missing entries through latent-variable inference | Native graph dependence | Can learn graph and parameters | Observed-data likelihood and learned structure | Learned-Graph M4 is a finite special case | Does nonignorable missingness invalidate ordinary structure scores, and what extra assumptions repair it? |
| Chow--Liu or forest structure learning | Missingness is not inherently nonignorable | Learns tree dependence efficiently | Selects a tree/forest | Best tree approximation under its score | V3 perfect-matching search is simpler than general tree learning | Can a constrained provenance graph be recovered under MNAR censorship rather than complete data? |
| Bayesian model averaging over graphs | Missingness depends on component model | Native through graph components | Native target | Posterior/predictive averaging | Graph-Averaged M4 is direct finite BMA-style averaging | Are standard likelihood weights reliable when graph and missingness law are jointly misspecified? |
| Composite likelihood and dependence-adjusted evidence aggregation | Often treats missing terms by availability | Corrects repeated/dependent contributions | May avoid selecting one exact graph | Calibrated or robust estimating objective | M23 is a simple inclusion--exclusion composite | Can a graph-robust score retain calibration without enumerating a correct graph? |
| Copula or latent-factor models for multivariate reports | Can include censoring/missing-data likelihoods | Native target | Structure/factor count may be uncertain | Joint multivariate likelihood | More general than pair tables | Does JMID need more than a standard latent-factor/censoring model? |
| Robust Bayes, imprecise probability and posterior sensitivity | Can range over missingness mechanisms | Can range over dependency models | Treats graph as an uncertainty set | Posterior sets, worst-case risk or sensitivity bounds | Not tested by v1/v2 | Can useful posterior bounds be obtained without identifying one graph? |
| Probabilistic opinion pooling and forecast aggregation | Missing experts may be modeled | Dependence adjustment is a central concern | Often avoids an explicit causal graph | Aggregated probability and calibration | Closely matches multi-source evidence fusion | What observable provenance constraint distinguishes JMID from standard dependent forecast aggregation? |
| Selective prediction and reject options | Does not identify missingness | Does not identify dependence | Can react to model uncertainty | Risk--coverage or reject-cost guarantees | Relevant to direction C, not current M4 | Can graph/missingness uncertainty drive a justified refusal rule rather than a confidence threshold? |

The table is a conceptual collision audit, not a completed literature review.
The primary papers and exact theorem scopes still require full-text verification.
In particular, the audit must not assume that a family lacks a capability merely
because the current repository has not retrieved its papers.

## Reference ledger for targeted full-text verification

High-confidence foundational records:

1. Rubin (1976), *Inference and Missing Data*, Biometrika 63(3), 581--592.
2. Little (1993), *Pattern-Mixture Models for Multivariate Incomplete Data*,
   Journal of the American Statistical Association 88(421), 125--134.
3. Diggle and Kenward (1994), *Informative Drop-Out in Longitudinal Data
   Analysis*, Journal of the Royal Statistical Society Series C 43(1), 49--93.
4. Chow and Liu (1968), *Approximating Discrete Probability Distributions with
   Dependence Trees*, IEEE Transactions on Information Theory 14(3), 462--467.
5. Madigan and Raftery (1994), *Model Selection and Accounting for Model
   Uncertainty in Graphical Models Using Occam's Window*, JASA 89(428),
   1535--1546.
6. Hoeting, Madigan, Raftery and Volinsky (1999), *Bayesian Model Averaging: A
   Tutorial*, Statistical Science 14(4), 382--417.
7. Genest and Zidek (1986), *Combining Probability Distributions: A Critique
   and an Annotated Bibliography*, Statistical Science 1(1), 114--135.
8. Varin, Reid and Firth (2011), *An Overview of Composite Likelihood Methods*,
   Statistica Sinica 21, 5--42.

Full-text search must additionally locate the closest primary results on:

- structural EM or Bayesian-network structure learning with missing data;
- graphical identification under nonignorable missingness;
- joint dependence and missingness-model sensitivity;
- robust or imprecise posterior inference over graph uncertainty;
- dependent forecast/source aggregation with unknown correlation;
- provenance-aware duplicate evidence fusion.

No novelty sentence is authorized until those closest results are compared by
model assumptions, observed information, target estimand and guarantee.

## JMID's smallest defensible research object

The empirical object is not “a new joint table.” It is this question:

> How much posterior distortion is caused by jointly misspecifying an
> informative missingness law and a source-dependency structure, and can that
> distortion be controlled without knowing the true graph?

There are three possible scientific outcomes:

1. **Recoverable structure:** training observations identify enough of the graph
   for a deployable joint model to retain calibrated predictions.
2. **Robust without recovery:** the graph is not reliably recovered, but model
   averaging or a conservative fusion rule controls predictive loss.
3. **Partial or non-identifiability:** multiple graph/missingness pairs induce
   the same observed law but different state posteriors, requiring bounds,
   external provenance information or refusal.

Outcome 1 is what v3 tests. Outcomes 2 and 3 are not solved by the frozen
protocol and are the possible bridge to directions A and B.

## Direction ranking after the audit

| Direction | Current value | Main collision risk | Decision |
|---|---|---|---|
| A. Unknown-structure robust inference | High diagnostic continuity | Plain graph selection/averaging is established | Run only the frozen v3 gate; require a robustness mechanism or guarantee before a paper claim |
| B. Missingness--dependence identifiability | Highest theoretical upside | Missing-data graphical identification is a mature field | Prepare only if v3 exposes recovery limits and a precise unaddressed estimand remains |
| C. Deception-aware selective inference | Meaningful applied backup | A confidence threshold or standard reject option is insufficient | Defer until model uncertainty can be quantified |
| D. Reliability and evidence deduplication | Easy to implement | Strong overlap with reliability weighting, latent factors and dependent aggregation | Do not prioritize without a concrete unmet constraint |

## Pre-v3 investment decision

The frozen v3 experiment is worth implementing only as the final bounded test
already specified in `jmid_graph_uncertainty_protocol_v3.md`. Do not expand its
candidate model class, add neural graph learning, increase source count beyond
the frozen design or begin formal confirmation.

After v3:

- if a deployable method passes the frozen gate, perform the targeted full-text
  audit before designing formal evidence;
- if only Oracle M4 succeeds, retain the known-graph result and move the research
  question toward identifiability or external provenance requirements;
- if M23 matches deployable joint methods, stop the complex joint-table route;
- if the result is inconclusive, do not tune the graph weights or append seeds.

Thus the current direction decision is **conditional GO for one v3 diagnostic,
NO-GO for formal evaluation or a JMID algorithm claim**.

