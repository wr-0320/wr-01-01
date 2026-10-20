# Route B literature scope and novelty audit

Status: **pre-implementation scoping record**  
Review date: 2026-10-20

## Question being audited

The candidate research question is narrower than online changepoint detection:

> Given a stream of partial, delayed and possibly coordinated multi-source
> reports, can an online system distinguish a change in the hidden situation
> from a change in the adversary-controlled observation mechanism, and can it
> decline to attribute the change when the two explanations are not
> identifiable from the available reports?

The internal name **D-MIR** (Dynamic Mechanism Identification and Robust
Inference) is only a placeholder. It is not a novelty claim or an implemented
method.

## What the existing repository establishes

The earlier RCL studies motivate the question but do not answer it:

- A fixed dual-mechanism likelihood clearly improved on matched single-mechanism
  and linear-weighting controls.
- Its preregistered probability-quality advantage over Huber was not confirmed.
- Finer source-region reliability priors had no stable advantage.
- The correlation-aware and intent-coupled extensions both produced NO-GO
  development results.

These results suggest that improving a fixed contamination likelihood is an
unpromising main line. They provide no evidence that a dynamic method can
attribute changes correctly.

## Literature map

| Area | Established capability | Why it does not by itself establish the proposed contribution | Required comparison |
|---|---|---|---|
| Bayesian online changepoint detection (BOCPD) and quickest detection | Detects a change in a predictive distribution and trades detection delay against false alarms | A generic changepoint does not identify whether the hidden state or the emission mechanism changed | BOCPD on the observation stream and a robust BOCPD variant |
| Switching state-space models, switching HMMs and factorial HMMs | Jointly filters discrete regimes and dynamic states | A composite regime can predict well while its factors have no unique semantic interpretation; standard filtering does not supply a state-change-versus-mechanism-change certificate | Matched switching/factorial HMM with the same state cardinality and emissions |
| Interacting multiple-model and adaptive filtering | Tracks changes among a known bank of motion/noise models | Usually assumes a specified model bank and treats model switches as dynamics; attribution under strategic, observation-only manipulation is not its main target | IMM or a finite model-bank Bayesian filter when compatible with the simulator |
| Robust filtering and heavy-tailed likelihoods | Limits the influence of outliers and distributional contamination | Downweighting can preserve state estimates without detecting a persistent, strategic change or explaining its source | Huber, Student-t and the archived RCL |
| Secure/resilient state estimation | Gives recovery conditions under sparse or bounded sensor attacks, often using known plant dynamics and attack sparsity | This is the closest theoretical neighbor, but recovery of the physical state is different from online posterior attribution between a physical transition and an emission-regime transition | A sparse-attack resilient estimator when its assumptions can be represented fairly |
| Fault detection and isolation | Uses structured residuals, unknown-input observers or model banks to isolate process, actuator and sensor faults | It may already address much of the state-change-versus-observation-change distinction; strategic event mixtures and evidence-based refusal must be checked directly | Closest process-fault versus sensor-fault isolation method |
| Adversarial HMM inference | Studies how corrupted observations manipulate hidden-state inference | Demonstrating or optimizing an attack does not necessarily solve online change attribution or calibrated refusal | The closest reproducible adversarial-HMM method found in the full-text review |
| Latent-model identifiability | Gives conditions such as full-rank transitions and linearly independent emissions for parameter recovery, often up to label permutation | Global parameter identifiability does not imply that a finite online observation block uniquely identifies which latent factor changed | Use these results to state assumptions; do not present standard HMM identifiability as a new theorem |
| Selective prediction and reject options | Permits abstention when predictive risk is high | Predictive entropy alone need not measure confusion between the two causal explanations | Entropy/confidence abstention baseline at matched coverage |

The candidate gap is therefore **online model-based attribution of distributional
change to a latent situation transition or an adversarial emission-mechanism
transition, together with an explicit confusability diagnostic and abstention**.
This remains a candidate gap until the closest papers have been checked in full.

## Core reference ledger

The entries below are divided by verification status so that incomplete network
access cannot silently turn a search lead into evidence of novelty.

### High-confidence bibliographic records

1. Adams, R. P. and MacKay, D. J. C. (2007), *Bayesian Online
   Changepoint Detection*, arXiv:0710.3742.
2. Fearnhead, P. and Liu, Z. (2007), *On-line inference for multiple
   changepoint problems*, Journal of the Royal Statistical Society: Series B,
   69(4), 589–605.
3. Basseville, M. and Nikiforov, I. V. (1993), *Detection of Abrupt Changes:
   Theory and Application*, Prentice Hall.
4. Ghahramani, Z. and Jordan, M. I. (1997), *Factorial Hidden Markov Models*,
   Machine Learning, 29, 245–273.
5. Ghahramani, Z. and Hinton, G. E. (2000), *Variational Learning for
   Switching State-Space Models*, Neural Computation, 12(4), 831–864.
6. Fawzi, H., Tabuada, P. and Diggavi, S. (2014), *Secure Estimation and
   Control for Cyber-Physical Systems Under Adversarial Attacks*, IEEE
   Transactions on Automatic Control, 59(6), 1454–1467.
7. Pajic, M., Weimer, J., Bezzo, N., Sokolsky, O., Pappas, G. J. and Lee,
   I. (2017), *Attack-Resilient State Estimation for Noisy Dynamical
   Systems*, IEEE Transactions on Control of Network Systems, 4(1), 82–92.
8. Allman, E. S., Matias, C. and Rhodes, J. A. (2009), *Identifiability of
   parameters in latent structure models with many observed variables*, The
   Annals of Statistics, 37(6A), 3099–3132.
9. Alexandrovich, G., Holzmann, H. and Leister, A. (2016), *Nonparametric
   identification and maximum likelihood estimation for hidden Markov
   models*, Biometrika, 103(2), 423–434.
10. Chow, C. K. (1970), *On optimum recognition error and reject tradeoff*,
    IEEE Transactions on Information Theory, 16(1), 41–46.
11. Chen, J. and Patton, R. J. (1999), *Robust Model-Based Fault Diagnosis
    for Dynamic Systems*, Kluwer Academic Publishers.
12. Blanke, M., Kinnaert, M., Lunze, J. and Staroswiecki, M. (2006),
    *Diagnosis and Fault-Tolerant Control*, 2nd edition, Springer.
13. Ding, S. X. (2008), *Model-based Fault Diagnosis Techniques: Design
    Schemes, Algorithms, and Tools*, Springer.
14. Pasqualetti, F., Doerfler, F. and Bullo, F. (2013), *Attack Detection
    and Identification in Cyber-Physical Systems*, IEEE Transactions on
    Automatic Control, 58(11), 2715–2729.

These references establish neighboring methods and standard limits. They do not
establish that no direct solution to the proposed problem exists.

### Search leads requiring metadata and full-text verification

- *Robust and Scalable Bayesian Online Changepoint Detection*.
- *Doubly Robust Bayesian Inference for Non-Stationary Streaming Data with
  beta-Divergences*.
- *Manipulating hidden-Markov-model inferences by corrupting batch data*.
- Byzantine/distributed quickest change detection with compromised sensors.
- Joint state/attack-mode estimation in switching or hidden-mode cyber-physical
  systems.
- Multi-sensor fault isolation methods that explicitly separate process changes
  from sensor-model changes.
- Unknown-input observers and multiple-model methods for process-fault versus
  sensor-fault isolation.

No novelty sentence may cite a lead from this second list until its full text and
closest references have been reviewed.

## Full-text search protocol

When scholarly database access is available, search IEEE Xplore, Scopus or Web
of Science, Google Scholar, ACM DL, arXiv and major publisher indexes with the
following query families:

1. `("hidden state" OR HMM OR "state space") AND (attack OR deception OR
   adversarial) AND (changepoint OR "change point" OR switching)`
2. `("state change" AND "sensor change") AND (distinguish OR identifiability OR
   isolation)`
3. `(fault isolation OR attack isolation) AND (process change OR mode change)
   AND online`
4. `(factorial HMM OR switching state-space) AND (identifiability OR
   attribution) AND adversarial`
5. `(Byzantine OR compromised sensor) AND "quickest change detection"`
6. `("strategic deception" OR "adversarial observation") AND (belief OR
   posterior) AND abstain`
7. `("fault isolability" OR "structured residual") AND (sensor fault AND
   process fault)`
8. `("unknown input observer" OR "multiple model fault diagnosis") AND
   isolation`

For every plausible direct competitor, record its hidden variables, transition
assumptions, attack knowledge, trusted-source assumptions, online outputs,
theoretical guarantee, abstention behavior, datasets and released code. Follow
backward and forward citations until two consecutive rounds add no new direct
competitor.

## Current search limitation

The managed cloud proxy returned HTTP 403 for OpenAlex, arXiv, PMLR, DOI and
GitHub API requests on 2026-10-20. GitHub HTML was reachable but is not a
substitute for a scholarly index or full text. Accordingly, this document makes
no claim that the literature search is exhaustive and labels the research gap
as provisional.

