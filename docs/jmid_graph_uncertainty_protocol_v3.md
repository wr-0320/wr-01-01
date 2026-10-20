# JMID v3 dependency-graph uncertainty protocol

Status: **design frozen; implementation and execution not started**  
Freeze date: 2026-10-20

## Single research question

When the report-dependency graph is unknown and must be estimated from allowed
training observations, can structure selection or graph averaging retain the
probability-quality advantage of joint missingness--dependence modeling over
M23?

This is the last exploratory gate before any formal-confirmation protocol. It
does not authorize a more complex neural model, a larger RCL experiment or a
claim of graph-robust inference.

## Fixed data-generating scope

V3 keeps the v2 condition-D mechanism fixed:

- positive-report content censorship strength: `1.0`;
- directed relay-copy strength: `0.85`;
- binary state and binary uncensored reports;
- observation masks sampled independently conditional on uncensored content;
- state labels available during training for every method.

The source count increases from six to eight solely to create four disjoint
leader--relay edges and three nonzero graph-error levels. The true dependency
graph is a perfect matching on eight labeled sources. Within every matched edge,
the lower-index source is the leader and the higher-index source is the relay;
this orientation rule is fixed for generation and logging. The matching is
sampled uniformly from the 105 possibilities for each seed, then held fixed
across that seed's training and paired test episodes. Source identities are
retained; models may not relabel sources using the hidden graph.

The v3 generator may generalize the independently implemented v2 generator to
eight sources, but it must not restore v1's state-availability table or shared
availability variable. Tests must verify the realized copy rate, content-
conditional missingness and within-state dependence before results are read.

## Graph quantities that must remain distinct

For a true graph (G_\star) and a declared or estimated graph (G), define

\[
d_E(G,G_\star)=1-\frac{|E(G)\cap E(G_\star)|}{4}.
\]

Two different quantities are reported:

1. **injected graph mismatch**: the preselected distance between a supplied
   diagnostic graph and the true graph;
2. **structure-recovery error**: the realized distance between a graph selected
   from training data and the true graph.

Injected mismatch is an experimental control. It is not evidence that a
learning procedure made the same amount of error. For perfect matchings on
eight nodes, the nonzero attainable levels are `0.50`, `0.75` and `1.00`, named
light, moderate and severe only within this protocol.

## Candidate graph space

The candidate set is all 105 perfect matchings on the eight labeled sources.
It is generated deterministically in lexicographic edge order. The candidate
set, order and graph-distance implementation must be unit-tested before the
first development run.

No candidate graph may be added or removed using validation or test results.
The study does not address arbitrary overlapping graphs, unknown source
identity, a time-varying graph or more than one parent per relay.

## Shared data and splits

Development seeds are fixed to `4401` through `4410`. Each seed uses:

- 5,000 labeled training episodes;
- the first 3,750 training episodes as `structure-fit`;
- the remaining 1,250 as `graph-validation`;
- 5,000 paired test episodes generated independently from the same true graph.

All methods receive the same total training set, labels, masks, observed report
values and source identities. Only Oracle M4 receives the true graph. Raw
uncensored reports, copy indicators and the true graph are diagnostic-only for
all deployable methods.

After graph choice or graph-weight calculation, selected component models may
be refitted on all 5,000 training episodes. Test data cannot affect graph
selection, weights, smoothing or calibration.

## Frozen methods

### Graph-free and fixed-composition baselines

1. **M2:** per-source missing-aware categorical model; no graph.
2. **Learned-graph M3:** chooses a graph using M3 validation NLL, refits M3 on
   the full training set, and ignores missingness as in v1/v2.
3. **Learned-graph M23:** uses M2 plus the selected M3 graph with the frozen
   v2 formula `logit(M2) + logit(M3) - logit(M1)`.

M3 and M23 must not receive the M4-selected graph automatically. Each graph
selection criterion is part of the named method and is logged.

### Joint models

4. **Oracle M4:** fits the true graph. It is an upper-bound diagnostic and is
   never described as deployable.
5. **Learned-Graph M4:** for every candidate, fits M4 on `structure-fit`, selects
   the graph with minimum mean graph-validation NLL, breaks exact ties by frozen
   candidate order, and refits that graph on all training episodes.
6. **Graph-Averaged M4:** fits every candidate on `structure-fit`. Its frozen
   weight is

   \[
   w_G=\frac{\exp[-N_v(\operatorname{NLL}_G-\operatorname{NLL}_{\min})]}
   {\sum_{G'}\exp[-N_v(\operatorname{NLL}_{G'}-\operatorname{NLL}_{\min})]},
   \qquad N_v=1250.
   \]

   Each graph component is then refitted on the full training set. Test
   posterior probabilities are combined as
   (p_{\mathrm{avg}}=\sum_G w_Gp_G). No temperature, weight floor, top-k rule
   or test-time reweighting is permitted in v3.

### Fixed-mismatch diagnostic

For each seed, M4 is also evaluated using every candidate at distances 0.50,
0.75 and 1.00 from the true graph. Metrics are first averaged across candidates
within distance and seed, then summarized across seeds. This produces a
controlled degradation curve; it is not an additional deployable method.

## Complexity record

For eight sources, the approximate free categorical parameter counts are:

- M2: 33 including one state-prior degree of freedom;
- M3: 25 for a selected graph;
- M23: 73 across its fixed components;
- one M4 graph: 65;
- graph-averaged M4: 105 component models plus 104 effective graph weights.

Graph-Averaged M4 is therefore a robustness and uncertainty baseline, not a
parameter-matched competitor. Runtime, fitted-table count and peak memory must
be reported alongside predictive metrics.

## Outputs and metrics

Primary predictive metrics are Brier and NLL. Secondary metrics are accuracy,
ten-bin ECE, mean confidence on errors and unconditional high-confidence error
rate.

Required graph outputs are:

- true, selected and highest-weight graph identifiers;
- injected mismatch distance for every fixed-mismatch run;
- learned-graph recovery error (d_E);
- exact-recovery indicator;
- Graph-Averaged M4 weight on the true graph;
- posterior expected graph error \(\sum_Gw_Gd_E(G,G_\star)\);
- validation NLL for all 105 candidates;
- per-seed runtime and model count.

All predictive comparisons use the same paired test episodes. The statistical
unit is the seed. Report each seed, the paired mean difference and a two-sided
95% Student-t confidence interval with 9 degrees of freedom. Episodes must not
be treated as independent replicates for the main interval.

## Predeclared decisions

The primary deployable comparisons are Learned-Graph M4 minus Learned-Graph
M23 and Graph-Averaged M4 minus Learned-Graph M23.

### Continue to a separately frozen formal protocol

At least one deployable joint method must satisfy all of the following:

1. the upper endpoint of the seed-level 95% interval is below zero for both
   Brier and NLL differences versus M23;
2. Brier and NLL are lower in at least 8 of 10 seeds;
3. the lower endpoint of its accuracy-difference interval is above `-0.01`;
4. its conclusion is unchanged when all graph-selection and averaging steps are
   rerun from the saved training split without access to test data.

Passing this gate authorizes protocol design only. It does not make v3 formal
evidence.

### Stop the complex joint-table route

Stop if neither deployable joint method has lower mean Brier and NLL than M23,
or if either metric improves in fewer than 6 of 10 seeds. Retain Oracle M4 as a
conditional upper bound and archive v1--v3 as exploratory evidence.

### Retain only the known-graph result

If Oracle M4 retains its advantage but both deployable joint methods fail the
continue rule, restrict the contribution to settings with known or externally
validated dependency structure. Do not claim a solution to general graph
uncertainty and do not increase model complexity to rescue v3.

### Inconclusive

All remaining outcomes are inconclusive. Do not add seeds after reading the
result, change graph weights, narrow the candidate set, or enter formal
confirmation. A new study would require a new rationale and protocol.

## Required checks before execution

Implementation may begin only after tests establish:

1. enumeration of exactly 105 unique perfect matchings;
2. attainable distances and correct edge intersections;
3. no true-graph, raw-report or copy-indicator access in learned methods;
4. graph selection and averaging depend only on the frozen training split;
5. all six methods see identical episode IDs and label budgets;
6. fixed-mismatch aggregation occurs within seed before confidence intervals;
7. output directories are non-overwriting and evidence manifests include code
   commit, protocol hash, seeds and graph candidates.

No JMID v3 code or result exists at the time this protocol is frozen.

