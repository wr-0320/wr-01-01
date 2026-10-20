# Joint Missingness and Dependent Evidence Study

Independent research repository for robust multi-source inference under
strategic deception. The completed RCL work and the passive Route B theory are
retained as frozen evidence. The active exploratory direction studies joint
inference under informative missingness and dependent evidence.

The first frozen protocol, implementation and development interpretation are:

- `docs/jmid_exploratory_protocol_v1.md`;
- `experiments/jmid_v1/`;
- `docs/development_result_jmid_v1.md`.

The independent-generator falsification follow-up is recorded in:

- `docs/jmid_falsification_protocol_v2.md`;
- `experiments/jmid_v2/`;
- `docs/development_result_jmid_v2.md`.

The dependency-graph uncertainty protocol, implementation audit and completed
single diagnostic are recorded in:

- `docs/jmid_graph_uncertainty_protocol_v3.md`;
- `docs/jmid_v3_implementation_audit.md`;
- `experiments/jmid_v3/`;
- `docs/development_result_jmid_v3.md`;
- `docs/jmid_post_v3_direction_decision.md`.

V3 passed its frozen empirical gate: Learned M4 improved Brier and NLL versus
M23 in all ten seeds. The true graph was also recovered in all ten seeds and
graph averaging collapsed to that single graph, so v3 did not demonstrate a
distinct robustness benefit from model averaging. The current method will not
enter formal confirmation. The active research question is now identifiability
and partial identification under joint MNAR and dependency uncertainty.

The first JMID-PI theory package is recorded in:

- `docs/jmid_pi_theory_protocol.md`;
- `docs/jmid_pi_identifiability_result.md`;
- `experiments/jmid_pi/`.

It separates target-supervised identification from unlabeled-target and
transport settings, gives an exact MNAR/dependent-evidence nonidentifiability
witness, and derives sharp finite-alphabet bounds under likelihood-ratio
envelopes. These are baseline results rather than a novelty claim. Further work
is limited to determining whether operational censorship and provenance
constraints imply a nonvacuous likelihood-ratio envelope.

The pre-v3 research-value audit remains in
`docs/jmid_research_value_review.md`, with a machine-readable comparison table
in `docs/jmid_prior_art_comparison.csv`. Its decision is a conditional GO for
one bounded v3 diagnostic and a NO-GO for formal evaluation or an algorithm
novelty claim.

This stage uses a controlled binary simulator and four small empirical models.
It is not a formal confirmation study or an algorithm novelty claim. The first
result permits only an independently generated falsification study with a
simple M2+M3 composition before any larger investment.

Run a wiring check without touching versioned evidence:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.jmid_v1.run \
  --smoke --output /tmp/jmid_v1_smoke
```

The frozen development artifacts are stored under
`evidence/development/jmid_v1/`, `evidence/development/jmid_v2/` and
`evidence/development/jmid_v3/`; do not rerun them as formal evidence. V2 failed
under a rotated dependency graph. V3 recovered its constrained graph exactly,
which preserves the empirical effect but leaves genuine structure uncertainty
and posterior identification unresolved.

The earlier Route B decision, literature map, problem statement and
pre-implementation gates remain archived in:

- `docs/route_b_research_decision.md`;
- `docs/route_b_literature_scope.md`;
- `docs/route_b_problem_formulation.md`;
- `docs/route_b_identifiability_audit.md`;
- `docs/route_b_finite_alphabet_theorem.md`;
- `docs/route_b_joint_window_theorem.md`;
- `docs/route_b_adaptive_attack_audit.md`;
- `docs/route_b_fault_isolation_audit.md`;
- `docs/route_b_final_fulltext_audit_protocol.md`;
- `docs/route_b_fulltext_audit_matrix.csv`;
- `docs/route_b_fulltext_access_log.md`;
- `docs/route_b_final_direction_decision.md`;
- `docs/route_b_direct_competitor_audit.md`;
- `docs/route_b_go_no_go_protocol.md`.

The passive Route B theory is currently `NO-GO` for implementation. Its final
full-text novelty audit is frozen but `INCONCLUSIVE-BLOCKED`: the required
primary papers are not present locally and the managed cloud proxy blocks the
scholarly hosts. No theorem coverage or novelty conclusion is inferred from
abstracts or metadata.

Historical evidence copied from the source project is read-only context under
`evidence/source_snapshot/`; new runs use `outputs/`, which is ignored by Git.

## Setup

```bash
./scripts/bootstrap.sh
```

This creates `.venv` inside this repository and installs a regular wheel. It does
not use an editable install or dependencies from the source project's environment.

The commands below reproduce or inspect the archived RCL stage. They are not the
active Route B implementation plan.

## Checks

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/test_rcl_granularity_baselines.py \
  tests/test_rcl_granularity_runner.py
./scripts/check_isolation.sh
```

## Granularity experiment

Small wiring check:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m experiments.run_rcl_granularity_development --smoke \
  --output-dir outputs/development/rcl_granularity_smoke
```

Three-seed development run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m experiments.run_rcl_granularity_development
```

Development output is exploratory. A later formal run requires a separately
frozen protocol, fresh output directory, and test draws that were not used for
development decisions.

## Controlled heterogeneity scan

The frozen protocol is in `docs/heterogeneity_scan_protocol.md`. Run the
three-seed development scan with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m experiments.run_rcl_heterogeneity_scan
```

The completed development result is documented in
`docs/development_result_heterogeneity_v1.md`. Under controlled, increasing
source-region contamination heterogeneity, the source-region reliability prior
did not show a stable advantage over global or source-level priors. The
predeclared development decision is therefore to use the global mixture as the
main candidate and retain source-region prior granularity as an optional
parameterization.

## Mechanism baselines

The matched development runner compares the global dual mixture with a single
mechanism, Student-t, Huber, and reliability-weighted Gaussian fusion:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m experiments.run_rcl_mechanism_baselines
```

Results and the exact full-precision audit are documented in
`docs/development_result_mechanism_baselines_v1.md` and
`docs/heterogeneity_result_audit.md`. The formal protocol is frozen in
`docs/formal_confirmation_protocol_v1.md`; its reserved entry point was executed
exactly once and must not be rerun for protocol v1. The formal outcome is
`NOT_CONFIRMED`; see `docs/formal_confirmation_result_v1.md` and the versioned
artifacts under `evidence/formal/rcl_mechanism_confirmation_v1/`.

