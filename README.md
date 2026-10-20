# Dynamic Deception Inference Study

Independent research repository for robust multi-source inference under
strategic deception. The completed RCL work is retained as frozen evidence and
as a baseline. The active research direction, **Route B**, studies whether an
online system can distinguish a hidden-situation transition from a change in the
observation/deception mechanism and abstain when the distinction is not
identifiable.

Route B is currently in literature and mathematical validation. No D-MIR method
has been implemented. The research decision, literature map, problem statement
and pre-implementation gates are in:

- `docs/route_b_research_decision.md`;
- `docs/route_b_literature_scope.md`;
- `docs/route_b_problem_formulation.md`;
- `docs/route_b_identifiability_audit.md`;
- `docs/route_b_finite_alphabet_theorem.md`;
- `docs/route_b_joint_window_theorem.md`;
- `docs/route_b_direct_competitor_audit.md`;
- `docs/route_b_go_no_go_protocol.md`.

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

