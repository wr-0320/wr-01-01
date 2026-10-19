# RCL Paper

Independent research repository for the working paper **Source-Region
Conditioned Mixture Likelihood Inference for Contaminated Multi-Source
Observations**.

The current milestone freezes the study boundary and compares four matched
structures: single mechanism, global mixture, source-level mixture, and the full
source-region RCL. Historical evidence copied from the source project is read-only
context under `evidence/source_snapshot/`; new runs use `outputs/`, which is ignored
by Git.

## Setup

```bash
./scripts/bootstrap.sh
```

This creates `.venv` inside this repository and installs a regular wheel. It does
not use an editable install or dependencies from the source project's environment.

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

