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

