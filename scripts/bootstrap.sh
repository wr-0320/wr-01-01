#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIP_CACHE_DIR="$project_root/.cache/pip"
python -m venv "$project_root/.venv"
"$project_root/.venv/bin/python" -m pip install \
  'torch==2.14.1+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
"$project_root/.venv/bin/python" -m pip install "$project_root[test]"
"$project_root/.venv/bin/python" -m pip check

