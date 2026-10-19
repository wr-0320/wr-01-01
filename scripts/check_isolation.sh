#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="/workspace/wr-01"

if find "$project_root" -path "$project_root/.venv" -prune -o -type l -print -quit | grep -q .; then
  echo "Isolation check failed: symbolic links exist outside the local virtual environment." >&2
  exit 1
fi

if git -C "$source_root" status --porcelain | grep -q .; then
  echo "Isolation check failed: the source repository is not clean." >&2
  git -C "$source_root" status --short >&2
  exit 1
fi

case "$project_root/.venv" in
  "$source_root"/*)
    echo "Isolation check failed: virtual environment is under the source repository." >&2
    exit 1
    ;;
esac

echo "Isolation checks passed."

