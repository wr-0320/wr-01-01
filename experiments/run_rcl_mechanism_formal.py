"""Frozen formal confirmation entry point; do not use for development tuning."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.run_rcl_mechanism_baselines import run


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "formal" / "rcl_mechanism_confirmation_v1"
FORMAL_SEEDS = tuple(range(13001, 13011))


def main():
    result = run(
        OUTPUT,
        seeds=FORMAL_SEEDS,
        episodes=100,
        likelihood_episodes=10,
        calibration_episodes=3,
        max_cycles=10,
        formal=True,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
