"""Run the complete Stage 14 evaluation and paper-artifact build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.build_stage14_paper_suite import build
from experiments.run_stage14_formal_baselines import run


ROOT = Path(__file__).resolve().parents[1]


def run_suite(output_root: Path, seed: int = 0):
    formal_output = output_root / "stage14_formal_baselines"
    paper_output = output_root / "stage14_paper_suite"
    formal_summary = run(formal_output, seed)
    paper = build(paper_output)
    return {
        "formal_summary_rows": len(formal_summary),
        "paper": paper,
        "training_started": False,
        "algorithms_modified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    print(json.dumps(run_suite(arguments.output_root, arguments.seed), indent=2))

