"""Rerun H1/H2 and H3 with both rule-based and trained GRU estimators."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from deception_rl.beliefs import GRUBeliefEstimator, TrajectoryBeliefEstimator
from experiments.run_baseline import run_experiment as run_h1_h2
from experiments.run_h3_commitment import run_experiment as run_h3

_SUMMARY_FIELDS = [
    "experiment",
    "estimator",
    "deceiver_policy",
    "episodes",
    "deceiver_success_rate",
    "final_belief_accuracy",
    "first_step_wrong_belief_rate",
    "commitment_accuracy",
    "commitment_wrong_belief_rate",
    "mean_deception_cost",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _summarize_h1_h2(estimator_name: str, path: Path) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_rows(path):
        groups[row["deceiver_policy"]].append(row)

    summaries = []
    for policy, rows in groups.items():
        final_correct = []
        first_step_wrong = []
        for row in rows:
            goal = row["true_goal"]
            belief = json.loads(row["estimated_belief"])
            final_correct.append(belief[goal] > 0.5)
            trace = json.loads(row["belief_trace"])
            first_step = next((item for item in trace if item["step"] == 1), None)
            false_goal = "B" if goal == "A" else "A"
            first_step_wrong.append(
                bool(first_step and first_step[false_goal] > 0.5)
            )
        summaries.append(
            {
                "experiment": "H1_H2",
                "estimator": estimator_name,
                "deceiver_policy": policy,
                "episodes": len(rows),
                "deceiver_success_rate": rows[0]["success_rate"],
                "final_belief_accuracy": sum(final_correct) / len(rows),
                "first_step_wrong_belief_rate": sum(first_step_wrong) / len(rows),
                "commitment_accuracy": "",
                "commitment_wrong_belief_rate": "",
                "mean_deception_cost": "",
            }
        )
    return summaries


def _summarize_h3(estimator_name: str, path: Path) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_rows(path):
        groups[row["deceiver_policy"]].append(row)

    summaries = []
    for policy, rows in groups.items():
        wrong_belief = []
        for row in rows:
            belief = json.loads(row["belief_at_commitment"])
            wrong_belief.append(belief[row["true_goal"]] < 0.5)
        summaries.append(
            {
                "experiment": "H3",
                "estimator": estimator_name,
                "deceiver_policy": policy,
                "episodes": len(rows),
                "deceiver_success_rate": rows[0]["deceiver_success_rate"],
                "final_belief_accuracy": "",
                "first_step_wrong_belief_rate": "",
                "commitment_accuracy": sum(
                    int(row["decision_correct"]) for row in rows
                )
                / len(rows),
                "commitment_wrong_belief_rate": sum(wrong_belief) / len(rows),
                "mean_deception_cost": sum(
                    int(row["deception_cost"]) for row in rows
                )
                / len(rows),
            }
        )
    return summaries


def run_comparison(
    checkpoint_path: str | Path = "outputs/intent_gru.pt",
    output_dir: str | Path = "outputs/estimator_comparison",
    episodes_per_policy: int = 20,
    step_commit: int = 2,
    seed: int = 0,
) -> Path:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"GRU checkpoint not found at {checkpoint}; run train_intent_model.py first."
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    estimators = [
        ("rule", TrajectoryBeliefEstimator()),
        ("gru", GRUBeliefEstimator.from_checkpoint(checkpoint)),
    ]
    summaries = []
    for estimator_name, estimator in estimators:
        h1_h2_path = destination / f"h1_h2_{estimator_name}.csv"
        h3_path = destination / f"h3_{estimator_name}.csv"
        run_h1_h2(
            episodes_per_policy=episodes_per_policy,
            seed=seed,
            output_path=h1_h2_path,
            estimator=estimator,
        )
        run_h3(
            episodes_per_policy=episodes_per_policy,
            step_commit=step_commit,
            seed=seed,
            output_path=h3_path,
            estimator=estimator,
        )
        summaries.extend(_summarize_h1_h2(estimator_name, h1_h2_path))
        summaries.extend(_summarize_h3(estimator_name, h3_path))

    summary_path = destination / "estimator_comparison.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/intent_gru.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/estimator_comparison"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--step-commit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary_path = run_comparison(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        episodes_per_policy=args.episodes,
        step_commit=args.step_commit,
        seed=args.seed,
    )
    print(f"Wrote estimator comparison to {summary_path}")


if __name__ == "__main__":
    main()
