"""Scale baseline data and evaluate counter-deception generalization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from deception_rl.beliefs import GRUBeliefEstimator, TrajectoryBeliefEstimator
from experiments.run_adaptation import evaluate_scripted_episode
from experiments.run_baseline import run_experiment as generate_baseline_data
from train_intent_model import train

_TRAINING_POLICIES = {
    "ModelA": ("honest_shortest_path", "random_policy"),
    "ModelB": (
        "honest_shortest_path",
        "random_policy",
        "scripted_deception",
    ),
}
_SUMMARY_FIELDS = [
    "model",
    "training_policies",
    "condition",
    "deception_steps",
    "episodes",
    "belief_accuracy",
    "commitment_accuracy",
    "deception_success_rate",
    "mean_true_goal_probability_at_commitment",
]
_EPISODE_FIELDS = [
    "model",
    "training_policies",
    "condition",
    "deception_steps",
    "episode",
    "seed",
    "true_goal",
    "belief_accuracy",
    "belief_at_commitment",
    "defender_decision",
    "commitment_accuracy",
    "deceiver_success",
    "trajectory",
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = row.copy()
            for field in ("belief_at_commitment", "trajectory"):
                if field in serialized:
                    serialized[field] = json.dumps(
                        serialized[field], separators=(",", ":"), sort_keys=True
                    )
            writer.writerow(serialized)


def run_data_scaling(
    output_dir: str | Path = "outputs/data_scaling",
    episodes_per_policy: int = 1000,
    seed_count: int = 10,
    test_episodes: int = 200,
    training_deception_steps: tuple[int, ...] = (1, 2, 3, 4),
    unseen_deception_steps: tuple[int, ...] = (5,),
    step_commit: int = 5,
    seed: int = 0,
    epochs: int = 120,
    patience: int = 20,
    batch_size: int = 128,
    device: str = "auto",
) -> dict[str, Path]:
    training_steps = tuple(sorted(set(training_deception_steps)))
    unseen_steps = tuple(sorted(set(unseen_deception_steps)))
    if episodes_per_policy < 1 or test_episodes < 2:
        raise ValueError("episodes_per_policy must be positive and test_episodes at least two.")
    if seed_count < 1 or episodes_per_policy % seed_count:
        raise ValueError("episodes_per_policy must be divisible by seed_count.")
    if not training_steps or any(step < 1 for step in training_steps):
        raise ValueError("training_deception_steps must contain positive values.")
    if not unseen_steps or any(step < 1 for step in unseen_steps):
        raise ValueError("unseen_deception_steps must contain positive values.")
    if set(training_steps) & set(unseen_steps):
        raise ValueError("Unseen deception steps must not occur in training variants.")
    if not 0 <= step_commit < 6:
        raise ValueError("step_commit must be in [0, 6) to precede goal arrival.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    training_data_path = destination / "training_episodes.csv"
    generate_baseline_data(
        episodes_per_policy=episodes_per_policy,
        seed=seed,
        output_path=training_data_path,
        estimator=TrajectoryBeliefEstimator(),
        scripted_deception_steps=training_steps,
        seed_count=seed_count,
    )

    checkpoints: dict[str, Path] = {}
    training_metrics = []
    for model_name, policies in _TRAINING_POLICIES.items():
        checkpoint_path = destination / f"{model_name.lower()}.pt"
        metrics = train(
            data_path=training_data_path,
            output_path=checkpoint_path,
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            training_policies=policies,
            device=device,
        )
        checkpoints[model_name] = checkpoint_path
        training_metrics.append(
            {
                "model": model_name,
                "training_policies": "+".join(policies),
                "training_device": metrics["training_device"],
                **metrics,
            }
        )

    episode_rows = []
    for model_name, policies in _TRAINING_POLICIES.items():
        estimator = GRUBeliefEstimator.from_checkpoint(checkpoints[model_name])
        for deception_steps in (*training_steps, *unseen_steps):
            if deception_steps in training_steps:
                condition = f"seen_scripted_deception_steps_{deception_steps}"
            else:
                condition = f"unseen_scripted_deception_steps_{deception_steps}"
            for episode in range(test_episodes):
                episode_seed = (
                    seed
                    + 10_000_019
                    + deception_steps * 1_000_003
                    + episode
                )
                goal = "A" if episode % 2 == 0 else "B"
                episode_rows.append(
                    evaluate_scripted_episode(
                        model_name=model_name,
                        training_policies=policies,
                        estimator=estimator,
                        condition=condition,
                        deception_steps=deception_steps,
                        episode=episode,
                        seed=episode_seed,
                        goal=goal,
                        step_commit=step_commit,
                    )
                )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in episode_rows:
        grouped.setdefault((row["model"], row["condition"]), []).append(row)
    summary_rows = []
    for (model_name, condition), rows in grouped.items():
        policies = _TRAINING_POLICIES[model_name]
        summary_rows.append(
            {
                "model": model_name,
                "training_policies": "+".join(policies),
                "condition": condition,
                "deception_steps": rows[0]["deception_steps"],
                "episodes": len(rows),
                "belief_accuracy": float(
                    np.mean([row["belief_accuracy"] for row in rows])
                ),
                "commitment_accuracy": float(
                    np.mean([row["commitment_accuracy"] for row in rows])
                ),
                "deception_success_rate": float(
                    np.mean([row["deceiver_success"] for row in rows])
                ),
                "mean_true_goal_probability_at_commitment": float(
                    np.mean(
                        [
                            row["belief_at_commitment"][row["true_goal"]]
                            for row in rows
                        ]
                    )
                ),
            }
        )

    summary_path = destination / "data_scaling_summary.csv"
    episode_results_path = destination / "data_scaling_episodes.csv"
    metrics_path = destination / "training_metrics.csv"
    _write_csv(summary_path, _SUMMARY_FIELDS, summary_rows)
    _write_csv(episode_results_path, _EPISODE_FIELDS, episode_rows)
    _write_csv(
        metrics_path,
        [
            "model",
            "training_policies",
            "unique_trajectories",
            "train_prefixes",
            "validation_prefixes",
            "validation_loss",
            "validation_accuracy",
            "best_epoch",
            "training_device",
            "checkpoint",
        ],
        training_metrics,
    )
    return {
        "training_data": training_data_path,
        "model_a": checkpoints["ModelA"],
        "model_b": checkpoints["ModelB"],
        "training_metrics": metrics_path,
        "episodes": episode_results_path,
        "summary": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data_scaling"))
    parser.add_argument("--episodes-per-policy", type=int, default=1000)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=200)
    parser.add_argument(
        "--training-deception-steps", type=int, nargs="+", default=[1, 2, 3, 4]
    )
    parser.add_argument(
        "--unseen-deception-steps", type=int, nargs="+", default=[5]
    )
    parser.add_argument("--step-commit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    outputs = run_data_scaling(
        output_dir=args.output_dir,
        episodes_per_policy=args.episodes_per_policy,
        seed_count=args.seed_count,
        test_episodes=args.test_episodes,
        training_deception_steps=tuple(args.training_deception_steps),
        unseen_deception_steps=tuple(args.unseen_deception_steps),
        step_commit=args.step_commit,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Wrote data-scaling summary to {outputs['summary']}")


if __name__ == "__main__":
    main()
