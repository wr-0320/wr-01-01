"""Train honest-prior and deception-aware GRUs, then evaluate adaptation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from deception_rl.agents import (
    BeliefBasedDefender,
    CommittedDefender,
    ScriptedDeception,
)
from deception_rl.beliefs import GRUBeliefEstimator, TrajectoryBeliefEstimator
from deception_rl.envs.gridworld import GUARD_A, GUARD_B, GridWorldParallelEnv
from experiments.run_baseline import _visible_trajectory, run_experiment
from train_intent_model import train

_TRAINING_POLICIES = {
    "ModelA": ("honest_shortest_path", "random_policy"),
    "ModelB": (
        "honest_shortest_path",
        "random_policy",
        "scripted_deception",
    ),
}
_GOALS = ("A", "B")
_DECISION_NAMES = {GUARD_A: "A", GUARD_B: "B"}
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


def evaluate_deceiver_episode(
    model_name: str,
    training_policies: tuple[str, ...],
    estimator: GRUBeliefEstimator,
    deceiver: Any,
    condition: str,
    deception_steps: int | str,
    episode: int,
    seed: int,
    goal: str,
    step_commit: int,
) -> dict[str, Any]:
    env = GridWorldParallelEnv()
    deceiver.reset(seed=seed)
    observations, _ = env.reset(seed=seed, options={"goal": goal})
    defender = CommittedDefender(
        step_commit=step_commit,
        defender=BeliefBasedDefender(estimator=estimator),
    )
    belief_correct = []

    while env.agents:
        current_belief = defender.defender.estimate_belief(observations["defender"])
        predicted_goal = "A" if current_belief["A"] >= current_belief["B"] else "B"
        belief_correct.append(int(predicted_goal == goal))
        defender_action = defender.act(observations["defender"])
        deceiver_action = deceiver.act(observations["deceiver"])
        observations, rewards, _, _, _ = env.step(
            {"deceiver": deceiver_action, "defender": defender_action}
        )

    final_belief = defender.defender.estimate_belief(observations["defender"])
    final_prediction = "A" if final_belief["A"] >= final_belief["B"] else "B"
    belief_correct.append(int(final_prediction == goal))
    if defender.decision is None or defender.belief_at_commitment is None:
        raise RuntimeError("Episode ended before the Defender committed.")

    return {
        "model": model_name,
        "training_policies": "+".join(training_policies),
        "condition": condition,
        "deception_steps": deception_steps,
        "episode": episode,
        "seed": seed,
        "true_goal": goal,
        "belief_accuracy": float(np.mean(belief_correct)),
        "belief_at_commitment": defender.belief_at_commitment,
        "defender_decision": _DECISION_NAMES[defender.decision],
        "commitment_accuracy": int(_DECISION_NAMES[defender.decision] == goal),
        "deceiver_success": int(rewards["deceiver"] > 0),
        "trajectory": _visible_trajectory(observations["defender"]),
    }


def evaluate_scripted_episode(
    model_name: str,
    training_policies: tuple[str, ...],
    estimator: GRUBeliefEstimator,
    condition: str,
    deception_steps: int,
    episode: int,
    seed: int,
    goal: str,
    step_commit: int,
) -> dict[str, Any]:
    return evaluate_deceiver_episode(
        model_name=model_name,
        training_policies=training_policies,
        estimator=estimator,
        deceiver=ScriptedDeception(deception_steps=deception_steps),
        condition=condition,
        deception_steps=deception_steps,
        episode=episode,
        seed=seed,
        goal=goal,
        step_commit=step_commit,
    )


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


def run_adaptation(
    output_dir: str | Path = "outputs/adaptation",
    train_episodes_per_policy: int = 40,
    test_episodes: int = 40,
    seen_deception_steps: int = 2,
    unseen_deception_steps: int = 3,
    step_commit: int = 3,
    seed: int = 0,
    epochs: int = 250,
    patience: int = 35,
) -> dict[str, Path]:
    if train_episodes_per_policy < 2 or test_episodes < 1:
        raise ValueError("Use at least two training episodes per policy and one test episode.")
    if seen_deception_steps < 1 or unseen_deception_steps < 1:
        raise ValueError("Deception step counts must be positive.")
    if seen_deception_steps == unseen_deception_steps:
        raise ValueError("Seen and unseen deception parameters must differ.")
    if not 0 <= step_commit < 6:
        raise ValueError("step_commit must be in [0, 6) to precede goal arrival.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    training_data_path = destination / "training_episodes.csv"
    run_experiment(
        episodes_per_policy=train_episodes_per_policy,
        seed=seed,
        output_path=training_data_path,
        estimator=TrajectoryBeliefEstimator(),
        scripted_deception_steps=seen_deception_steps,
    )

    checkpoints = {}
    training_metrics = []
    for model_index, (model_name, policies) in enumerate(_TRAINING_POLICIES.items()):
        checkpoint_path = destination / f"{model_name.lower()}.pt"
        metrics = train(
            data_path=training_data_path,
            output_path=checkpoint_path,
            seed=seed + model_index,
            epochs=epochs,
            patience=patience,
            training_policies=policies,
        )
        checkpoints[model_name] = checkpoint_path
        training_metrics.append(
            {
                "model": model_name,
                "training_policies": "+".join(policies),
                **metrics,
            }
        )

    episode_rows = []
    for model_name, policies in _TRAINING_POLICIES.items():
        estimator = GRUBeliefEstimator.from_checkpoint(checkpoints[model_name])
        for condition_index, (condition, deception_steps) in enumerate(
            (
                ("seen_scripted_deception", seen_deception_steps),
                ("unseen_scripted_deception", unseen_deception_steps),
            )
        ):
            for episode in range(test_episodes):
                episode_seed = (
                    seed
                    + 100_000
                    + condition_index * test_episodes
                    + episode
                )
                goal = _GOALS[episode % len(_GOALS)]
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

    summary_rows = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in episode_rows:
        grouped.setdefault((row["model"], row["condition"]), []).append(row)
    for (model_name, condition), rows in grouped.items():
        policies = _TRAINING_POLICIES[model_name]
        summary_rows.append(
            {
                "model": model_name,
                "training_policies": "+".join(policies),
                "condition": condition,
                "deception_steps": rows[0]["deception_steps"],
                "episodes": len(rows),
                "belief_accuracy": float(np.mean([row["belief_accuracy"] for row in rows])),
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

    episode_results_path = destination / "adaptation_episodes.csv"
    summary_path = destination / "adaptation_summary.csv"
    training_metrics_path = destination / "training_metrics.csv"
    _write_csv(episode_results_path, _EPISODE_FIELDS, episode_rows)
    _write_csv(summary_path, _SUMMARY_FIELDS, summary_rows)
    _write_csv(
        training_metrics_path,
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
        "training_metrics": training_metrics_path,
        "episodes": episode_results_path,
        "summary": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/adaptation"))
    parser.add_argument("--train-episodes", type=int, default=40)
    parser.add_argument("--test-episodes", type=int, default=40)
    parser.add_argument("--seen-deception-steps", type=int, default=2)
    parser.add_argument("--unseen-deception-steps", type=int, default=3)
    parser.add_argument("--step-commit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    args = parser.parse_args()
    outputs = run_adaptation(
        output_dir=args.output_dir,
        train_episodes_per_policy=args.train_episodes,
        test_episodes=args.test_episodes,
        seen_deception_steps=args.seen_deception_steps,
        unseen_deception_steps=args.unseen_deception_steps,
        step_commit=args.step_commit,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
    )
    print(f"Wrote adaptation summary to {outputs['summary']}")


if __name__ == "__main__":
    main()
