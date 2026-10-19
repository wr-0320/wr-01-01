"""Evaluate frozen Model A/B on extrapolated and unseen deception families."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from deception_rl.agents import ScriptedDeception
from deception_rl.beliefs import GRUBeliefEstimator
from experiments.deception_families import (
    DelayedDeception,
    DetourDeception,
    MultiSwitchDeception,
)
from experiments.run_adaptation import evaluate_deceiver_episode

_TRAINING_POLICIES = {
    "ModelA": "honest_shortest_path+random_policy",
    "ModelB": "honest_shortest_path+random_policy+scripted_deception",
}
_GOALS = ("A", "B")
_EPISODE_FIELDS = [
    "model",
    "training_policies",
    "condition",
    "deception_family",
    "deception_parameters",
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
    "deception_family",
    "deception_parameters",
    "deception_steps",
    "episodes",
    "belief_accuracy",
    "commitment_accuracy",
    "deception_success_rate",
    "mean_true_goal_probability_at_commitment",
]


@dataclass(frozen=True)
class Scenario:
    family: str
    parameters: str
    deception_steps: int | str
    policy_factory: Callable[[], Any]


def _scenarios() -> list[Scenario]:
    scenarios = [
        Scenario(
            family="scripted_extrapolation",
            parameters=f"deception_steps={steps}",
            deception_steps=steps,
            policy_factory=lambda steps=steps: ScriptedDeception(steps),
        )
        for steps in (6, 7)
    ]
    scenarios.extend(
        [
            Scenario(
                family="delayed_deception",
                parameters="delay_steps=2;feint_steps=2",
                deception_steps="",
                policy_factory=lambda: DelayedDeception(delay_steps=2, feint_steps=2),
            ),
            Scenario(
                family="detour_deception",
                parameters="lateral_detour_steps=1",
                deception_steps="",
                policy_factory=DetourDeception,
            ),
            Scenario(
                family="multi_switch_deception",
                parameters="switches=3",
                deception_steps="",
                policy_factory=lambda: MultiSwitchDeception(switches=3),
            ),
        ]
    )
    return scenarios


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialized = row.copy()
            for field in ("belief_at_commitment", "trajectory"):
                if field in serialized:
                    serialized[field] = json.dumps(
                        serialized[field], separators=(",", ":"), sort_keys=True
                    )
            writer.writerow(serialized)


def run_family_generalization(
    model_a_checkpoint: str | Path = "outputs/data_scaling/modela.pt",
    model_b_checkpoint: str | Path = "outputs/data_scaling/modelb.pt",
    output_dir: str | Path = "outputs/deception_family_generalization",
    episodes_per_scenario: int = 200,
    step_commit: int = 5,
    seed: int = 0,
) -> dict[str, Path]:
    if episodes_per_scenario < 2:
        raise ValueError("Use at least two episodes per scenario to balance goals.")
    if not 0 <= step_commit < 6:
        raise ValueError("step_commit must be in [0, 6) to precede target arrival.")

    checkpoint_paths = {
        "ModelA": Path(model_a_checkpoint),
        "ModelB": Path(model_b_checkpoint),
    }
    for name, path in checkpoint_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Frozen {name} checkpoint not found: {path}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    estimators = {
        model: GRUBeliefEstimator.from_checkpoint(path)
        for model, path in checkpoint_paths.items()
    }
    episode_rows = []
    for model_name, estimator in estimators.items():
        for scenario_index, scenario in enumerate(_scenarios()):
            condition = f"unseen_family_{scenario.family}"
            for episode in range(episodes_per_scenario):
                episode_seed = (
                    seed
                    + 20_000_033
                    + scenario_index * 1_000_003
                    + episode
                )
                base_row = evaluate_deceiver_episode(
                    model_name=model_name,
                    training_policies=tuple(_TRAINING_POLICIES[model_name].split("+")),
                    estimator=estimator,
                    deceiver=scenario.policy_factory(),
                    condition=condition,
                    deception_steps=scenario.deception_steps,
                    episode=episode,
                    seed=episode_seed,
                    goal=_GOALS[episode % len(_GOALS)],
                    step_commit=step_commit,
                )
                base_row.update(
                    {
                        "condition": condition,
                        "deception_family": scenario.family,
                        "deception_parameters": scenario.parameters,
                    }
                )
                episode_rows.append(base_row)

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in episode_rows:
        grouped.setdefault(
            (
                row["model"],
                row["deception_family"],
                row["deception_parameters"],
            ),
            [],
        ).append(row)
    summary_rows = []
    for (model_name, family, parameters), rows in grouped.items():
        summary_rows.append(
            {
                "model": model_name,
                "training_policies": _TRAINING_POLICIES[model_name],
                "condition": rows[0]["condition"],
                "deception_family": family,
                "deception_parameters": parameters,
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

    episode_path = destination / "family_generalization_episodes.csv"
    summary_path = destination / "family_generalization_summary.csv"
    _write_csv(episode_path, _EPISODE_FIELDS, episode_rows)
    _write_csv(summary_path, _SUMMARY_FIELDS, summary_rows)
    return {"episodes": episode_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", type=Path, default=Path("outputs/data_scaling/modela.pt"))
    parser.add_argument("--model-b", type=Path, default=Path("outputs/data_scaling/modelb.pt"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/deception_family_generalization")
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--step-commit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    outputs = run_family_generalization(
        model_a_checkpoint=args.model_a,
        model_b_checkpoint=args.model_b,
        output_dir=args.output_dir,
        episodes_per_scenario=args.episodes,
        step_commit=args.step_commit,
        seed=args.seed,
    )
    print(f"Wrote unseen-family summary to {outputs['summary']}")


if __name__ == "__main__":
    main()
