"""Factorial exposure/diversity experiment for frozen-environment GRU agents."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from deception_rl.agents import BeliefBasedDefender, HonestShortestPath, RandomPolicy, ScriptedDeception
from deception_rl.beliefs import GRUBeliefEstimator, TrajectoryBeliefEstimator
from deception_rl.envs.gridworld import GridWorldParallelEnv
from experiments.deception_families import (
    DelayedDeception,
    DetourDeception,
    MultiSwitchDeception,
)
from experiments.run_adaptation import evaluate_deceiver_episode
from experiments.run_baseline import _visible_trajectory
from train_intent_model import train

_MODELS = {
    "C1": {
        "scripted_per_variant": "low",
        "families": ("scripted_deception",),
    },
    "C2": {
        "scripted_per_variant": "high",
        "families": ("scripted_deception",),
    },
    "C3": {
        "scripted_per_variant": "diverse",
        "families": (
            "scripted_deception",
            "delayed_deception",
            "detour_deception",
            "multi_switch_deception",
        ),
    },
}
_SCRIPTED_STEPS = (1, 2, 3, 4)
_TEST_GOALS = ("A", "B")
_SUMMARY_FIELDS = [
    "model",
    "training_seed",
    "training_families",
    "training_family_count",
    "deception_training_episodes",
    "total_training_episodes",
    "deception_steps",
    "test_scenario",
    "test_family",
    "test_parameters",
    "test_episodes",
    "belief_accuracy",
    "commitment_accuracy",
    "deception_success_rate",
    "mean_true_goal_probability_at_commitment",
]
_EPISODE_FIELDS = [
    "model",
    "training_seed",
    "training_policies",
    "training_families",
    "training_family_count",
    "deception_training_episodes",
    "total_training_episodes",
    "test_scenario",
    "condition",
    "test_family",
    "test_parameters",
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
_DIVERSITY_FIELDS = [
    "model",
    "training_family_count",
    "deception_training_episodes",
    "total_training_episodes",
    "test_scenarios",
    "training_repeats",
    "belief_accuracy",
    "robustness_all_scenarios",
    "robustness_scripted_extrapolation",
    "robustness_new_families",
    "robustness_commitment_accuracy",
    "robustness_std_across_training_seeds",
    "robustness_scripted_std_across_seeds",
    "robustness_new_families_std_across_seeds",
    "deception_success_rate",
]


@dataclass(frozen=True)
class TrainingSpec:
    policy_name: str
    family_name: str
    parameters: str
    episode_count: int
    policy_index: int
    factory: Callable[[], Any]
    deception_steps: int | str = ""


@dataclass(frozen=True)
class TestScenario:
    name: str
    family: str
    parameters: str
    deception_steps: int | str
    factory: Callable[[], Any]


def _training_specs(
    model: str,
    base_episodes: int,
    scripted_low: int,
    scripted_high: int,
) -> list[TrainingSpec]:
    specs = [
        TrainingSpec(
            "honest_shortest_path",
            "honest",
            "shortest_path",
            base_episodes,
            0,
            HonestShortestPath,
        ),
        TrainingSpec(
            "random_policy",
            "random",
            "uniform_actions",
            base_episodes,
            1,
            RandomPolicy,
        ),
    ]
    if model == "C1":
        scripted_count = scripted_low
    elif model == "C2":
        scripted_count = scripted_high
    else:
        scripted_count = scripted_high // len(_SCRIPTED_STEPS)

    for index, steps in enumerate(_SCRIPTED_STEPS):
        specs.append(
            TrainingSpec(
                "scripted_deception",
                "scripted_deception",
                f"deception_steps={steps}",
                scripted_count,
                10 + index,
                lambda steps=steps: ScriptedDeception(steps),
                deception_steps=steps,
            )
        )

    if model == "C3":
        family_specs = [
            TrainingSpec(
                "delayed_deception",
                "delayed_deception",
                "delay_steps=1;feint_steps=1",
                scripted_high,
                20,
                lambda: DelayedDeception(delay_steps=1, feint_steps=1),
            ),
            TrainingSpec(
                "detour_deception",
                "detour_deception",
                "detour_steps=1",
                scripted_high,
                21,
                lambda: DetourDeception(detour_steps=1),
            ),
            TrainingSpec(
                "multi_switch_deception",
                "multi_switch_deception",
                "switches=1",
                scripted_high,
                22,
                lambda: MultiSwitchDeception(switches=1),
            ),
        ]
        specs.extend(family_specs)
    return specs


def _training_episode(
    spec: TrainingSpec,
    episode: int,
    seed: int,
    seed_count: int,
) -> dict[str, Any]:
    seed_group = episode % seed_count
    replicate = episode // seed_count
    episode_seed = seed + seed_group * 100_003 + replicate * 1_009 + spec.policy_index * 1_000_003
    goal = _TEST_GOALS[episode % len(_TEST_GOALS)]
    policy = spec.factory()
    policy.reset(seed=episode_seed)
    env = GridWorldParallelEnv()
    observations, _ = env.reset(seed=episode_seed, options={"goal": goal})
    defender = BeliefBasedDefender(estimator=TrajectoryBeliefEstimator())
    while env.agents:
        defender_action = defender.act(observations["defender"])
        deceiver_action = policy.act(observations["deceiver"])
        observations, *_ = env.step(
            {"deceiver": deceiver_action, "defender": defender_action}
        )
    return {
        "deceiver_policy": spec.policy_name,
        "deception_family": spec.family_name,
        "deception_parameters": spec.parameters,
        "deception_steps": spec.deception_steps,
        "episode": episode,
        "seed_group": seed_group,
        "episode_seed": episode_seed,
        "true_goal": goal,
        "trajectory": _visible_trajectory(observations["defender"]),
    }


def _scenarios() -> list[TestScenario]:
    scenarios = [
        TestScenario(
            name=f"scripted_unseen_steps_{steps}",
            family="scripted_deception",
            parameters=f"deception_steps={steps}",
            deception_steps=steps,
            factory=lambda steps=steps: ScriptedDeception(steps),
        )
        for steps in (5, 6, 7)
    ]
    scenarios.extend(
        [
            TestScenario(
                "delayed_parameter_shift",
                "delayed_deception",
                "delay_steps=2;feint_steps=2",
                "",
                lambda: DelayedDeception(delay_steps=2, feint_steps=2),
            ),
            TestScenario(
                "detour_parameter_shift",
                "detour_deception",
                "detour_steps=2",
                "",
                lambda: DetourDeception(detour_steps=2),
            ),
            TestScenario(
                "multi_switch_parameter_shift",
                "multi_switch_deception",
                "switches=3",
                "",
                lambda: MultiSwitchDeception(switches=3),
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
            for field in ("trajectory", "belief_at_commitment"):
                if field in serialized:
                    serialized[field] = json.dumps(
                        serialized[field], separators=(",", ":"), sort_keys=True
                    )
            writer.writerow(serialized)


def _aggregate_results(
    episode_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in episode_rows:
        grouped.setdefault(
            (row["model"], int(row["training_seed"]), row["test_scenario"]), []
        ).append(row)

    summary_rows = []
    for (model, training_seed, scenario_name), rows in grouped.items():
        summary_rows.append(
            {
                "model": model,
                "training_seed": training_seed,
                "training_families": rows[0]["training_families"],
                "training_family_count": rows[0]["training_family_count"],
                "deception_training_episodes": rows[0]["deception_training_episodes"],
                "total_training_episodes": rows[0]["total_training_episodes"],
                "test_scenario": scenario_name,
                "test_family": rows[0]["test_family"],
                "test_parameters": rows[0]["test_parameters"],
                "deception_steps": rows[0]["deception_steps"],
                "test_episodes": len(rows),
                "belief_accuracy": float(
                    np.mean([float(row["belief_accuracy"]) for row in rows])
                ),
                "commitment_accuracy": float(
                    np.mean([int(row["commitment_accuracy"]) for row in rows])
                ),
                "deception_success_rate": float(
                    np.mean([int(row["deceiver_success"]) for row in rows])
                ),
                "mean_true_goal_probability_at_commitment": float(
                    np.mean(
                        [
                            json.loads(row["belief_at_commitment"])[row["true_goal"]]
                            if isinstance(row["belief_at_commitment"], str)
                            else row["belief_at_commitment"][row["true_goal"]]
                            for row in rows
                        ]
                    )
                ),
            }
        )

    diversity_rows = []
    for model in _MODELS:
        model_rows = [row for row in episode_rows if row["model"] == model]
        model_summaries = [row for row in summary_rows if row["model"] == model]
        scripted_summaries = [
            row for row in model_summaries if row["test_family"] == "scripted_deception"
        ]
        novel_summaries = [
            row for row in model_summaries if row["test_family"] != "scripted_deception"
        ]

        def seed_robustness(rows: list[dict[str, Any]]) -> list[float]:
            return [
                float(
                    np.mean(
                        [
                            float(row["commitment_accuracy"])
                            for row in rows
                            if int(row["training_seed"]) == training_seed
                        ]
                    )
                )
                for training_seed in sorted(
                    {int(row["training_seed"]) for row in rows}
                )
            ]

        all_seed_robustness = seed_robustness(model_summaries)
        scripted_seed_robustness = seed_robustness(scripted_summaries)
        novel_seed_robustness = seed_robustness(novel_summaries)
        diversity_rows.append(
            {
                "model": model,
                "training_family_count": model_rows[0]["training_family_count"],
                "deception_training_episodes": model_rows[0]["deception_training_episodes"],
                "total_training_episodes": model_rows[0]["total_training_episodes"],
                "test_scenarios": len(
                    {row["test_scenario"] for row in model_rows}
                ),
                "training_repeats": len(all_seed_robustness),
                "belief_accuracy": float(
                    np.mean([float(row["belief_accuracy"]) for row in model_rows])
                ),
                "robustness_all_scenarios": float(
                    np.mean([int(row["commitment_accuracy"]) for row in model_rows])
                ),
                "robustness_scripted_extrapolation": float(
                    np.mean(
                        [
                            int(row["commitment_accuracy"])
                            for row in model_rows
                            if row["test_family"] == "scripted_deception"
                        ]
                    )
                ),
                "robustness_new_families": float(
                    np.mean(
                        [
                            int(row["commitment_accuracy"])
                            for row in model_rows
                            if row["test_family"] != "scripted_deception"
                        ]
                    )
                ),
                "robustness_commitment_accuracy": float(
                    np.mean([int(row["commitment_accuracy"]) for row in model_rows])
                ),
                "robustness_std_across_training_seeds": float(
                    np.std(all_seed_robustness, ddof=1)
                    if len(all_seed_robustness) > 1
                    else 0.0
                ),
                "robustness_scripted_std_across_seeds": float(
                    np.std(scripted_seed_robustness, ddof=1)
                    if len(scripted_seed_robustness) > 1
                    else 0.0
                ),
                "robustness_new_families_std_across_seeds": float(
                    np.std(novel_seed_robustness, ddof=1)
                    if len(novel_seed_robustness) > 1
                    else 0.0
                ),
                "deception_success_rate": float(
                    np.mean([int(row["deceiver_success"]) for row in model_rows])
                ),
            }
        )
    return summary_rows, diversity_rows


def refresh_exposure_summaries(
    output_dir: str | Path = "outputs/exposure_diversity",
) -> dict[str, Path]:
    """Recompute summaries from existing episodes without training checkpoints."""
    destination = Path(output_dir)
    episode_path = destination / "exposure_diversity_episodes.csv"
    with episode_path.open(newline="", encoding="utf-8") as csv_file:
        episode_rows = list(csv.DictReader(csv_file))
    summary_rows, diversity_rows = _aggregate_results(episode_rows)
    summary_path = destination / "exposure_diversity_summary.csv"
    diversity_path = destination / "robustness_vs_training_diversity.csv"
    _write_csv(summary_path, _SUMMARY_FIELDS, summary_rows)
    _write_csv(diversity_path, _DIVERSITY_FIELDS, diversity_rows)
    return {"summary": summary_path, "diversity": diversity_path}


def run_exposure_diversity(
    output_dir: str | Path = "outputs/exposure_diversity",
    base_episodes_per_policy: int = 500,
    scripted_low_per_variant: int = 50,
    scripted_high_per_variant: int = 500,
    seed_count: int = 10,
    test_episodes_per_scenario: int = 200,
    step_commit: int = 5,
    seed: int = 0,
    epochs: int = 120,
    patience: int = 20,
    batch_size: int = 128,
    device: str = "auto",
    training_seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, Path]:
    counts = (
        base_episodes_per_policy,
        scripted_low_per_variant,
        scripted_high_per_variant,
        test_episodes_per_scenario,
    )
    if any(count < 1 for count in counts):
        raise ValueError("All episode counts must be positive.")
    if seed_count < 1 or any(count % seed_count for count in counts[:3]):
        raise ValueError("Training episode counts must be divisible by seed_count.")
    if test_episodes_per_scenario < 2:
        raise ValueError("Use at least two test episodes to balance goals.")
    if not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("training_seeds must be a non-empty sequence of unique seeds.")
    if scripted_high_per_variant % len(_SCRIPTED_STEPS):
        raise ValueError("scripted_high_per_variant must divide evenly across four steps.")
    if not 0 <= step_commit < 6:
        raise ValueError("step_commit must be in [0, 6) to precede target arrival.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset_paths: dict[str, Path] = {}
    checkpoint_paths: dict[str, list[Path]] = {}
    training_metrics = []
    exposure_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    model_specs = {
        model: _training_specs(
            model,
            base_episodes_per_policy,
            scripted_low_per_variant,
            scripted_high_per_variant,
        )
        for model in _MODELS
    }

    for model, specs in model_specs.items():
        dataset_path = destination / f"{model.lower()}_training_episodes.csv"
        dataset_rows = [
            _training_episode(spec, episode, seed, seed_count)
            for spec in specs
            for episode in range(spec.episode_count)
        ]
        _write_csv(
            dataset_path,
            [
                "deceiver_policy",
                "deception_family",
                "deception_parameters",
                "deception_steps",
                "episode",
                "seed_group",
                "episode_seed",
                "true_goal",
                "trajectory",
            ],
            dataset_rows,
        )
        dataset_paths[model] = dataset_path
        training_families = sorted(
            {
                spec.family_name
                for spec in specs
                if spec.family_name not in ("honest", "random")
            }
        )
        exposure_counts[model] = sum(
            spec.episode_count
            for spec in specs
            if spec.family_name not in ("honest", "random")
        )
        total_counts[model] = len(dataset_rows)
        family_counts[model] = len(training_families)
        policy_names = tuple(sorted({spec.policy_name for spec in specs}))
        checkpoint_paths[model] = []
        for training_seed in training_seeds:
            checkpoint_path = destination / f"{model.lower()}_seed{training_seed}.pt"
            metrics = train(
                data_path=dataset_path,
                output_path=checkpoint_path,
                seed=seed + training_seed,
                epochs=epochs,
                patience=patience,
                batch_size=batch_size,
                training_policies=policy_names,
                device=device,
            )
            checkpoint_paths[model].append(checkpoint_path)
            training_metrics.append(
                {
                    "model": model,
                    "training_seed": training_seed,
                    "training_families": "+".join(training_families),
                    "training_family_count": family_counts[model],
                    "deception_training_episodes": exposure_counts[model],
                    "total_training_episodes": total_counts[model],
                    **metrics,
                }
            )

    episode_rows = []
    for model, policies in _MODELS.items():
        for training_seed, checkpoint_path in zip(
            training_seeds, checkpoint_paths[model]
        ):
            estimator = GRUBeliefEstimator.from_checkpoint(checkpoint_path)
            for scenario_index, scenario in enumerate(_scenarios()):
                for episode in range(test_episodes_per_scenario):
                    episode_seed = (
                        seed + 90_000_001 + scenario_index * 1_000_003 + episode
                    )
                    base_row = evaluate_deceiver_episode(
                        model_name=model,
                        training_policies=policies["families"],
                        estimator=estimator,
                        deceiver=scenario.factory(),
                        condition=scenario.name,
                        deception_steps=scenario.deception_steps,
                        episode=episode,
                        seed=episode_seed,
                        goal=_TEST_GOALS[episode % len(_TEST_GOALS)],
                        step_commit=step_commit,
                    )
                    episode_rows.append(
                        {
                            **base_row,
                            "training_seed": training_seed,
                            "training_families": "+".join(policies["families"]),
                            "training_family_count": family_counts[model],
                            "deception_training_episodes": exposure_counts[model],
                            "total_training_episodes": total_counts[model],
                            "test_scenario": scenario.name,
                            "test_family": scenario.family,
                            "test_parameters": scenario.parameters,
                        }
                    )

    episode_path = destination / "exposure_diversity_episodes.csv"
    metrics_path = destination / "training_metrics.csv"
    _write_csv(episode_path, _EPISODE_FIELDS, episode_rows)
    _write_csv(
        metrics_path,
        [
            "model",
            "training_seed",
            "training_policies",
            "training_families",
            "training_family_count",
            "deception_training_episodes",
            "total_training_episodes",
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
    refreshed = refresh_exposure_summaries(destination)
    summary_path = refreshed["summary"]
    diversity_path = refreshed["diversity"]
    return {
        "training_data_c1": dataset_paths["C1"],
        "training_data_c2": dataset_paths["C2"],
        "training_data_c3": dataset_paths["C3"],
        "model_c1": checkpoint_paths["C1"][0],
        "model_c2": checkpoint_paths["C2"][0],
        "model_c3": checkpoint_paths["C3"][0],
        "training_metrics": metrics_path,
        "episodes": episode_path,
        "summary": summary_path,
        "diversity": diversity_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exposure_diversity"))
    parser.add_argument("--base-episodes", type=int, default=500)
    parser.add_argument("--scripted-low", type=int, default=50)
    parser.add_argument("--scripted-high", type=int, default=500)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=200)
    parser.add_argument("--step-commit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Refresh summaries from existing episode CSVs without retraining.",
    )
    args = parser.parse_args()
    if args.summarize_only:
        outputs = refresh_exposure_summaries(args.output_dir)
        print(f"Refreshed exposure/diversity summary to {outputs['diversity']}")
        return
    outputs = run_exposure_diversity(
        output_dir=args.output_dir,
        base_episodes_per_policy=args.base_episodes,
        scripted_low_per_variant=args.scripted_low,
        scripted_high_per_variant=args.scripted_high,
        seed_count=args.seed_count,
        test_episodes_per_scenario=args.test_episodes,
        step_commit=args.step_commit,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
        training_seeds=tuple(args.training_seeds),
    )
    print(f"Wrote exposure/diversity summary to {outputs['diversity']}")


if __name__ == "__main__":
    main()
