"""Compare task, deception-shaping, and movement-cost PPO reward ablations."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from deception_rl.agents import HonestShortestPath, RandomPolicy, ScriptedDeception
from deception_rl.beliefs import GRUBeliefEstimator
from experiments.ppo_deceiver_env import PPODeceiverEnv
from experiments.run_ppo_deceiver import (
    LearnedDeceiver,
    _EPISODE_FIELDS,
    _evaluate_episode,
    _write_csv,
)

_ABLATION_FIELDS = [
    "ablation",
    "policy",
    "episodes",
    "deceiver_success_rate",
    "commitment_accuracy",
    "misdirection_rate",
    "belief_accuracy",
    "mean_belief_shift_true_goal",
    "mean_deception_cost",
    "mean_path_efficiency",
    "success_rate_delta_vs_random",
    "success_rate_delta_vs_scripted",
]
_LEARNING_CURVE_FIELDS = [
    "ablation",
    "timesteps",
    "episode_index",
    "episode_reward",
    "task_reward",
    "deception_reward",
    "movement_cost",
    "detour_cost",
    "success",
    "rolling_success_rate",
]
_TRAINING_METRICS_FIELDS = [
    "ablation",
    "timesteps_requested",
    "timesteps_trained",
    "evaluation_episodes_per_policy",
    "checkpoint",
    "tensorboard_log_dir",
]


@dataclass(frozen=True)
class RewardAblation:
    name: str
    deception_reward: float
    movement_cost: float
    detour_cost: float


def reward_ablation_configs() -> tuple[RewardAblation, ...]:
    return (
        RewardAblation("A_task_only", 0.0, 0.0, 0.0),
        RewardAblation("B_task_plus_deception", 0.25, 0.0, 0.0),
        RewardAblation("C_task_deception_cost", 0.25, 0.01, 0.05),
    )


class RewardTensorboardCallback(BaseCallback):
    """Record per-episode reward components and rolling task success."""

    def __init__(self, ablation: str, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.ablation = ablation
        self.rows: list[dict[str, Any]] = []
        self._recent_successes: deque[int] = deque(maxlen=100)

    def _on_training_start(self) -> None:
        env_count = self.training_env.num_envs
        self._episode_totals = np.zeros(env_count, dtype=np.float64)
        self._component_totals = {
            name: np.zeros(env_count, dtype=np.float64)
            for name in (
                "task_reward",
                "deception_reward",
                "movement_cost",
                "detour_cost",
            )
        }
        self._episode_indices = np.zeros(env_count, dtype=np.int64)

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64).reshape(-1)
        dones = np.asarray(self.locals["dones"], dtype=bool).reshape(-1)
        infos = self.locals["infos"]
        self._episode_totals += rewards
        for index, info in enumerate(infos):
            for component in self._component_totals:
                self._component_totals[component][index] += float(
                    info.get(component, 0.0)
                )
            if not dones[index]:
                continue

            episode_metrics = info.get("episode_metrics", {})
            success = int(episode_metrics.get("deceiver_success", 0))
            self._recent_successes.append(success)
            self._episode_indices[index] += 1
            rolling_success = float(np.mean(self._recent_successes))
            row = {
                "ablation": self.ablation,
                "timesteps": self.num_timesteps,
                "episode_index": int(self._episode_indices[index]),
                "episode_reward": float(self._episode_totals[index]),
                "task_reward": float(self._component_totals["task_reward"][index]),
                "deception_reward": float(
                    self._component_totals["deception_reward"][index]
                ),
                "movement_cost": float(
                    self._component_totals["movement_cost"][index]
                ),
                "detour_cost": float(self._component_totals["detour_cost"][index]),
                "success": success,
                "rolling_success_rate": rolling_success,
            }
            self.rows.append(row)
            for metric in (
                "episode_reward",
                "task_reward",
                "deception_reward",
                "movement_cost",
                "detour_cost",
                "rolling_success_rate",
            ):
                self.logger.record(f"stage7_1/{metric}", row[metric])
            self.logger.dump(self.num_timesteps)
            self._episode_totals[index] = 0.0
            for values in self._component_totals.values():
                values[index] = 0.0
        return True


def _evaluate_ablation(
    model: PPO,
    estimator: GRUBeliefEstimator,
    ablation: str,
    episodes_per_policy: int,
    seed: int,
    step_commit: int,
) -> list[dict[str, Any]]:
    policies = [
        HonestShortestPath(),
        RandomPolicy(seed=seed),
        ScriptedDeception(deception_steps=2),
        LearnedDeceiver(model),
    ]
    names = (
        "honest_shortest_path",
        "random_policy",
        "scripted_deception",
        "ppo_learned_deceiver",
    )
    rows = []
    for policy_index, (policy_name, policy) in enumerate(zip(names, policies)):
        for episode in range(episodes_per_policy):
            episode_seed = seed + 100_003 + policy_index * 10_007 + episode
            goal = "A" if episode % 2 == 0 else "B"
            row = _evaluate_episode(
                policy,
                episode,
                episode_seed,
                goal,
                estimator,
                step_commit,
            )
            row["ablation"] = ablation
            rows.append(row)
    return rows


def _summarize_evaluation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["ablation"], row["policy"]), []).append(row)
    summaries = []
    by_ablation: dict[str, dict[str, float]] = {}
    for (ablation, policy), policy_rows in groups.items():
        summaries.append(
            {
                "ablation": ablation,
                "policy": policy,
                "episodes": len(policy_rows),
                "deceiver_success_rate": float(
                    np.mean([row["deceiver_success"] for row in policy_rows])
                ),
                "commitment_accuracy": float(
                    np.mean([row["commitment_accuracy"] for row in policy_rows])
                ),
                "misdirection_rate": float(
                    np.mean([row["misdirection_rate"] for row in policy_rows])
                ),
                "belief_accuracy": float(
                    np.mean([row["belief_accuracy"] for row in policy_rows])
                ),
                "mean_belief_shift_true_goal": float(
                    np.mean([row["belief_shift_true_goal"] for row in policy_rows])
                ),
                "mean_deception_cost": float(
                    np.mean([row["deception_cost"] for row in policy_rows])
                ),
                "mean_path_efficiency": float(
                    np.mean([row["path_efficiency"] for row in policy_rows])
                ),
                "success_rate_delta_vs_random": "",
                "success_rate_delta_vs_scripted": "",
            }
        )
        by_ablation.setdefault(ablation, {})[policy] = summaries[-1][
            "deceiver_success_rate"
        ]
    for row in summaries:
        baselines = by_ablation[row["ablation"]]
        row["success_rate_delta_vs_random"] = (
            row["deceiver_success_rate"] - baselines["random_policy"]
        )
        row["success_rate_delta_vs_scripted"] = (
            row["deceiver_success_rate"] - baselines["scripted_deception"]
        )
    return summaries


def run_reward_ablation(
    output_dir: str | Path = "outputs/ppo_reward_ablation",
    defender_checkpoint: str | Path = "outputs/data_scaling/modela.pt",
    total_timesteps: int = 50_000,
    evaluation_episodes_per_policy: int = 20,
    seed: int = 0,
    step_commit: int = 5,
    device: str = "cpu",
) -> dict[str, Path]:
    if total_timesteps < 50_000:
        raise ValueError("Each reward ablation must train for at least 50,000 timesteps.")
    if evaluation_episodes_per_policy < 2:
        raise ValueError("Use at least two evaluation episodes per policy.")
    checkpoint = Path(defender_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Fixed Model A checkpoint not found: {checkpoint}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    all_curve_rows = []
    all_eval_rows = []
    training_rows = []

    for ablation_index, config in enumerate(reward_ablation_configs()):
        config_dir = destination / config.name
        config_dir.mkdir(parents=True, exist_ok=True)
        estimator = GRUBeliefEstimator.from_checkpoint(checkpoint)
        env = PPODeceiverEnv(
            estimator,
            step_commit=step_commit,
            task_reward_scale=1.0,
            misdirection_reward=config.deception_reward,
            movement_cost=config.movement_cost,
            detour_cost=config.detour_cost,
        )
        check_env(env, warn=True)
        config_seed = seed + ablation_index
        torch.manual_seed(config_seed)
        callback = RewardTensorboardCallback(config.name)
        model = PPO(
            "MlpPolicy",
            env,
            seed=config_seed,
            n_steps=256,
            batch_size=128,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.02,
            verbose=0,
            device=device,
            tensorboard_log=str(config_dir / "tensorboard"),
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=config.name,
        )
        model_path = config_dir / "ppo_deceiver.zip"
        model.save(model_path)
        all_curve_rows.extend(callback.rows)
        all_eval_rows.extend(
            _evaluate_ablation(
                model,
                estimator,
                config.name,
                evaluation_episodes_per_policy,
                seed + 100_000,
                step_commit,
            )
        )
        training_rows.append(
            {
                "ablation": config.name,
                "timesteps_requested": total_timesteps,
                "timesteps_trained": model.num_timesteps,
                "evaluation_episodes_per_policy": evaluation_episodes_per_policy,
                "checkpoint": str(model_path),
                "tensorboard_log_dir": str(config_dir / "tensorboard"),
            }
        )

    learning_curve_path = destination / "learning_curves.csv"
    episode_path = destination / "evaluation_episodes.csv"
    summary_path = destination / "evaluation_summary.csv"
    action_path = destination / "action_statistics.csv"
    training_path = destination / "training_metrics.csv"
    _write_csv(learning_curve_path, _LEARNING_CURVE_FIELDS, all_curve_rows)
    _write_csv(episode_path, ["ablation", *_EPISODE_FIELDS], all_eval_rows)
    _write_csv(summary_path, _ABLATION_FIELDS, _summarize_evaluation(all_eval_rows))
    action_rows = []
    for config in reward_ablation_configs():
        config_rows = [row for row in all_eval_rows if row["ablation"] == config.name]
        for policy in (
            "honest_shortest_path",
            "random_policy",
            "scripted_deception",
            "ppo_learned_deceiver",
        ):
            policy_rows = [row for row in config_rows if row["policy"] == policy]
            counts = Counter()
            for row in policy_rows:
                counts.update(row["action_sequence"])
            action_row: dict[str, Any] = {
                "ablation": config.name,
                "policy": policy,
                "episodes": len(policy_rows),
            }
            for action_name in ("up", "down", "left", "right", "stay"):
                action_row[f"mean_{action_name}_count"] = float(
                    np.mean(
                        [row["action_sequence"].count(action_name) for row in policy_rows]
                    )
                )
            for metric in (
                "direction_switches",
                "goal_regressions",
                "false_goal_progress_steps",
                "deception_cost",
                "path_efficiency",
            ):
                column = "mean_path_efficiency" if metric == "path_efficiency" else f"mean_{metric}"
                action_row[column] = float(
                    np.mean([row[metric] for row in policy_rows])
                )
            action_rows.append(action_row)
    action_fields = [
        "ablation",
        "policy",
        "episodes",
        "mean_up_count",
        "mean_down_count",
        "mean_left_count",
        "mean_right_count",
        "mean_stay_count",
        "mean_direction_switches",
        "mean_goal_regressions",
        "mean_false_goal_progress_steps",
        "mean_deception_cost",
        "mean_path_efficiency",
    ]
    _write_csv(action_path, action_fields, action_rows)
    _write_csv(training_path, _TRAINING_METRICS_FIELDS, training_rows)

    from experiments.plot_ppo_reward_ablation import plot_reward_ablation

    figures = plot_reward_ablation(learning_curve_path, episode_path, destination)
    return {
        "learning_curves": learning_curve_path,
        "episodes": episode_path,
        "summary": summary_path,
        "actions": action_path,
        "training_metrics": training_path,
        "learning_curve_figure": figures[0],
        "trajectory_figure": figures[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ppo_reward_ablation"))
    parser.add_argument("--defender-checkpoint", type=Path, default=Path("outputs/data_scaling/modela.pt"))
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--step-commit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    outputs = run_reward_ablation(
        output_dir=args.output_dir,
        defender_checkpoint=args.defender_checkpoint,
        total_timesteps=args.timesteps,
        evaluation_episodes_per_policy=args.evaluation_episodes,
        seed=args.seed,
        step_commit=args.step_commit,
        device=args.device,
    )
    print(f"Wrote reward-ablation evaluation to {outputs['summary']}")


if __name__ == "__main__":
    main()
