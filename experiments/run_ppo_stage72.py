"""Belief-observation and reward-curriculum PPO Deceiver experiments."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from deception_rl.agents import HonestShortestPath, RandomPolicy, ScriptedDeception
from deception_rl.beliefs import GRUBeliefEstimator
from experiments.ppo_deceiver_env import PPODeceiverEnv, encode_deceiver_observation
from experiments.run_ppo_deceiver import _EPISODE_FIELDS, _evaluate_episode, _write_csv
from experiments.plot_ppo_stage72 import plot_stage72

EXPERIMENTS = {
    "Experiment1_PPO_no_belief": {"belief": False, "curriculum": False},
    "Experiment2_PPO_with_belief": {"belief": True, "curriculum": False},
    "Experiment3_belief_curriculum": {"belief": True, "curriculum": True},
}
_SUMMARY_FIELDS = [
    "experiment",
    "observation_mode",
    "curriculum",
    "policy",
    "episodes",
    "deceiver_success_rate",
    "commitment_accuracy",
    "deception_success_rate",
    "belief_accuracy",
    "mean_belief_shift_true_goal",
    "mean_deception_cost",
    "mean_path_efficiency",
    "mean_episode_return",
    "success_rate_delta_vs_random",
    "success_rate_delta_vs_scripted",
]
_CURVE_FIELDS = [
    "experiment",
    "observation_mode",
    "curriculum",
    "timestep",
    "episode_index",
    "phase",
    "episode_reward",
    "task_reward",
    "deception_reward",
    "movement_cost",
    "detour_cost",
    "success",
    "rolling_success_rate",
]
_PHASE_FIELDS = [
    "experiment",
    "phase",
    "start_timestep",
    "end_timestep",
    "phase_timesteps",
]
_TRAINING_METRICS_FIELDS = [
    "experiment",
    "observation_mode",
    "curriculum",
    "timesteps_requested",
    "timesteps_trained",
    "evaluation_episodes_per_policy",
    "checkpoint",
    "tensorboard_log_dir",
]


class Stage72Policy:
    name = "ppo_learned_deceiver"

    def __init__(self, model: PPO, belief_observation: bool) -> None:
        self.model = model
        self.belief_observation = belief_observation
        self._belief = {"A": 0.5, "B": 0.5}
        self._committed = False

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._belief = {"A": 0.5, "B": 0.5}
        self._committed = False

    def set_defender_context(
        self, belief: dict[str, float], committed: bool
    ) -> None:
        self._belief = belief.copy()
        self._committed = committed

    def act(self, observation: dict[str, np.ndarray | int]) -> int:
        encoded = encode_deceiver_observation(observation)
        if self.belief_observation:
            probabilities = np.array(
                [self._belief["A"], self._belief["B"]], dtype=np.float32
            )
            entropy = float(
                -(
                    probabilities
                    * np.log2(np.clip(probabilities, 1e-8, 1.0))
                ).sum()
            )
            encoded = np.concatenate(
                (
                    encoded,
                    probabilities,
                    np.array([np.clip(entropy, 0.0, 1.0), float(self._committed)]),
                )
            ).astype(np.float32)
        action, _ = self.model.predict(encoded, deterministic=True)
        return int(action)


class Stage72EpisodeCallback(BaseCallback):
    def __init__(
        self,
        experiment: str,
        observation_mode: str,
        curriculum: bool,
        total_timesteps: int,
    ) -> None:
        super().__init__(verbose=0)
        self.experiment = experiment
        self.observation_mode = observation_mode
        self.curriculum = curriculum
        self.total_timesteps = total_timesteps
        self.curve_rows: list[dict[str, Any]] = []
        self.phase_rows: list[dict[str, Any]] = []
        self._success_window: deque[int] = deque(maxlen=100)

    def _on_training_start(self) -> None:
        env_count = self.training_env.num_envs
        self._episode_rewards = np.zeros(env_count, dtype=np.float64)
        self._components = {
            name: np.zeros(env_count, dtype=np.float64)
            for name in (
                "task_reward",
                "deception_reward",
                "movement_cost",
                "detour_cost",
            )
        }
        self._episode_indices = np.zeros(env_count, dtype=np.int64)
        self._phase = "task_only"
        self._phase_start = 0
        self._phase_schedule = (
            (
                max(1, self.total_timesteps // 3),
                "task_plus_deception",
            ),
            (
                max(2, (2 * self.total_timesteps) // 3),
                "task_deception_cost",
            ),
        )
        if self.curriculum:
            self.training_env.env_method("set_reward_mode", self._phase)

    def _on_step(self) -> bool:
        if self.curriculum:
            while (
                self._phase_schedule
                and self.num_timesteps >= self._phase_schedule[0][0]
            ):
                boundary, next_phase = self._phase_schedule[0]
                self._record_phase(self._phase, self._phase_start, boundary)
                self._phase_schedule = self._phase_schedule[1:]
                self._phase = next_phase
                self._phase_start = boundary
                self.training_env.env_method("set_reward_mode", self._phase)

        rewards = np.asarray(self.locals["rewards"], dtype=np.float64).reshape(-1)
        dones = np.asarray(self.locals["dones"], dtype=bool).reshape(-1)
        infos = self.locals["infos"]
        self._episode_rewards += rewards
        for index, info in enumerate(infos):
            for component in self._components:
                self._components[component][index] += float(
                    info.get(component, 0.0)
                )
            if not dones[index]:
                continue
            success = int(info.get("episode_metrics", {}).get("deceiver_success", 0))
            self._success_window.append(success)
            self._episode_indices[index] += 1
            row = {
                "experiment": self.experiment,
                "observation_mode": self.observation_mode,
                "curriculum": int(self.curriculum),
                "timestep": self.num_timesteps,
                "episode_index": int(self._episode_indices[index]),
                "phase": info.get("reward_mode", self._phase),
                "episode_reward": float(self._episode_rewards[index]),
                "task_reward": float(self._components["task_reward"][index]),
                "deception_reward": float(
                    self._components["deception_reward"][index]
                ),
                "movement_cost": float(self._components["movement_cost"][index]),
                "detour_cost": float(self._components["detour_cost"][index]),
                "success": success,
                "rolling_success_rate": float(np.mean(self._success_window)),
            }
            self.curve_rows.append(row)
            for metric in (
                "episode_reward",
                "task_reward",
                "deception_reward",
                "movement_cost",
                "detour_cost",
                "rolling_success_rate",
            ):
                self.logger.record(f"stage7_2/{metric}", row[metric])
            self.logger.dump(self.num_timesteps)
            self._episode_rewards[index] = 0.0
            for values in self._components.values():
                values[index] = 0.0
        rollout_size = self.model.n_steps * self.training_env.num_envs
        return (
            self.num_timesteps < self.total_timesteps
            or self.num_timesteps % rollout_size == 0
        )

    def _record_phase(self, phase: str, start: int, end: int) -> None:
        self.phase_rows.append(
            {
                "experiment": self.experiment,
                "phase": phase,
                "start_timestep": start,
                "end_timestep": end,
                "phase_timesteps": max(0, end - start),
            }
        )

    def _on_training_end(self) -> None:
        if self.curriculum:
            self._record_phase(
                self._phase,
                self._phase_start,
                self.num_timesteps,
            )


def _evaluate_experiment(
    experiment: str,
    model: PPO,
    belief_observation: bool,
    estimator: GRUBeliefEstimator,
    evaluation_episodes: int,
    seed: int,
    step_commit: int,
) -> list[dict[str, Any]]:
    policies = [
        HonestShortestPath(),
        RandomPolicy(seed=seed),
        ScriptedDeception(deception_steps=2),
        Stage72Policy(model, belief_observation),
    ]
    names = (
        "honest_shortest_path",
        "random_policy",
        "scripted_deception",
        "ppo_learned_deceiver",
    )
    rows = []
    for policy_index, (name, policy) in enumerate(zip(names, policies)):
        for episode in range(evaluation_episodes):
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
            row["experiment"] = experiment
            row["observation_mode"] = (
                "with_belief" if belief_observation else "no_belief"
            )
            row["curriculum"] = int(experiment.endswith("curriculum"))
            row["episode_return"] = float(
                row["task_reward"]
                + row["deception_reward"]
                + row["movement_cost"]
                + row["detour_cost"]
            )
            rows.append(row)
    return rows


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["experiment"], row["policy"]), []).append(row)
    summary_rows = []
    random_success: dict[str, float] = {}
    scripted_success: dict[str, float] = {}
    for (experiment, policy), policy_rows in groups.items():
        success = float(np.mean([row["deceiver_success"] for row in policy_rows]))
        if policy == "random_policy":
            random_success[experiment] = success
        elif policy == "scripted_deception":
            scripted_success[experiment] = success
        summary_rows.append(
            {
                "experiment": experiment,
                "observation_mode": policy_rows[0]["observation_mode"],
                "curriculum": policy_rows[0]["curriculum"],
                "policy": policy,
                "episodes": len(policy_rows),
                "deceiver_success_rate": success,
                "commitment_accuracy": float(
                    np.mean([row["commitment_accuracy"] for row in policy_rows])
                ),
                "deception_success_rate": float(
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
                "mean_episode_return": float(
                    np.mean([row["episode_return"] for row in policy_rows])
                ),
                "success_rate_delta_vs_random": "",
                "success_rate_delta_vs_scripted": "",
            }
        )
    for row in summary_rows:
        row["success_rate_delta_vs_random"] = (
            row["deceiver_success_rate"] - random_success[row["experiment"]]
        )
        row["success_rate_delta_vs_scripted"] = (
            row["deceiver_success_rate"] - scripted_success[row["experiment"]]
        )
    return summary_rows


def run_stage72(
    output_dir: str | Path = "outputs/ppo_stage72",
    defender_checkpoint: str | Path = "outputs/data_scaling/modela.pt",
    total_timesteps: int = 50_000,
    evaluation_episodes: int = 20,
    seed: int = 0,
    step_commit: int = 5,
    device: str = "cpu",
) -> dict[str, Path]:
    if total_timesteps < 128:
        raise ValueError("Smoke runs require at least 128 timesteps.")
    if evaluation_episodes < 2:
        raise ValueError("Use at least two evaluation episodes per policy.")
    checkpoint = Path(defender_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Fixed Model A checkpoint not found: {checkpoint}")

    destination = Path(output_dir)
    checkpoint_dir = destination / "checkpoints"
    tensorboard_dir = destination / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    estimator = GRUBeliefEstimator.from_checkpoint(checkpoint)
    curve_rows = []
    phase_rows = []
    evaluation_rows = []
    training_metrics = []

    for experiment, options in EXPERIMENTS.items():
        observation_mode = "with_belief" if options["belief"] else "no_belief"
        env = PPODeceiverEnv(
            estimator,
            step_commit=step_commit,
            task_reward_scale=1.0,
            misdirection_reward=0.25,
            movement_cost=0.01,
            detour_cost=0.05,
            belief_observation=options["belief"],
        )
        check_env(env, warn=True)
        callback = Stage72EpisodeCallback(
            experiment,
            observation_mode,
            options["curriculum"],
            total_timesteps,
        )
        experiment_seed = seed
        torch.manual_seed(experiment_seed)
        model = PPO(
            "MlpPolicy",
            env,
            seed=experiment_seed,
            n_steps=256 if total_timesteps >= 256 else 128,
            batch_size=128 if total_timesteps >= 256 else 64,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.02,
            verbose=0,
            device=device,
            tensorboard_log=str(tensorboard_dir / experiment),
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=experiment,
        )
        model_path = checkpoint_dir / f"{experiment}.zip"
        model.save(model_path)
        curve_rows.extend(callback.curve_rows)
        phase_rows.extend(callback.phase_rows)
        evaluation_rows.extend(
            _evaluate_experiment(
                experiment,
                model,
                options["belief"],
                estimator,
                evaluation_episodes,
                seed + 100_000,
                step_commit,
            )
        )
        training_metrics.append(
            {
                "experiment": experiment,
                "observation_mode": observation_mode,
                "curriculum": int(options["curriculum"]),
                "timesteps_requested": total_timesteps,
                "timesteps_trained": model.num_timesteps,
                "evaluation_episodes_per_policy": evaluation_episodes,
                "checkpoint": str(model_path),
                "tensorboard_log_dir": str(tensorboard_dir / experiment),
            }
        )

    learning_path = destination / "learning_curves.csv"
    episodes_path = destination / "evaluation_episodes.csv"
    summary_path = destination / "evaluation_summary.csv"
    phase_path = destination / "curriculum_phases.csv"
    metrics_path = destination / "training_metrics.csv"
    _write_csv(learning_path, _CURVE_FIELDS, curve_rows)
    _write_csv(
        episodes_path,
        ["experiment", "observation_mode", "curriculum", "episode_return", *_EPISODE_FIELDS],
        evaluation_rows,
    )
    _write_csv(summary_path, _SUMMARY_FIELDS, _summarize(evaluation_rows))
    _write_csv(phase_path, _PHASE_FIELDS, phase_rows)
    _write_csv(metrics_path, _TRAINING_METRICS_FIELDS, training_metrics)
    from experiments.plot_ppo_stage72 import plot_stage72

    figures = plot_stage72(learning_path, episodes_path, destination)
    return {
        "checkpoints": checkpoint_dir,
        "tensorboard": tensorboard_dir,
        "learning_curves": learning_path,
        "episodes": episodes_path,
        "summary": summary_path,
        "phases": phase_path,
        "training_metrics": metrics_path,
        "learning_curve_figure": figures[0],
        "belief_ablation_figure": figures[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ppo_stage72"))
    parser.add_argument("--defender-checkpoint", type=Path, default=Path("outputs/data_scaling/modela.pt"))
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--step-commit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    outputs = run_stage72(
        output_dir=args.output_dir,
        defender_checkpoint=args.defender_checkpoint,
        total_timesteps=args.timesteps,
        evaluation_episodes=args.evaluation_episodes,
        seed=args.seed,
        step_commit=args.step_commit,
        device=args.device,
    )
    print(f"Wrote Stage7.2 summary to {outputs['summary']}")


if __name__ == "__main__":
    main()
