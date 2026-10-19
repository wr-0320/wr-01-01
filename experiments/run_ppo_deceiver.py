"""Train a small PPO Deceiver against a frozen Model A Defender."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from deception_rl.agents import HonestShortestPath, RandomPolicy, ScriptedDeception
from deception_rl.beliefs import GRUBeliefEstimator
from deception_rl.envs.gridworld import (
    GUARD_A,
    GUARD_B,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
    MOVE_STAY,
    GridWorldParallelEnv,
)
from experiments.ppo_deceiver_env import PPODeceiverEnv, encode_deceiver_observation
from experiments.plot_ppo_trajectories import plot_trajectories

_POLICY_NAMES = (
    "honest_shortest_path",
    "random_policy",
    "scripted_deception",
    "ppo_learned_deceiver",
)
_ACTION_NAMES = {
    MOVE_UP: "up",
    MOVE_DOWN: "down",
    MOVE_LEFT: "left",
    MOVE_RIGHT: "right",
    MOVE_STAY: "stay",
}
_EPISODE_FIELDS = [
    "policy",
    "episode",
    "seed",
    "true_goal",
    "deceiver_success",
    "reached_goal",
    "commitment_accuracy",
    "misdirection_rate",
    "belief_accuracy",
    "belief_at_commitment",
    "belief_shift_true_goal",
    "deception_cost",
    "path_length",
    "shortest_path_length",
    "path_efficiency",
    "up_count",
    "down_count",
    "left_count",
    "right_count",
    "stay_count",
    "direction_switches",
    "goal_regressions",
    "false_goal_progress_steps",
    "action_sequence",
    "trajectory",
    "task_reward",
    "deception_reward",
    "movement_cost",
    "detour_cost",
]
_SUMMARY_FIELDS = [
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
_ACTION_FIELDS = [
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


class LearnedDeceiver:
    name = "ppo_learned_deceiver"

    def __init__(self, model: PPO) -> None:
        self.model = model

    def reset(self, seed: int | None = None) -> None:
        del seed

    def act(self, observation: dict[str, np.ndarray | int]) -> int:
        policy_observation = encode_deceiver_observation(observation)
        action, _ = self.model.predict(policy_observation, deterministic=True)
        return int(action)


def _policy_instances(model: PPO, seed: int) -> list[Any]:
    return [
        HonestShortestPath(),
        RandomPolicy(seed=seed),
        ScriptedDeception(deception_steps=2),
        LearnedDeceiver(model),
    ]


def _movement_statistics(
    actions: list[int],
    positions: list[list[int]],
    true_goal: int,
    target_positions: np.ndarray,
) -> dict[str, int]:
    counts = Counter(actions)
    directional_actions = [action for action in actions if action != MOVE_STAY]
    direction_switches = sum(
        previous != current
        for previous, current in zip(directional_actions, directional_actions[1:])
    )
    true_target = target_positions[true_goal]
    false_target = target_positions[1 - true_goal]
    goal_regressions = 0
    false_goal_progress = 0
    for previous, current in zip(positions, positions[1:]):
        previous_position = np.asarray(previous)
        current_position = np.asarray(current)
        goal_regressions += int(
            np.abs(current_position - true_target).sum()
            > np.abs(previous_position - true_target).sum()
        )
        false_goal_progress += int(
            np.abs(current_position - false_target).sum()
            < np.abs(previous_position - false_target).sum()
        )
    return {
        "up_count": counts[MOVE_UP],
        "down_count": counts[MOVE_DOWN],
        "left_count": counts[MOVE_LEFT],
        "right_count": counts[MOVE_RIGHT],
        "stay_count": counts[MOVE_STAY],
        "direction_switches": direction_switches,
        "goal_regressions": goal_regressions,
        "false_goal_progress_steps": false_goal_progress,
    }


def _evaluate_episode(
    policy: Any,
    episode: int,
    seed: int,
    goal: str,
    estimator: GRUBeliefEstimator,
    step_commit: int,
) -> dict[str, Any]:
    env = GridWorldParallelEnv()
    policy.reset(seed=seed)
    observations, _ = env.reset(seed=seed, options={"goal": goal})
    from deception_rl.agents import BeliefBasedDefender, CommittedDefender

    defender = CommittedDefender(
        step_commit=step_commit,
        defender=BeliefBasedDefender(estimator=estimator),
    )
    positions = [observations["deceiver"]["position"].astype(int).tolist()]
    actions: list[int] = []
    belief_correct = []
    task_reward = 0.0
    deception_reward = 0.0
    movement_cost = 0.0
    detour_cost = 0.0
    commitment_correct: int | None = None
    target_positions = np.asarray(
        observations["deceiver"]["target_positions"], dtype=np.int64
    )
    true_goal = int(observations["deceiver"]["goal"])

    while env.agents:
        current_belief = defender.defender.estimate_belief(observations["defender"])
        belief_correct.append(
            int(
                ("A" if current_belief["A"] >= current_belief["B"] else "B")
                == goal
            )
        )
        previous_commitment = defender.decision
        defender_action = defender.act(observations["defender"])
        if previous_commitment is None and defender.decision is not None:
            commitment_correct = int(defender.decision == true_goal)
        if hasattr(policy, "set_defender_context"):
            policy.set_defender_context(
                current_belief,
                committed=previous_commitment is not None,
            )
        deceiver_action = int(policy.act(observations["deceiver"]))
        actions.append(deceiver_action)
        previous_position = np.asarray(observations["deceiver"]["position"])
        previous_distance = int(
            np.abs(previous_position - target_positions[true_goal]).sum()
        )
        observations, rewards, _, _, _ = env.step(
            {"deceiver": deceiver_action, "defender": defender_action}
        )
        next_position = np.asarray(observations["deceiver"]["position"])
        positions.append(next_position.astype(int).tolist())
        next_distance = int(np.abs(next_position - target_positions[true_goal]).sum())
        moved = int(not np.array_equal(previous_position, next_position))
        task_reward += float(rewards["deceiver"])
        movement_cost -= 0.01 * moved
        detour_cost -= 0.05 * max(0, next_distance - previous_distance)
        if commitment_correct is not None and len(actions) == step_commit + 1:
            deception_reward = float(commitment_correct == 0) * 0.25

    final_belief = defender.defender.estimate_belief(observations["defender"])
    belief_correct.append(
        int(("A" if final_belief["A"] >= final_belief["B"] else "B") == goal)
    )
    if commitment_correct is None or defender.belief_at_commitment is None:
        raise RuntimeError("Episode ended before fixed Model A committed.")

    start = np.asarray(positions[0])
    target = target_positions[true_goal]
    shortest_length = int(np.abs(start - target).sum())
    reached_goal = bool(np.array_equal(np.asarray(positions[-1]), target))
    path_length = int(
        sum(
            np.abs(np.asarray(next_position) - np.asarray(previous_position)).sum()
            for previous_position, next_position in zip(positions, positions[1:])
        )
    )
    action_stats = _movement_statistics(actions, positions, true_goal, target_positions)
    true_goal_key = "A" if true_goal == 0 else "B"
    return {
        "policy": policy.name,
        "episode": episode,
        "seed": seed,
        "true_goal": true_goal_key,
        "deceiver_success": int(task_reward > 0),
        "reached_goal": int(reached_goal),
        "commitment_accuracy": commitment_correct,
        "misdirection_rate": 1 - commitment_correct,
        "belief_accuracy": float(np.mean(belief_correct)),
        "belief_at_commitment": defender.belief_at_commitment,
        "belief_shift_true_goal": float(
            defender.belief_at_commitment[true_goal_key] - 0.5
        ),
        "deception_cost": max(0, path_length - shortest_length),
        "path_length": path_length,
        "shortest_path_length": shortest_length,
        "path_efficiency": (
            min(1.0, shortest_length / path_length)
            if reached_goal and path_length
            else 0.0
        ),
        **action_stats,
        "action_sequence": [
            _ACTION_NAMES[action] for action in actions
        ],
        "trajectory": positions,
        "task_reward": task_reward,
        "deception_reward": deception_reward,
        "movement_cost": movement_cost,
        "detour_cost": detour_cost,
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialized = row.copy()
            for field in ("belief_at_commitment", "action_sequence", "trajectory"):
                if field in serialized:
                    serialized[field] = json.dumps(
                        serialized[field], separators=(",", ":"), sort_keys=True
                    )
            writer.writerow(serialized)


def _summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["policy"], []).append(row)
    summary_rows = []
    action_rows = []
    for policy_name, policy_rows in grouped.items():
        summary_rows.append(
            {
                "policy": policy_name,
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
        action_summary = {"policy": policy_name, "episodes": len(policy_rows)}
        for action_name in ("up", "down", "left", "right", "stay"):
            action_summary[f"mean_{action_name}_count"] = float(
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
            output_name = "mean_" + metric if metric != "path_efficiency" else "mean_path_efficiency"
            action_summary[output_name] = float(
                np.mean([row[metric] for row in policy_rows])
            )
        action_rows.append(action_summary)

    success_by_policy = {
        row["policy"]: row["deceiver_success_rate"] for row in summary_rows
    }
    random_success = success_by_policy["random_policy"]
    scripted_success = success_by_policy["scripted_deception"]
    for row in summary_rows:
        row["success_rate_delta_vs_random"] = (
            row["deceiver_success_rate"] - random_success
        )
        row["success_rate_delta_vs_scripted"] = (
            row["deceiver_success_rate"] - scripted_success
        )
    return summary_rows, action_rows


def run_ppo_experiment(
    output_dir: str | Path = "outputs/ppo_deceiver",
    defender_checkpoint: str | Path = "outputs/data_scaling/modela.pt",
    total_timesteps: int = 2048,
    evaluation_episodes: int = 10,
    seed: int = 0,
    step_commit: int = 5,
    device: str = "auto",
) -> dict[str, Path]:
    if total_timesteps < 128 or evaluation_episodes < 2:
        raise ValueError("Sanity training requires >=128 timesteps and >=2 eval episodes.")
    checkpoint = Path(defender_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Fixed Model A checkpoint not found: {checkpoint}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    estimator = GRUBeliefEstimator.from_checkpoint(checkpoint)
    train_env = PPODeceiverEnv(estimator, step_commit=step_commit)
    check_env(train_env, warn=True)
    torch.manual_seed(seed)
    model = PPO(
        "MlpPolicy",
        train_env,
        seed=seed,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        verbose=0,
        device=device,
    )
    model.learn(total_timesteps=total_timesteps)
    model_path = output_path / "ppo_deceiver.zip"
    model.save(model_path)

    evaluation_rows = []
    for policy_index, policy_name in enumerate(_POLICY_NAMES):
        for episode in range(evaluation_episodes):
            episode_seed = seed + 100_003 + policy_index * 10_007 + episode
            goal = "A" if episode % 2 == 0 else "B"
            policy = _policy_instances(model, episode_seed)[policy_index]
            evaluation_rows.append(
                _evaluate_episode(
                    policy,
                    episode,
                    episode_seed,
                    goal,
                    estimator,
                    step_commit,
                )
            )

    summary_rows, action_rows = _summarize(evaluation_rows)
    episode_path = output_path / "evaluation_episodes.csv"
    summary_path = output_path / "evaluation_summary.csv"
    action_path = output_path / "action_statistics.csv"
    _write_csv(episode_path, _EPISODE_FIELDS, evaluation_rows)
    _write_csv(summary_path, _SUMMARY_FIELDS, summary_rows)
    _write_csv(action_path, _ACTION_FIELDS, action_rows)
    trajectory_figure = output_path / "trajectory_diagnostics.png"
    plot_trajectories(episode_path, trajectory_figure)
    return {
        "model": model_path,
        "episodes": episode_path,
        "summary": summary_path,
        "actions": action_path,
        "trajectories": trajectory_figure,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ppo_deceiver"))
    parser.add_argument("--defender-checkpoint", type=Path, default=Path("outputs/data_scaling/modela.pt"))
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--step-commit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    outputs = run_ppo_experiment(
        output_dir=args.output_dir,
        defender_checkpoint=args.defender_checkpoint,
        total_timesteps=args.timesteps,
        evaluation_episodes=args.episodes,
        seed=args.seed,
        step_commit=args.step_commit,
        device=args.device,
    )
    print(f"Wrote PPO Deceiver sanity results to {outputs['summary']}")


if __name__ == "__main__":
    main()
