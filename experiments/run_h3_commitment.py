"""Compare belief-driven commitment under honest and deceptive trajectories."""

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
    HonestShortestPath,
    ScriptedDeception,
)
from deception_rl.beliefs import BeliefEstimator
from deception_rl.envs.gridworld import (
    GUARD_A,
    GUARD_B,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
    GridWorldParallelEnv,
)

_ACTION_DELTAS = {
    MOVE_UP: (-1, 0),
    MOVE_DOWN: (1, 0),
    MOVE_LEFT: (0, -1),
    MOVE_RIGHT: (0, 1),
}
_GOAL_IDS = {"A": 0, "B": 1}
_DECISIONS = {GUARD_A: "A", GUARD_B: "B"}
_FIELDNAMES = [
    "deceiver_policy",
    "episode",
    "step_commit",
    "true_goal",
    "belief_at_commitment",
    "defender_decision",
    "decision_correct",
    "trajectory",
    "deceiver_success",
    "defender_success",
    "shortest_path_length",
    "scripted_path_length",
    "deception_cost",
    "deceiver_success_rate",
]


def _visible_trajectory(observation: dict[str, np.ndarray]) -> list[list[int]]:
    trajectory = observation["trajectory"]
    mask = observation["trajectory_mask"].astype(bool)
    return [
        [int(time), int(position[0]), int(position[1])]
        for time, position in enumerate(trajectory)
        if mask[time]
    ]


def _scripted_path_length(
    deception_steps: int,
    goal: str,
    start: np.ndarray,
    target_positions: np.ndarray,
    size: int,
) -> int:
    policy = ScriptedDeception(deception_steps=deception_steps)
    position = start.copy()
    observation: dict[str, np.ndarray | int] = {
        "position": position.copy(),
        "goal": _GOAL_IDS[goal],
        "target_positions": target_positions,
    }
    true_target = target_positions[_GOAL_IDS[goal]]
    path_length = 0

    for _ in range(size * size * 2):
        if np.array_equal(position, true_target):
            return path_length
        action = policy.act(observation)
        if action not in _ACTION_DELTAS:
            raise ValueError("Scripted route unexpectedly selected a non-movement action.")
        delta = np.asarray(_ACTION_DELTAS[action], dtype=np.int64)
        next_position = np.clip(position + delta, 0, size - 1)
        path_length += int(np.abs(next_position - position).sum())
        position = next_position
        observation["position"] = position.copy()

    raise RuntimeError("Scripted route did not reach its true objective.")


def _run_episode(
    policy: Any,
    episode: int,
    seed: int,
    goal: str,
    step_commit: int,
    max_cycles: int,
    estimator: BeliefEstimator | None,
) -> dict[str, Any]:
    env = GridWorldParallelEnv(max_cycles=max_cycles)
    policy.reset(seed=seed)
    observations, _ = env.reset(seed=seed, options={"goal": goal})
    defender = CommittedDefender(
        step_commit=step_commit,
        defender=BeliefBasedDefender(estimator=estimator),
    )

    while env.agents:
        defender_action = defender.act(observations["defender"])
        deceiver_action = policy.act(observations["deceiver"])
        observations, rewards, _, _, _ = env.step(
            {"deceiver": deceiver_action, "defender": defender_action}
        )

    if defender.decision is None or defender.belief_at_commitment is None:
        raise RuntimeError("Episode ended before the Defender committed.")

    target_positions = observations["deceiver"]["target_positions"]
    start = np.array([env.size // 2, env.size // 2], dtype=np.int64)
    shortest_path_length = int(
        np.abs(start - target_positions[_GOAL_IDS[goal]]).sum()
    )
    scripted_path_length = _scripted_path_length(
        deception_steps=2,
        goal=goal,
        start=start,
        target_positions=target_positions,
        size=env.size,
    )

    return {
        "deceiver_policy": policy.name,
        "episode": episode,
        "step_commit": step_commit,
        "true_goal": goal,
        "belief_at_commitment": defender.belief_at_commitment,
        "defender_decision": _DECISIONS[defender.decision],
        "decision_correct": int(_DECISIONS[defender.decision] == goal),
        "trajectory": _visible_trajectory(observations["defender"]),
        "deceiver_success": int(rewards["deceiver"] > 0),
        "defender_success": int(rewards["defender"] > 0),
        "shortest_path_length": shortest_path_length,
        "scripted_path_length": scripted_path_length,
        "deception_cost": scripted_path_length - shortest_path_length,
    }


def run_experiment(
    episodes_per_policy: int = 20,
    step_commit: int = 2,
    seed: int = 0,
    output_path: str | Path = "outputs/h3_commitment_results.csv",
    max_cycles: int = 30,
    estimator: BeliefEstimator | None = None,
) -> Path:
    if episodes_per_policy < 1:
        raise ValueError("episodes_per_policy must be positive.")
    if not 0 <= step_commit < 6:
        raise ValueError("step_commit must be in [0, 6) to precede goal arrival.")
    if max_cycles <= step_commit:
        raise ValueError("max_cycles must exceed step_commit.")

    policies = [HonestShortestPath(), ScriptedDeception()]
    rows: list[dict[str, Any]] = []
    for policy_index, policy in enumerate(policies):
        policy_rows = []
        for episode in range(episodes_per_policy):
            episode_seed = seed + policy_index * episodes_per_policy + episode
            goal = "A" if episode % 2 == 0 else "B"
            policy_rows.append(
                _run_episode(
                    policy,
                    episode,
                    episode_seed,
                    goal,
                    step_commit,
                    max_cycles,
                    estimator,
                )
            )
        success_rate = float(
            np.mean([row["deceiver_success"] for row in policy_rows])
        )
        for row in policy_rows:
            row["deceiver_success_rate"] = success_rate
        rows.extend(policy_rows)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            serialized = row.copy()
            for field in ("belief_at_commitment", "trajectory"):
                serialized[field] = json.dumps(
                    serialized[field], separators=(",", ":"), sort_keys=True
                )
            writer.writerow(serialized)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--step-commit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=30)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/h3_commitment_results.csv")
    )
    args = parser.parse_args()
    output_path = run_experiment(
        episodes_per_policy=args.episodes,
        step_commit=args.step_commit,
        seed=args.seed,
        output_path=args.output,
        max_cycles=args.max_cycles,
    )
    print(f"Wrote H3 commitment results to {output_path}")


if __name__ == "__main__":
    main()
