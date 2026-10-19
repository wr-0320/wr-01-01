"""Run the scripted H1/H2 baselines and save episode-level results to CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from deception_rl.agents import (
    BeliefBasedDefender,
    HonestShortestPath,
    RandomPolicy,
    ScriptedDeception,
)
from deception_rl.beliefs import BeliefEstimator
from deception_rl.envs.gridworld import (
    GUARD_A,
    GUARD_B,
    OBSERVE,
    TRACK,
    GridWorldParallelEnv,
)

_DEFENDER_ACTION_NAMES = {
    GUARD_A: "guard_A",
    GUARD_B: "guard_B",
    OBSERVE: "observe",
    TRACK: "track",
}
_FIELDNAMES = [
    "deceiver_policy",
    "deception_steps",
    "episode",
    "seed_group",
    "episode_seed",
    "true_goal",
    "trajectory",
    "estimated_belief",
    "belief_trace",
    "defender_action",
    "deceiver_success",
    "defender_success",
    "success_rate",
]


def _visible_trajectory(observation: dict[str, np.ndarray]) -> list[list[int]]:
    trajectory = observation["trajectory"]
    mask = observation["trajectory_mask"].astype(bool)
    return [
        [int(time), int(position[0]), int(position[1])]
        for time, position in enumerate(trajectory)
        if mask[time]
    ]


def _run_episode(
    deceiver: Any,
    episode: int,
    seed: int,
    goal: str,
    max_cycles: int,
    estimator: BeliefEstimator | None,
) -> dict[str, Any]:
    env = GridWorldParallelEnv(max_cycles=max_cycles)
    deceiver.reset(seed=seed)
    observations, _ = env.reset(seed=seed, options={"goal": goal})
    defender = BeliefBasedDefender(estimator=estimator)
    defender_action_history: list[dict[str, Any]] = []
    belief_trace: list[dict[str, float | int]] = []
    defender_observation = observations["defender"]
    initial_belief = defender.estimate_belief(defender_observation)
    belief_trace.append({"step": 0, **initial_belief})

    while env.agents:
        defender_action = defender.act(defender_observation)
        deceiver_action = deceiver.act(observations["deceiver"])
        defender_action_history.append(
            {
                "step": len(defender_action_history) + 1,
                "action": _DEFENDER_ACTION_NAMES[defender_action],
            }
        )
        observations, rewards, _, _, _ = env.step(
            {"deceiver": deceiver_action, "defender": defender_action}
        )
        defender_observation = observations["defender"]
        belief = defender.estimate_belief(defender_observation)
        belief_trace.append({"step": len(defender_action_history), **belief})

    return {
        "deceiver_policy": deceiver.name,
        "deception_steps": getattr(deceiver, "deception_steps", ""),
        "episode": episode,
        "true_goal": goal,
        "trajectory": _visible_trajectory(defender_observation),
        "estimated_belief": defender.last_belief,
        "belief_trace": belief_trace,
        "defender_action": defender_action_history,
        "deceiver_success": int(rewards["deceiver"] > 0),
        "defender_success": int(rewards["defender"] > 0),
    }


def run_experiment(
    episodes_per_policy: int = 20,
    seed: int = 0,
    output_path: str | Path = "outputs/baseline_results.csv",
    max_cycles: int = 30,
    estimator: BeliefEstimator | None = None,
    scripted_deception_steps: int | Sequence[int] = 2,
    seed_count: int = 1,
) -> Path:
    if episodes_per_policy < 1:
        raise ValueError("episodes_per_policy must be positive.")
    if max_cycles < 1:
        raise ValueError("max_cycles must be positive.")
    if seed_count < 1 or episodes_per_policy % seed_count:
        raise ValueError("episodes_per_policy must be divisible by positive seed_count.")

    deception_step_values = (
        (scripted_deception_steps,)
        if isinstance(scripted_deception_steps, int)
        else tuple(scripted_deception_steps)
    )
    if not deception_step_values or any(steps < 1 for steps in deception_step_values):
        raise ValueError("scripted_deception_steps must contain positive integers.")
    if len(set(deception_step_values)) != len(deception_step_values):
        raise ValueError("scripted_deception_steps must not contain duplicates.")

    policies = [
        HonestShortestPath(),
        RandomPolicy(),
        *[
            ScriptedDeception(deception_steps=steps)
            for steps in deception_step_values
        ],
    ]
    rows: list[dict[str, Any]] = []
    episodes_per_seed = episodes_per_policy // seed_count
    for policy_index, policy in enumerate(policies):
        policy_rows = []
        for episode in range(episodes_per_policy):
            seed_group = episode // episodes_per_seed
            episode_seed = seed + seed_group * 100_003 + episode % episodes_per_seed
            goal = "A" if episode % 2 == 0 else "B"
            row = _run_episode(
                policy, episode, episode_seed, goal, max_cycles, estimator
            )
            row["seed_group"] = seed_group
            row["episode_seed"] = episode_seed
            policy_rows.append(row)
        success_rate = float(
            np.mean([row["deceiver_success"] for row in policy_rows])
        )
        for row in policy_rows:
            row["success_rate"] = success_rate
        rows.extend(policy_rows)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            serialized = row.copy()
            for field in ("trajectory", "estimated_belief", "belief_trace", "defender_action"):
                serialized[field] = json.dumps(
                    serialized[field], separators=(",", ":"), sort_keys=True
                )
            writer.writerow(serialized)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=30)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument(
        "--scripted-deception-steps", type=int, nargs="+", default=[2]
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/baseline_results.csv")
    )
    args = parser.parse_args()
    output_path = run_experiment(
        episodes_per_policy=args.episodes,
        seed=args.seed,
        output_path=args.output,
        max_cycles=args.max_cycles,
        scripted_deception_steps=args.scripted_deception_steps,
        seed_count=args.seed_count,
    )
    print(f"Wrote baseline results to {output_path}")


if __name__ == "__main__":
    main()
