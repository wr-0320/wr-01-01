"""Single-agent training adapter for a PPO Deceiver against fixed Model A."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from deception_rl.agents import BeliefBasedDefender, CommittedDefender
from deception_rl.beliefs import GRUBeliefEstimator
from deception_rl.envs.gridworld import (
    GUARD_A,
    GUARD_B,
    GridWorldParallelEnv,
)

OBSERVATION_SIZE = 8


def encode_deceiver_observation(observation: dict[str, np.ndarray | int]) -> np.ndarray:
    """Encode only fields present in the Deceiver's public environment view."""
    position = np.asarray(observation["position"], dtype=np.float32) / 6.0
    goal = int(observation["goal"])
    goal_one_hot = np.eye(2, dtype=np.float32)[goal]
    targets = np.asarray(observation["target_positions"], dtype=np.float32).reshape(-1) / 6.0
    return np.concatenate((position, goal_one_hot, targets)).astype(np.float32)


class PPODeceiverEnv(gym.Env[np.ndarray, int]):
    """Train one Deceiver while Defender follows a frozen estimator/policy."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        defender_estimator: GRUBeliefEstimator,
        step_commit: int = 5,
        max_cycles: int = 30,
        task_reward_scale: float = 1.0,
        misdirection_reward: float = 0.25,
        movement_cost: float = 0.01,
        detour_cost: float = 0.05,
        belief_observation: bool = False,
    ) -> None:
        super().__init__()
        if step_commit < 0 or step_commit >= max_cycles:
            raise ValueError("step_commit must be within the episode horizon.")
        if min(task_reward_scale, misdirection_reward, movement_cost, detour_cost) < 0:
            raise ValueError("Reward scales and costs must be non-negative.")

        self.parallel_env = GridWorldParallelEnv(max_cycles=max_cycles)
        self.fixed_defender = CommittedDefender(
            step_commit=step_commit,
            defender=BeliefBasedDefender(estimator=defender_estimator),
        )
        self.step_commit = step_commit
        self.task_reward_scale = task_reward_scale
        self.misdirection_reward = misdirection_reward
        self.movement_cost = movement_cost
        self.detour_cost = detour_cost
        self.belief_observation = belief_observation
        self.reward_mode = "custom"
        self._base_observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_SIZE + (4 if belief_observation else 0),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)
        self._observations: dict[str, dict[str, np.ndarray | int]] = {}
        self._positions: list[list[int]] = []
        self._actions: list[int] = []
        self._goal = 0
        self._episode_reward_components = {
            "task_reward": 0.0,
            "deception_reward": 0.0,
            "movement_cost": 0.0,
            "detour_cost": 0.0,
        }
        self._commitment_correct: int | None = None

    def set_reward_mode(self, reward_mode: str) -> None:
        reward_components = {
            "task_only": (0.0, 0.0, 0.0),
            "task_plus_deception": (0.25, 0.0, 0.0),
            "task_deception_cost": (0.25, 0.01, 0.05),
        }
        if reward_mode not in reward_components:
            raise ValueError(f"Unknown reward mode: {reward_mode}")
        self.misdirection_reward, self.movement_cost, self.detour_cost = (
            reward_components[reward_mode]
        )
        self.reward_mode = reward_mode

    def _encode_execution_observation(self) -> np.ndarray:
        deceiver_observation = self._observations["deceiver"]
        encoded = encode_deceiver_observation(deceiver_observation)
        if not self.belief_observation:
            return encoded

        belief = self.fixed_defender.defender.estimate_belief(
            self._observations["defender"]
        )
        probabilities = np.array([belief["A"], belief["B"]], dtype=np.float32)
        entropy = float(
            -(probabilities * np.log2(np.clip(probabilities, 1e-8, 1.0))).sum()
        )
        entropy = float(np.clip(entropy, 0.0, 1.0))
        committed = float(self.fixed_defender.decision is not None)
        return np.concatenate(
            (encoded, probabilities, np.array([entropy, committed], dtype=np.float32))
        ).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.fixed_defender.reset()
        self._observations, _ = self.parallel_env.reset(seed=seed, options=options)
        deceiver_observation = self._observations["deceiver"]
        self._goal = int(deceiver_observation["goal"])
        start = np.asarray(deceiver_observation["position"], dtype=np.int64)
        self._positions = [start.tolist()]
        self._actions = []
        self._episode_reward_components = {
            "task_reward": 0.0,
            "deception_reward": 0.0,
            "movement_cost": 0.0,
            "detour_cost": 0.0,
        }
        self._commitment_correct = None
        return self._encode_execution_observation(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.parallel_env.agents:
            raise RuntimeError("Call reset() before stepping a finished episode.")
        if not self.action_space.contains(action):
            raise ValueError("Invalid Deceiver action.")

        previous_observation = self._observations["deceiver"]
        previous_position = np.asarray(previous_observation["position"], dtype=np.int64)
        previous_target = np.asarray(
            previous_observation["target_positions"][self._goal], dtype=np.int64
        )
        previous_distance = int(np.abs(previous_position - previous_target).sum())

        defender_action = self.fixed_defender.act(self._observations["defender"])
        self._actions.append(int(action))
        self._observations, rewards, terminations, truncations, _ = self.parallel_env.step(
            {"deceiver": int(action), "defender": defender_action}
        )

        next_observation = self._observations["deceiver"]
        next_position = np.asarray(next_observation["position"], dtype=np.int64)
        next_distance = int(np.abs(next_position - previous_target).sum())
        moved = int(not np.array_equal(previous_position, next_position))

        task_reward = self.task_reward_scale * float(rewards["deceiver"])
        deception_reward = 0.0
        if self._commitment_correct is None and defender_action in (GUARD_A, GUARD_B):
            self._commitment_correct = int(defender_action == self._goal)
            if not self._commitment_correct:
                deception_reward = self.misdirection_reward

        movement_penalty = -self.movement_cost
        distance_increase = max(0, next_distance - previous_distance)
        detour_penalty = -self.detour_cost * distance_increase
        reward = task_reward + deception_reward + movement_penalty + detour_penalty

        self._episode_reward_components["task_reward"] += task_reward
        self._episode_reward_components["deception_reward"] += deception_reward
        self._episode_reward_components["movement_cost"] += movement_penalty
        self._episode_reward_components["detour_cost"] += detour_penalty
        self._positions.append(next_position.tolist())

        terminated = bool(terminations["deceiver"])
        truncated = bool(truncations["deceiver"])
        info = {
            "task_reward": task_reward,
            "deception_reward": deception_reward,
            "movement_cost": movement_penalty,
            "detour_cost": detour_penalty,
            "public_defender_action": int(defender_action),
            "reward_mode": self.reward_mode,
        }
        if terminated or truncated:
            info["episode_metrics"] = self.episode_metrics()
        return self._encode_execution_observation(), float(reward), terminated, truncated, info

    def episode_metrics(self) -> dict[str, Any]:
        target_positions = np.asarray(
            self._observations["deceiver"]["target_positions"], dtype=np.int64
        )
        start = np.asarray(self._positions[0], dtype=np.int64)
        target = target_positions[self._goal]
        shortest_length = int(np.abs(start - target).sum())
        path_length = int(
            sum(
                np.abs(np.asarray(next_position) - np.asarray(previous_position)).sum()
                for previous_position, next_position in zip(
                    self._positions, self._positions[1:]
                )
            )
        )
        success = bool(self._episode_reward_components["task_reward"] > 0)
        reached_goal = bool(np.array_equal(np.asarray(self._positions[-1]), target))
        return {
            "true_goal": "A" if self._goal == 0 else "B",
            "deceiver_success": int(success),
            "reached_goal": int(reached_goal),
            "commitment_accuracy": self._commitment_correct,
            "path_length": path_length,
            "shortest_path_length": shortest_length,
            "deception_cost": max(0, path_length - shortest_length),
            "path_efficiency": (
                min(1.0, shortest_length / path_length)
                if reached_goal and path_length
                else 0.0
            ),
            "action_sequence": self._actions.copy(),
            "trajectory": self._positions.copy(),
            **self._episode_reward_components,
        }
