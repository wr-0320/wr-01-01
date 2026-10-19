"""Non-learning Deceiver policies for the first experiments."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from deception_rl.envs.gridworld import (
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_STAY,
    MOVE_UP,
    GOAL_A,
)


class DeceiverPolicy(Protocol):
    name: str

    def reset(self, seed: int | None = None) -> None: ...

    def act(self, observation: dict[str, np.ndarray | int]) -> int: ...


def _action_toward(position: np.ndarray, target: np.ndarray) -> int:
    row_delta, col_delta = target - position
    if row_delta < 0:
        return MOVE_UP
    if row_delta > 0:
        return MOVE_DOWN
    if col_delta < 0:
        return MOVE_LEFT
    if col_delta > 0:
        return MOVE_RIGHT
    return MOVE_STAY


class HonestShortestPath:
    """Move along a deterministic shortest path to the true objective."""

    name = "honest_shortest_path"

    def reset(self, seed: int | None = None) -> None:
        del seed

    def act(self, observation: dict[str, np.ndarray | int]) -> int:
        position = np.asarray(observation["position"], dtype=np.int64)
        goal = int(observation["goal"])
        targets = np.asarray(observation["target_positions"], dtype=np.int64)
        return _action_toward(position, targets[goal])


class RandomPolicy:
    """Choose uniformly from the Deceiver's five movement actions."""

    name = "random_policy"

    def __init__(self, seed: int | None = None) -> None:
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, np.ndarray | int]) -> int:
        del observation
        return int(self._rng.integers(0, 5))


class ScriptedDeception:
    """Approach the false objective briefly, then pursue the true objective."""

    name = "scripted_deception"

    def __init__(self, deception_steps: int = 2) -> None:
        if deception_steps < 1:
            raise ValueError("deception_steps must be positive.")
        self.deception_steps = deception_steps
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._steps_taken = 0

    def act(self, observation: dict[str, np.ndarray | int]) -> int:
        position = np.asarray(observation["position"], dtype=np.int64)
        goal = int(observation["goal"])
        targets = np.asarray(observation["target_positions"], dtype=np.int64)
        intended_goal = 1 - goal if self._steps_taken < self.deception_steps else goal
        self._steps_taken += 1
        return _action_toward(position, targets[intended_goal])
