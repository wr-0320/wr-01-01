"""Unseen Deceiver strategies used only by generalization experiments."""

from __future__ import annotations

import numpy as np

from deception_rl.envs.gridworld import (
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_STAY,
    MOVE_UP,
)


_MOVES = {
    MOVE_UP: (-1, 0),
    MOVE_DOWN: (1, 0),
    MOVE_LEFT: (0, -1),
    MOVE_RIGHT: (0, 1),
    MOVE_STAY: (0, 0),
}


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


class DelayedDeception:
    """First approach the true target, then feint toward the false target."""

    name = "delayed_deception"

    def __init__(self, delay_steps: int = 2, feint_steps: int = 2) -> None:
        if delay_steps < 1 or feint_steps < 1:
            raise ValueError("delay_steps and feint_steps must be positive.")
        self.delay_steps = delay_steps
        self.feint_steps = feint_steps
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._steps_taken = 0

    def act(self, observation: dict[str, np.ndarray | int]) -> int:
        position = np.asarray(observation["position"], dtype=np.int64)
        goal = int(observation["goal"])
        targets = np.asarray(observation["target_positions"], dtype=np.int64)
        deception_start = self.delay_steps
        deception_end = deception_start + self.feint_steps
        if deception_start <= self._steps_taken < deception_end:
            intended_goal = 1 - goal
        else:
            intended_goal = goal
        self._steps_taken += 1
        return _action_toward(position, targets[intended_goal])


class DetourDeception:
    """Take one lateral step toward the false side, then route to the true goal."""

    name = "detour_deception"

    def __init__(self, detour_steps: int = 1) -> None:
        if detour_steps < 1:
            raise ValueError("detour_steps must be positive.")
        self.detour_steps = detour_steps
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._steps_taken = 0

    def act(self, observation: dict[str, np.ndarray | int]) -> int:
        position = np.asarray(observation["position"], dtype=np.int64)
        goal = int(observation["goal"])
        targets = np.asarray(observation["target_positions"], dtype=np.int64)
        if self._steps_taken < self.detour_steps:
            false_target = targets[1 - goal]
            lateral_target = np.array([position[0], false_target[1]])
            action = _action_toward(position, lateral_target)
        else:
            action = _action_toward(position, targets[goal])
        self._steps_taken += 1
        return action


class MultiSwitchDeception:
    """Alternate true/false intent cues before committing to the true target."""

    name = "multi_switch_deception"

    def __init__(self, switches: int = 3) -> None:
        if switches < 1:
            raise ValueError("switches must be positive.")
        self.switches = switches
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._steps_taken = 0

    def act(self, observation: dict[str, np.ndarray | int]) -> int:
        position = np.asarray(observation["position"], dtype=np.int64)
        goal = int(observation["goal"])
        targets = np.asarray(observation["target_positions"], dtype=np.int64)
        if self._steps_taken < self.switches + 1:
            intended_goal = goal if self._steps_taken % 2 else 1 - goal
        else:
            intended_goal = goal
        self._steps_taken += 1
        return _action_toward(position, targets[intended_goal])


def action_delta(action: int) -> tuple[int, int]:
    """Expose movement deltas for policy-level unit tests."""
    return _MOVES[action]
