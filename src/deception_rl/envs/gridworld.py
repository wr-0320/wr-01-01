"""7x7 partially observable deception game implemented as a PettingZoo env."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

GOAL_A = 0
GOAL_B = 1

MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3
MOVE_STAY = 4

GUARD_A = 0
GUARD_B = 1
OBSERVE = 2
TRACK = 3

_MOVES = {
    MOVE_UP: (-1, 0),
    MOVE_DOWN: (1, 0),
    MOVE_LEFT: (0, -1),
    MOVE_RIGHT: (0, 1),
    MOVE_STAY: (0, 0),
}


class GridWorldParallelEnv(ParallelEnv):
    """A two-agent grid game with hidden intent and action-dependent sensing."""

    metadata = {"name": "deception_gridworld_v0", "render_modes": ["ansi"]}

    def __init__(
        self,
        size: int = 7,
        max_cycles: int = 30,
        vision_radius: int = 1,
        observe_radius: int = 2,
        track_radius: int = 4,
        render_mode: str | None = None,
    ) -> None:
        if size != 7:
            raise ValueError("The MVP GridWorld has a fixed size of 7x7.")
        if max_cycles < 1:
            raise ValueError("max_cycles must be positive.")
        if not 0 <= vision_radius <= observe_radius <= track_radius:
            raise ValueError("Sensor radii must satisfy vision <= observe <= track.")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'.")

        self.size = size
        self.max_cycles = max_cycles
        self.vision_radius = vision_radius
        self.observe_radius = observe_radius
        self.track_radius = track_radius
        self.render_mode = render_mode

        self.possible_agents = ["deceiver", "defender"]
        self.agents: list[str] = []
        self._action_spaces = {
            "deceiver": spaces.Discrete(len(_MOVES)),
            "defender": spaces.Discrete(4),
        }
        self._observation_spaces = {
            "deceiver": spaces.Dict(
                {
                    "position": spaces.Box(0, size - 1, shape=(2,), dtype=np.int64),
                    "goal": spaces.Discrete(2),
                    "target_positions": spaces.Box(
                        0, size - 1, shape=(2, 2), dtype=np.int64
                    ),
                }
            ),
            "defender": spaces.Dict(
                {
                    "local_map": spaces.Box(
                        0, 4, shape=(size, size), dtype=np.int8
                    ),
                    "trajectory": spaces.Box(
                        -1,
                        size - 1,
                        shape=(max_cycles + 1, 2),
                        dtype=np.int64,
                    ),
                    "trajectory_mask": spaces.MultiBinary(max_cycles + 1),
                    "sensor_origin": spaces.Box(
                        0, size - 1, shape=(2,), dtype=np.int64
                    ),
                    "target_positions": spaces.Box(
                        0, size - 1, shape=(2, 2), dtype=np.int64
                    ),
                }
            ),
        }
        self.np_random, _ = seeding.np_random(None)
        self._target_positions = np.array(
            [[0, 0], [size - 1, size - 1]], dtype=np.int64
        )
        self._sensor_origin = np.array([size // 2, size // 2], dtype=np.int64)
        self._deceiver_position = self._sensor_origin.copy()
        self._goal = GOAL_A
        self._step_count = 0
        self._sensor_radius = vision_radius
        self._observed_trajectory = np.full(
            (max_cycles + 1, 2), -1, dtype=np.int64
        )
        self._trajectory_mask = np.zeros(max_cycles + 1, dtype=np.int8)

    def observation_space(self, agent: str) -> spaces.Space[Any]:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space[Any]:
        return self._action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray | int]], dict[str, dict[str, Any]]]:
        self.np_random, _ = seeding.np_random(seed)
        options = options or {}
        requested_goal = options.get("goal")
        if requested_goal is None:
            self._goal = int(self.np_random.integers(0, 2))
        elif requested_goal in (GOAL_A, GOAL_B):
            self._goal = int(requested_goal)
        elif requested_goal == "A":
            self._goal = GOAL_A
        elif requested_goal == "B":
            self._goal = GOAL_B
        else:
            raise ValueError("options['goal'] must be 'A', 'B', 0, or 1.")

        self.agents = self.possible_agents.copy()
        self._deceiver_position = self._sensor_origin.copy()
        self._step_count = 0
        self._sensor_radius = self.vision_radius
        self._observed_trajectory.fill(-1)
        self._trajectory_mask.fill(0)
        self._capture_visible_position()

        observations = self._observations()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(
        self, actions: Mapping[str, int]
    ) -> tuple[
        dict[str, dict[str, np.ndarray | int]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self.agents:
            raise RuntimeError("Call reset() before stepping a finished episode.")
        if set(actions) != set(self.agents):
            raise ValueError("Actions must be provided for both active agents.")

        deceiver_action = int(actions["deceiver"])
        defender_action = int(actions["defender"])
        if not self._action_spaces["deceiver"].contains(deceiver_action):
            raise ValueError("Invalid Deceiver action.")
        if not self._action_spaces["defender"].contains(defender_action):
            raise ValueError("Invalid Defender action.")

        row_delta, col_delta = _MOVES[deceiver_action]
        next_position = self._deceiver_position + (row_delta, col_delta)
        self._deceiver_position = np.clip(next_position, 0, self.size - 1)
        self._step_count += 1

        self._sensor_radius = {
            GUARD_A: self.vision_radius,
            GUARD_B: self.vision_radius,
            OBSERVE: self.observe_radius,
            TRACK: self.track_radius,
        }[defender_action]
        self._capture_visible_position()

        reached_goal = np.array_equal(
            self._deceiver_position, self._target_positions[self._goal]
        )
        blocked = reached_goal and defender_action == self._goal
        escaped = reached_goal and not blocked
        terminated = blocked or escaped
        truncated = not terminated and self._step_count >= self.max_cycles

        rewards = {"deceiver": 0.0, "defender": 0.0}
        if blocked:
            rewards = {"deceiver": -1.0, "defender": 1.0}
        elif escaped:
            rewards = {"deceiver": 1.0, "defender": -1.0}

        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        observations = self._observations()
        infos = {agent: {} for agent in self.agents}
        if terminated or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None

        board = np.full((self.size, self.size), ".", dtype="<U1")
        board[tuple(self._target_positions[GOAL_A])] = "A"
        board[tuple(self._target_positions[GOAL_B])] = "B"
        board[tuple(self._sensor_origin)] = "D"
        board[tuple(self._deceiver_position)] = "X"
        return "\n".join(" ".join(row) for row in board)

    def _observations(self) -> dict[str, dict[str, np.ndarray | int]]:
        deceiver_observation: dict[str, np.ndarray | int] = {
            "position": self._deceiver_position.copy(),
            "goal": self._goal,
            "target_positions": self._target_positions.copy(),
        }
        defender_observation = self._defender_observation()
        return {"deceiver": deceiver_observation, "defender": defender_observation}

    def _defender_observation(self) -> dict[str, np.ndarray]:
        local_map = np.zeros((self.size, self.size), dtype=np.int8)
        local_map[tuple(self._target_positions[GOAL_A])] = 2
        local_map[tuple(self._target_positions[GOAL_B])] = 3

        rows, cols = np.indices((self.size, self.size))
        visible = (
            np.abs(rows - self._sensor_origin[0])
            + np.abs(cols - self._sensor_origin[1])
            <= self._sensor_radius
        )
        local_map[~visible] = 0
        local_map[visible & (local_map == 0)] = 1
        if visible[tuple(self._deceiver_position)]:
            local_map[tuple(self._deceiver_position)] = 4

        return {
            "local_map": local_map,
            "trajectory": self._observed_trajectory.copy(),
            "trajectory_mask": self._trajectory_mask.copy(),
            "sensor_origin": self._sensor_origin.copy(),
            "target_positions": self._target_positions.copy(),
        }

    def _capture_visible_position(self) -> None:
        distance = np.abs(self._deceiver_position - self._sensor_origin).sum()
        if distance <= self._sensor_radius:
            self._observed_trajectory[self._step_count] = self._deceiver_position
            self._trajectory_mask[self._step_count] = 1


def env(**kwargs: Any) -> GridWorldParallelEnv:
    """Factory for the MVP environment."""
    return GridWorldParallelEnv(**kwargs)
