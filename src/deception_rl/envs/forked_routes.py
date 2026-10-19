"""A second, obstacle-constrained anti-deception environment.

This environment is intentionally isolated from ``GridWorldParallelEnv``.  It
implements the same agent/action and timed-trajectory observation contract,
while its forked corridor topology gives both goals equal path length.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from deception_rl.envs.gridworld import (
    GOAL_A,
    GOAL_B,
    GUARD_A,
    GUARD_B,
    OBSERVE,
    TRACK,
)


_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}


class ForkedRoutesParallelEnv(ParallelEnv):
    """Symmetric forked routes with partial, action-dependent sensing."""

    metadata = {"name": "forked_routes_v0", "render_modes": ["ansi"]}
    TRAVERSABLE = frozenset(
        {(row, 3) for row in range(2, 7)}
        | {(2, col) for col in range(1, 6)}
        | {(row, 1) for row in range(3)}
        | {(row, 5) for row in range(3)}
    )

    def __init__(
        self,
        max_cycles: int = 30,
        vision_radius: int = 1,
        observe_radius: int = 2,
        track_radius: int = 4,
        render_mode: str | None = None,
    ) -> None:
        if max_cycles < 1:
            raise ValueError("max_cycles must be positive")
        if not 0 <= vision_radius <= observe_radius <= track_radius:
            raise ValueError("Sensor radii must satisfy vision <= observe <= track")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.size = 7
        self.max_cycles = int(max_cycles)
        self.vision_radius = int(vision_radius)
        self.observe_radius = int(observe_radius)
        self.track_radius = int(track_radius)
        self.render_mode = render_mode
        self.possible_agents = ["deceiver", "defender"]
        self.agents: list[str] = []
        self._action_spaces = {
            "deceiver": spaces.Discrete(5),
            "defender": spaces.Discrete(4),
        }
        self._observation_spaces = {
            "deceiver": spaces.Dict(
                {
                    "position": spaces.Box(0, 6, (2,), dtype=np.int64),
                    "goal": spaces.Discrete(2),
                    "target_positions": spaces.Box(0, 6, (2, 2), dtype=np.int64),
                    "traversable_map": spaces.MultiBinary((7, 7)),
                }
            ),
            "defender": spaces.Dict(
                {
                    "local_map": spaces.Box(0, 5, (7, 7), dtype=np.int8),
                    "trajectory": spaces.Box(-1, 6, (self.max_cycles + 1, 2), dtype=np.int64),
                    "trajectory_mask": spaces.MultiBinary(self.max_cycles + 1),
                    "sensor_origin": spaces.Box(0, 6, (2,), dtype=np.int64),
                    "target_positions": spaces.Box(0, 6, (2, 2), dtype=np.int64),
                    "traversable_map": spaces.MultiBinary((7, 7)),
                }
            ),
        }
        self._start = np.asarray([6, 3], dtype=np.int64)
        self._targets = np.asarray([[0, 1], [0, 5]], dtype=np.int64)
        self._sensor_origin = np.asarray([3, 3], dtype=np.int64)
        self._traversable_map = np.zeros((7, 7), dtype=np.int8)
        for position in self.TRAVERSABLE:
            self._traversable_map[position] = 1
        self.np_random, _ = seeding.np_random(None)
        self._trajectory = np.full((self.max_cycles + 1, 2), -1, dtype=np.int64)
        self._mask = np.zeros(self.max_cycles + 1, dtype=np.int8)
        self._position = self._start.copy()
        self._goal = GOAL_A
        self._step_count = 0
        self._sensor_radius = self.vision_radius

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def reset(self, seed=None, options: Mapping[str, Any] | None = None):
        # PettingZoo's contract checker may override max_cycles before reset.
        # Keep the time-indexed space and buffers consistent with that public
        # attribute rather than relying on the constructor value.
        if self._trajectory.shape[0] != self.max_cycles + 1:
            self._trajectory = np.full(
                (self.max_cycles + 1, 2), -1, dtype=np.int64
            )
            self._mask = np.zeros(self.max_cycles + 1, dtype=np.int8)
            self._observation_spaces["defender"] = spaces.Dict(
                {
                    "local_map": spaces.Box(0, 5, (7, 7), dtype=np.int8),
                    "trajectory": spaces.Box(
                        -1, 6, (self.max_cycles + 1, 2), dtype=np.int64
                    ),
                    "trajectory_mask": spaces.MultiBinary(self.max_cycles + 1),
                    "sensor_origin": spaces.Box(0, 6, (2,), dtype=np.int64),
                    "target_positions": spaces.Box(0, 6, (2, 2), dtype=np.int64),
                    "traversable_map": spaces.MultiBinary((7, 7)),
                }
            )
        self.np_random, _ = seeding.np_random(seed)
        requested = (options or {}).get("goal")
        if requested is None:
            self._goal = int(self.np_random.integers(0, 2))
        elif requested in (GOAL_A, "A"):
            self._goal = GOAL_A
        elif requested in (GOAL_B, "B"):
            self._goal = GOAL_B
        else:
            raise ValueError("options['goal'] must be 'A', 'B', 0, or 1")
        self.agents = self.possible_agents.copy()
        self._position = self._start.copy()
        self._step_count = 0
        self._sensor_radius = self.vision_radius
        self._trajectory.fill(-1)
        self._mask.fill(0)
        self._capture()
        return self._observations(), {agent: {} for agent in self.agents}

    def step(self, actions: Mapping[str, int]):
        if not self.agents:
            raise RuntimeError("Call reset() before stepping a finished episode")
        if set(actions) != set(self.agents):
            raise ValueError("Actions must be provided for both active agents")
        move = int(actions["deceiver"])
        defender_action = int(actions["defender"])
        if not self._action_spaces["deceiver"].contains(move):
            raise ValueError("Invalid Deceiver action")
        if not self._action_spaces["defender"].contains(defender_action):
            raise ValueError("Invalid Defender action")
        delta = _MOVES[move]
        candidate = (int(self._position[0] + delta[0]), int(self._position[1] + delta[1]))
        if candidate in self.TRAVERSABLE:
            self._position[:] = candidate
        self._step_count += 1
        self._sensor_radius = {
            GUARD_A: self.vision_radius,
            GUARD_B: self.vision_radius,
            OBSERVE: self.observe_radius,
            TRACK: self.track_radius,
        }[defender_action]
        self._capture()
        reached = np.array_equal(self._position, self._targets[self._goal])
        blocked = bool(reached and defender_action == self._goal)
        escaped = bool(reached and not blocked)
        terminated = blocked or escaped
        truncated = not terminated and self._step_count >= self.max_cycles
        rewards = {"deceiver": 0.0, "defender": 0.0}
        if blocked:
            rewards = {"deceiver": -1.0, "defender": 1.0}
        elif escaped:
            rewards = {"deceiver": 1.0, "defender": -1.0}
        observations = self._observations()
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        if terminated or truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def _capture(self) -> None:
        if np.abs(self._position - self._sensor_origin).sum() <= self._sensor_radius:
            self._trajectory[self._step_count] = self._position
            self._mask[self._step_count] = 1

    def _observations(self):
        local_map = np.zeros((7, 7), dtype=np.int8)
        rows, cols = np.indices((7, 7))
        visible = np.abs(rows - 3) + np.abs(cols - 3) <= self._sensor_radius
        local_map[visible & (self._traversable_map == 1)] = 1
        local_map[visible & (self._traversable_map == 0)] = 5
        local_map[tuple(self._targets[0])] = 2
        local_map[tuple(self._targets[1])] = 3
        if visible[tuple(self._position)]:
            local_map[tuple(self._position)] = 4
        common = {
            "target_positions": self._targets.copy(),
            "traversable_map": self._traversable_map.copy(),
        }
        return {
            "deceiver": {"position": self._position.copy(), "goal": self._goal, **common},
            "defender": {
                "local_map": local_map,
                "trajectory": self._trajectory.copy(),
                "trajectory_mask": self._mask.copy(),
                "sensor_origin": self._sensor_origin.copy(),
                **common,
            },
        }

    def render(self):
        if self.render_mode != "ansi":
            return None
        board = np.full((7, 7), "#", dtype="<U1")
        for position in self.TRAVERSABLE:
            board[position] = "."
        board[tuple(self._targets[0])] = "A"
        board[tuple(self._targets[1])] = "B"
        board[tuple(self._sensor_origin)] = "D"
        board[tuple(self._position)] = "X"
        return "\n".join(" ".join(row) for row in board)

