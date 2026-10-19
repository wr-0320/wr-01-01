"""Environment-neutral interface used by the Stage 14 experiment layer.

The interface describes dynamics needed by classical intent filters without
changing either environment implementation.  Coordinates are deliberately
kept as integer pairs so the existing trajectory models remain usable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


Position = tuple[int, int]


@runtime_checkable
class AntiDeceptionParallelEnv(Protocol):
    """Small common surface shared by anti-deception environments."""

    possible_agents: list[str]
    agents: list[str]

    def reset(
        self, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]: ...

    def step(self, actions: Mapping[str, int]) -> tuple[dict, dict, dict, dict, dict]: ...

    def action_space(self, agent: str): ...

    def observation_space(self, agent: str): ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class DiscreteEnvironmentModel:
    """Finite deterministic motion model exposed to classical baselines."""

    name: str
    shape: tuple[int, int]
    start: Position
    goals: tuple[Position, Position]
    traversable: frozenset[Position]
    action_deltas: tuple[Position, ...]
    max_cycles: int

    def transition(self, state: Position, action: int) -> Position:
        delta = self.action_deltas[action]
        candidate = (state[0] + delta[0], state[1] + delta[1])
        return candidate if candidate in self.traversable else state

    def successors(self, state: Position) -> tuple[Position, ...]:
        return tuple(self.transition(state, action) for action in range(len(self.action_deltas)))

    def shortest_distance(self, start: Position, goal: Position) -> int:
        if start not in self.traversable or goal not in self.traversable:
            raise ValueError("start and goal must be traversable")
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            state, distance = queue.popleft()
            if state == goal:
                return distance
            for successor in set(self.successors(state)):
                if successor not in visited:
                    visited.add(successor)
                    queue.append((successor, distance + 1))
        raise ValueError(f"No path from {start} to {goal}")

    def distances_to(self, goal: Position) -> dict[Position, int]:
        return {state: self.shortest_distance(state, goal) for state in self.traversable}


class EnvironmentAdapter(Protocol):
    """Factory plus exact finite model for a supported environment."""

    model: DiscreteEnvironmentModel

    def make_env(self, **kwargs: Any) -> AntiDeceptionParallelEnv: ...


def validate_environment_interface(adapter: EnvironmentAdapter) -> dict[str, Any]:
    """Run a deterministic contract check and return paper-manifest metadata."""

    env = adapter.make_env()
    observations, infos = env.reset(seed=0, options={"goal": "A"})
    if set(observations) != {"deceiver", "defender"} or set(infos) != set(observations):
        raise AssertionError("Environment must return both agents")
    defender = observations["defender"]
    required = {"trajectory", "trajectory_mask", "target_positions", "sensor_origin"}
    if not required <= set(defender):
        raise AssertionError(f"Defender observation missing {required - set(defender)}")
    for agent in env.possible_agents:
        if not env.observation_space(agent).contains(observations[agent]):
            raise AssertionError(f"Initial {agent} observation violates its space")
    _, rewards, terminations, truncations, _ = env.step({"deceiver": 4, "defender": 2})
    if set(rewards) != set(env.possible_agents):
        raise AssertionError("Reward dictionary does not cover both agents")
    env.close()
    return {
        "environment": adapter.model.name,
        "states": len(adapter.model.traversable),
        "start": list(adapter.model.start),
        "goals": [list(goal) for goal in adapter.model.goals],
        "shortest_path_A": adapter.model.shortest_distance(adapter.model.start, adapter.model.goals[0]),
        "shortest_path_B": adapter.model.shortest_distance(adapter.model.start, adapter.model.goals[1]),
        "parallel_api": True,
        "observation_contract": "timed visible trajectory + target positions + sensor origin",
        "smoke_terminated": bool(any(terminations.values()) or any(truncations.values())),
    }

