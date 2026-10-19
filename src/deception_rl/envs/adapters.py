"""Adapters for the original GridWorld and the Stage 14 second environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deception_rl.envs.forked_routes import ForkedRoutesParallelEnv
from deception_rl.envs.gridworld import GridWorldParallelEnv
from deception_rl.envs.interface import DiscreteEnvironmentModel


ACTION_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))


@dataclass(frozen=True)
class GridWorldAdapter:
    model: DiscreteEnvironmentModel = DiscreteEnvironmentModel(
        name="deception_gridworld_v0",
        shape=(7, 7),
        start=(3, 3),
        goals=((0, 0), (6, 6)),
        traversable=frozenset((row, col) for row in range(7) for col in range(7)),
        action_deltas=ACTION_DELTAS,
        max_cycles=30,
    )

    def make_env(self, **kwargs: Any) -> GridWorldParallelEnv:
        return GridWorldParallelEnv(**kwargs)


@dataclass(frozen=True)
class ForkedRoutesAdapter:
    model: DiscreteEnvironmentModel = DiscreteEnvironmentModel(
        name="forked_routes_v0",
        shape=(7, 7),
        start=(6, 3),
        goals=((0, 1), (0, 5)),
        traversable=ForkedRoutesParallelEnv.TRAVERSABLE,
        action_deltas=ACTION_DELTAS,
        max_cycles=30,
    )

    def make_env(self, **kwargs: Any) -> ForkedRoutesParallelEnv:
        return ForkedRoutesParallelEnv(**kwargs)

