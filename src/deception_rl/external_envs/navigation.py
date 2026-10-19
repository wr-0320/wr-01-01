"""Shared-interface wrappers for the frozen GridWorld and Forked Routes."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import numpy as np
from gymnasium import spaces

from deception_rl.agents import HonestShortestPath, ScriptedDeception
from deception_rl.envs import ForkedRoutesAdapter, GridWorldAdapter
from deception_rl.envs.gridworld import GUARD_A, GUARD_B, OBSERVE, TRACK
from deception_rl.envs.interface import DiscreteEnvironmentModel, EnvironmentAdapter, Position
from deception_rl.external_envs.interface import (
    DefenderAction,
    DefenderObservation,
    DefenderTransition,
)


@lru_cache(maxsize=16)
def _distances(model: DiscreteEnvironmentModel, goal: Position) -> dict[Position, int]:
    return model.distances_to(goal)


def _shortest_action(position, target, model: DiscreteEnvironmentModel) -> int:
    state = tuple(int(value) for value in position)
    destination = tuple(int(value) for value in target)
    if state == destination:
        return 4
    distances = _distances(model, destination)
    return min(
        (distances[model.transition(state, action)], action) for action in range(4)
    )[1]


class _RouteAwareAttacker:
    def __init__(self, family: str, model: DiscreteEnvironmentModel) -> None:
        if family not in {"honest", "scripted_3", "scripted_8"}:
            raise ValueError("Unknown navigation attack family")
        self.family = family
        self.model = model
        self.reset()

    def reset(self, seed=None) -> None:
        del seed
        self.step = 0

    def act(self, observation) -> int:
        true_goal = int(observation["goal"])
        duration = 0 if self.family == "honest" else int(self.family.rsplit("_", 1)[1])
        target_goal = 1 - true_goal if self.step < duration else true_goal
        self.step += 1
        return _shortest_action(
            observation["position"], observation["target_positions"][target_goal], self.model
        )


class NavigationDefenderAdapter:
    """Convert either frozen navigation environment to the Stage 17 contract."""

    observation_size = 8
    action_space = spaces.Discrete(4)
    ACTION_MAP = {
        DefenderAction.OBSERVE: OBSERVE,
        DefenderAction.TRACK: TRACK,
        DefenderAction.COMMIT_A: GUARD_A,
        DefenderAction.COMMIT_B: GUARD_B,
    }

    def __init__(
        self,
        adapter: EnvironmentAdapter,
        attack_family: str = "scripted_3",
        observe_cost: float = 0.01,
        track_cost: float = 0.03,
    ) -> None:
        if attack_family not in {"honest", "scripted_3", "scripted_8"}:
            raise ValueError("Unknown navigation attack family")
        self.adapter = adapter
        self.model = adapter.model
        self.attack_family = attack_family
        self.name = f"{self.model.name}_external"
        self.costs = {
            DefenderAction.OBSERVE: float(observe_cost),
            DefenderAction.TRACK: float(track_cost),
            DefenderAction.COMMIT_A: 0.0,
            DefenderAction.COMMIT_B: 0.0,
        }
        if min(self.costs.values()) < 0:
            raise ValueError("Action costs must be nonnegative")
        self._done = True

    def _make_attacker(self):
        if isinstance(self.adapter, GridWorldAdapter):
            if self.attack_family == "honest":
                return HonestShortestPath()
            return ScriptedDeception(int(self.attack_family.rsplit("_", 1)[1]))
        return _RouteAwareAttacker(self.attack_family, self.model)

    def reset(self, seed=None, options: Mapping[str, Any] | None = None):
        self._env = self.adapter.make_env()
        self._raw, _ = self._env.reset(seed=seed, options=options)
        self._attacker = self._make_attacker()
        self._attacker.reset(seed=seed)
        self._goal = int(self._raw["deceiver"]["goal"])
        self._step = 0
        self._cumulative_cost = 0.0
        self._committed_goal = None
        self._done = False
        self._belief = (0.5, 0.5)
        return self._observation()

    def _update_belief(self) -> None:
        defender = self._raw["defender"]
        mask = np.asarray(defender["trajectory_mask"], dtype=bool)
        times = np.flatnonzero(mask)
        if not len(times):
            return
        trajectory = np.asarray(defender["trajectory"], dtype=int)
        current = tuple(int(value) for value in trajectory[times[-1]])
        logits = []
        for goal in self.model.goals:
            distance = _distances(self.model, goal)[current]
            progress = 0.0
            if len(times) > 1:
                previous = tuple(int(value) for value in trajectory[times[-2]])
                progress = _distances(self.model, goal)[previous] - distance
            logits.append(-0.55 * distance + 1.1 * progress)
        logits = np.asarray(logits, dtype=float)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        self._belief = (float(probabilities[0]), float(probabilities[1]))

    def _observation(self) -> DefenderObservation:
        defender = self._raw["defender"]
        mask = np.asarray(defender["trajectory_mask"], dtype=bool)
        times = np.flatnonzero(mask)
        if len(times):
            latest = np.asarray(defender["trajectory"], dtype=float)[times[-1]] / np.asarray(
                [self.model.shape[0] - 1, self.model.shape[1] - 1]
            )
            age = (self._step - int(times[-1])) / self.model.max_cycles
        else:
            latest = np.asarray(self.model.start, dtype=float) / np.asarray(
                [self.model.shape[0] - 1, self.model.shape[1] - 1]
            )
            age = 1.0
        entropy = -sum(value * np.log2(max(value, 1e-12)) for value in self._belief)
        vector = np.asarray(
            [
                latest[0],
                latest[1],
                self._belief[0],
                self._belief[1],
                entropy,
                age,
                self._step / self.model.max_cycles,
                self._cumulative_cost,
            ],
            dtype=np.float32,
        )
        if self._done:
            mask_actions = (False, False, False, False)
        elif self._committed_goal is None:
            mask_actions = (True, True, True, True)
        else:
            allowed = int(DefenderAction.COMMIT_A) + self._committed_goal
            mask_actions = tuple(index == allowed for index in range(4))
        return DefenderObservation(
            observation=vector,
            belief=self._belief,
            action_mask=mask_actions,
            step=self._step,
            cumulative_cost=self._cumulative_cost,
            committed_goal=self._committed_goal,
            source=self.name,
        )

    def step(self, action: DefenderAction | int) -> DefenderTransition:
        if self._done:
            raise RuntimeError("Call reset() before stepping a finished episode")
        action = DefenderAction(int(action))
        if self._committed_goal is not None:
            expected = DefenderAction.COMMIT_A if self._committed_goal == 0 else DefenderAction.COMMIT_B
            if action != expected:
                raise ValueError("Navigation commitment is irreversible")
        elif action in (DefenderAction.COMMIT_A, DefenderAction.COMMIT_B):
            self._committed_goal = int(action) - int(DefenderAction.COMMIT_A)
        raw_action = self.ACTION_MAP[action]
        deceiver_action = int(self._attacker.act(self._raw["deceiver"]))
        self._raw, rewards, terminations, truncations, _ = self._env.step(
            {"deceiver": deceiver_action, "defender": raw_action}
        )
        cost = self.costs[action]
        self._cumulative_cost += cost
        self._step += 1
        self._update_belief()
        terminated = bool(terminations["defender"])
        truncated = bool(truncations["defender"])
        self._done = terminated or truncated
        success = bool(rewards["defender"] > 0) if self._done else None
        observation = self._observation()
        info: dict[str, Any] = {"attack_family": self.attack_family}
        if self._done:
            info.update(
                {
                    "true_goal": self._goal,
                    "committed_goal": self._committed_goal,
                    "success": success,
                }
            )
        return DefenderTransition(
            observation=observation,
            belief=observation.belief,
            action=action,
            cost=cost,
            success=success,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def close(self) -> None:
        if hasattr(self, "_env"):
            self._env.close()
        self._done = True


class GridWorldDefenderAdapter(NavigationDefenderAdapter):
    def __init__(self, attack_family="scripted_3") -> None:
        super().__init__(GridWorldAdapter(), attack_family)


class ForkedRoutesDefenderAdapter(NavigationDefenderAdapter):
    def __init__(self, attack_family="scripted_3") -> None:
        super().__init__(ForkedRoutesAdapter(), attack_family)
