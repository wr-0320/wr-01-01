"""Fixed evaluation-only attackers for Stage 15 robustness validation.

No class in this module learns from reward or updates model parameters.  The
adaptive attacker keeps only observable, cross-episode Defender statistics.
Route-aware policies use an environment adapter so the same declared family
semantics can be exercised on the obstacle-constrained Forked Routes map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from deception_rl.envs.gridworld import GUARD_A, GUARD_B, MOVE_STAY, OBSERVE, TRACK
from deception_rl.envs.interface import DiscreteEnvironmentModel, Position


@lru_cache(maxsize=16)
def _distance_map(model: DiscreteEnvironmentModel, target: Position) -> dict[Position, int]:
    return model.distances_to(target)


def shortest_action(
    position: np.ndarray | Position,
    target: np.ndarray | Position,
    model: DiscreteEnvironmentModel,
) -> int:
    """Return a deterministic legal action on ``model`` toward ``target``."""
    state = tuple(int(value) for value in position)
    destination = tuple(int(value) for value in target)
    if state == destination:
        return MOVE_STAY
    distances = _distance_map(model, destination)
    candidates = []
    for action in range(4):
        successor = model.transition(state, action)
        candidates.append((distances[successor], action))
    return min(candidates)[1]


@dataclass
class ObservableDefenderProfile:
    """Only completed public actions and visibility outcomes are retained."""

    actions: list[int] = field(default_factory=list)
    visible: list[bool] = field(default_factory=list)
    commitments: list[tuple[int, int]] = field(default_factory=list)


class AdaptiveDeceiverV2:
    """Predeclared stronger rule attacker; it is never optimized on results.

    Compared with Stage 11, v2 uses a wider commitment window, varies reversal
    timing when historic commitment time is diffuse, extends a false cue when
    the Defender's historic target bias agrees with that cue, and responds to
    completed sensing actions with fixed multi-switch behavior.  It sees no
    current simultaneous Defender action or private model state.
    """

    name = "adaptive_deceiver_v2"

    def __init__(self, model: DiscreteEnvironmentModel, seed: int = 0) -> None:
        self.model = model
        self.commit_times: list[int] = []
        self.committed_targets = np.zeros(2, dtype=np.int64)
        self.episode_index = 0
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._steps = 0
        self.behavior = ObservableDefenderProfile()
        self.last_rule = "commitment_window"
        self._timing_jitter = int(self._rng.choice((-2, 0, 2)))
        self.episode_index += 1

    def reset_profile(self, seed: int | None = None) -> None:
        self.commit_times.clear()
        self.committed_targets.fill(0)
        self.episode_index = 0
        self.reset(seed)

    def observe_defender(self, action: int, timestep: int, attacker_was_visible: bool) -> None:
        action = int(action)
        self.behavior.actions.append(action)
        self.behavior.visible.append(bool(attacker_was_visible))
        if action in (GUARD_A, GUARD_B) and not self.behavior.commitments:
            self.behavior.commitments.append((int(timestep), action))
            self.commit_times.append(int(timestep))
            self.committed_targets[action] += 1

    @property
    def expected_commit_step(self) -> int:
        if not self.commit_times:
            return 6
        return int(np.clip(round(float(np.mean(self.commit_times))), 3, 14))

    @property
    def commitment_spread(self) -> float:
        if len(self.commit_times) < 2:
            return 0.0
        return float(np.std(self.commit_times))

    @property
    def defender_target_bias(self) -> int | None:
        total = int(self.committed_targets.sum())
        if total < 2 or self.committed_targets[0] == self.committed_targets[1]:
            return None
        return int(np.argmax(self.committed_targets))

    def act(self, observation: dict[str, np.ndarray | int]) -> int:
        position = np.asarray(observation["position"], dtype=np.int64)
        true_goal = int(observation["goal"])
        targets = np.asarray(observation["target_positions"], dtype=np.int64)
        false_goal = 1 - true_goal
        expected = self.expected_commit_step
        jitter = self._timing_jitter if self.commitment_spread > 1.0 else 0
        fake_start = max(0, expected - 5 + jitter)
        fake_end = expected + 3 + jitter
        if self.defender_target_bias == false_goal:
            fake_end += 2

        target = targets[true_goal]
        self.last_rule = "honest_tail"
        if fake_start <= self._steps < fake_end:
            target = targets[false_goal]
            self.last_rule = "commitment_window"

        latest = self.behavior.actions[-1] if self.behavior.actions else None
        recent_sensing = sum(action in (OBSERVE, TRACK) for action in self.behavior.actions[-3:])
        if not self.behavior.commitments and latest == OBSERVE and self._steps <= fake_end + 2:
            target = targets[false_goal]
            self.last_rule = "observe_reinforcement"
        elif not self.behavior.commitments and latest == TRACK and self._steps <= fake_end + 3:
            target = targets[false_goal if (self._steps // 2) % 2 == 0 else true_goal]
            self.last_rule = "track_multi_switch"
        elif not self.behavior.commitments and recent_sensing >= 2 and self._steps <= fake_end + 3:
            target = targets[false_goal]
            self.last_rule = "sensing_pressure"

        # Once a completed commitment is public, stop paying detour cost and
        # head to the true goal.  This reaction uses only the previous step.
        if self.behavior.commitments:
            target = targets[true_goal]
            self.last_rule = "post_commit_reversal"

        self._steps += 1
        return shortest_action(position, target, self.model)

    def profile(self) -> dict[str, object]:
        return {
            "expected_commit_step": self.expected_commit_step,
            "commitment_spread": self.commitment_spread,
            "commit_times": list(self.commit_times),
            "committed_targets": self.committed_targets.tolist(),
            "defender_target_bias": self.defender_target_bias,
            "timing_jitter": self._timing_jitter,
        }


class RouteAwareFamily:
    """Fixed deception families using only legal adapter transitions."""

    SUPPORTED = {
        "honest",
        "scripted_3",
        "scripted_8",
        "scripted_10",
        "delay_detour",
        "detour_switch",
        "stochastic_deception",
    }

    def __init__(self, family: str, model: DiscreteEnvironmentModel) -> None:
        if family not in self.SUPPORTED:
            raise ValueError(f"Unknown route-aware family: {family}")
        self.family = family
        self.model = model
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self.steps = 0
        self.rng = np.random.default_rng(seed)
        self.last_rule = self.family

    def _target_index(self, true_goal: int) -> int:
        timestep = self.steps
        if self.family == "honest":
            return true_goal
        if self.family.startswith("scripted_"):
            duration = int(self.family.rsplit("_", 1)[1])
            return 1 - true_goal if timestep < duration else true_goal
        if self.family == "delay_detour":
            if timestep < 3:
                return true_goal
            if timestep < 8:
                return 1 - true_goal
            return true_goal
        if self.family == "detour_switch":
            if timestep < 8:
                return (1 - true_goal) if (timestep // 2) % 2 == 0 else true_goal
            return true_goal
        if self.family == "stochastic_deception" and timestep < 8:
            draw = float(self.rng.random())
            if draw < 0.5:
                return 1 - true_goal
            if draw < 0.8:
                return true_goal
            return -1
        return true_goal

    def act(self, observation: dict[str, np.ndarray | int]) -> int:
        position = np.asarray(observation["position"], dtype=np.int64)
        true_goal = int(observation["goal"])
        targets = np.asarray(observation["target_positions"], dtype=np.int64)
        target_index = self._target_index(true_goal)
        self.steps += 1
        if target_index < 0:
            legal = [
                action
                for action in range(5)
                if self.model.transition(tuple(position.tolist()), action)
                in self.model.traversable
            ]
            return int(self.rng.choice(legal))
        return shortest_action(position, targets[target_index], self.model)
