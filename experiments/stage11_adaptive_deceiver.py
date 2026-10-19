"""Stage 11 strategic Deceiver policies, isolated from the core environment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from deception_rl.envs.gridworld import (
    GUARD_A,
    GUARD_B,
    MOVE_STAY,
    OBSERVE,
    TRACK,
)
from experiments.deception_families import _action_toward


@dataclass
class DefenderBehavior:
    """Information exposed to the attacker by the Stage 11 experiment layer."""

    actions: list[int] = field(default_factory=list)
    commitments: list[tuple[int, int]] = field(default_factory=list)
    observation_visible: list[bool] = field(default_factory=list)


class AdaptiveDeceiver:
    """Rule-based online attacker with commitment, target and sensing adaptation.

    The cross-episode profile is intentionally tiny and deterministic: commit times
    and committed targets. ``reset`` clears only the current episode, while
    ``reset_profile`` also clears long-run behavior. No model or reward is trained.
    """

    name = "adaptive_deceiver"

    def __init__(self, seed: int | None = None) -> None:
        self.commit_times: list[int] = []
        self.committed_targets = np.zeros(2, dtype=np.int64)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._steps = 0
        self.behavior = DefenderBehavior()
        self.last_rule = "commitment_aware"

    def reset_profile(self, seed: int | None = None) -> None:
        self.commit_times.clear()
        self.committed_targets.fill(0)
        self.reset(seed)

    def observe_defender(
        self, action: int, timestep: int, attacker_was_visible: bool
    ) -> None:
        """Add the completed previous step to attacker-visible behavior history."""
        action = int(action)
        self.behavior.actions.append(action)
        self.behavior.observation_visible.append(bool(attacker_was_visible))
        if action in (GUARD_A, GUARD_B) and not self.behavior.commitments:
            self.behavior.commitments.append((int(timestep), action))
            self.commit_times.append(int(timestep))
            self.committed_targets[action] += 1

    @property
    def expected_commit_step(self) -> int:
        if not self.commit_times:
            return 5
        return int(np.clip(round(float(np.mean(self.commit_times))), 3, 12))

    @property
    def preferred_fake_target(self) -> int | None:
        total = int(self.committed_targets.sum())
        if total < 2 or self.committed_targets[0] == self.committed_targets[1]:
            return None
        # Target-aware rule requested by the protocol: induce the target opposite
        # the Defender's long-run commitment bias.
        biased_target = int(np.argmax(self.committed_targets))
        return 1 - biased_target

    def _fake_target(self, true_goal: int) -> int:
        preferred = self.preferred_fake_target
        return 1 - true_goal if preferred is None else preferred

    def act(self, observation: dict[str, np.ndarray | int]) -> int:
        position = np.asarray(observation["position"], dtype=np.int64)
        true_goal = int(observation["goal"])
        targets = np.asarray(observation["target_positions"], dtype=np.int64)
        fake_goal = self._fake_target(true_goal)
        expected_commit = self.expected_commit_step
        latest_action = self.behavior.actions[-1] if self.behavior.actions else None

        # A. Commitment-aware: place the fake cue immediately before a learned
        # fixed commitment time, then switch to the real goal after it.
        fake_start = max(0, expected_commit - 3)
        fake_end = expected_commit + 1
        target = targets[true_goal]
        self.last_rule = "commitment_aware"
        if fake_start <= self._steps < fake_end:
            target = targets[fake_goal]

        # B. Target-aware: when a long-run bias exists, reinforce the opposite
        # candidate during the pre-commit window, unless it is the true goal.
        preferred = self.preferred_fake_target
        if preferred is not None and self._steps <= expected_commit:
            target = targets[preferred]
            self.last_rule = "target_aware"

        # C. Observation-aware: TRACK gets an alternating cross-goal cue, OBSERVE
        # gets a stronger direct fake cue. Only completed previous actions count.
        if latest_action == TRACK and self._steps <= expected_commit + 2:
            target = targets[fake_goal if self._steps % 2 == 0 else true_goal]
            self.last_rule = "observation_aware_track"
        elif latest_action == OBSERVE and self._steps <= expected_commit + 2:
            target = targets[fake_goal]
            self.last_rule = "observation_aware_observe"

        self._steps += 1
        if np.array_equal(position, target):
            return MOVE_STAY
        return _action_toward(position, target)

    def profile(self) -> dict[str, object]:
        return {
            "expected_commit_step": self.expected_commit_step,
            "commit_times": list(self.commit_times),
            "committed_targets": self.committed_targets.tolist(),
            "preferred_fake_target": self.preferred_fake_target,
        }

