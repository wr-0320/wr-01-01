"""Lightweight cyber-deception simulator with the Stage 17 Defender API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces

from deception_rl.external_envs.interface import (
    DefenderAction,
    DefenderObservation,
    DefenderTransition,
)


class CyberDeceptionEnv:
    """Two-asset intrusion simulator with deceptive alerts and containment.

    The hidden target is asset A or B.  Defender observations contain alerts
    and optional forensic cues, never the hidden target.  Scenario rules are
    fixed and contain no learned attacker.
    """

    name = "cyber_deception"
    observation_size = 8
    action_space = spaces.Discrete(4)
    SCENARIOS = ("honest_intrusion", "credential_decoy", "lateral_switch", "low_and_slow")

    def __init__(
        self,
        scenario: str = "credential_decoy",
        horizon: int = 10,
        observe_cost: float = 0.02,
        track_cost: float = 0.06,
    ) -> None:
        if scenario not in self.SCENARIOS:
            raise ValueError("Unknown cyber scenario")
        if horizon < 2:
            raise ValueError("horizon must be at least two")
        self.scenario = scenario
        self.horizon = int(horizon)
        self.costs = {
            DefenderAction.OBSERVE: float(observe_cost),
            DefenderAction.TRACK: float(track_cost),
            DefenderAction.COMMIT_A: 0.0,
            DefenderAction.COMMIT_B: 0.0,
        }
        if min(self.costs.values()) < 0:
            raise ValueError("Action costs must be nonnegative")
        self._done = True

    @staticmethod
    def _evidence_increment(cue: int, reliability: float) -> float:
        magnitude = float(np.log(reliability / (1 - reliability)))
        return magnitude if cue == 1 else -magnitude

    def reset(self, seed=None, options: Mapping[str, Any] | None = None):
        self._rng = np.random.default_rng(seed)
        requested = (options or {}).get("goal")
        if requested is None:
            self._goal = int(self._rng.integers(0, 2))
        elif requested in (0, "A"):
            self._goal = 0
        elif requested in (1, "B"):
            self._goal = 1
        else:
            raise ValueError("options['goal'] must be A, B, 0, or 1")
        self._step = 0
        self._belief_log_odds = 0.0
        self._cumulative_cost = 0.0
        self._committed_goal = None
        self._last_action = None
        self._last_alert = None
        self._last_forensic = None
        self._done = False
        return self._observation()

    def _intended_alert(self) -> int | None:
        if self.scenario == "honest_intrusion":
            return self._goal
        if self.scenario == "credential_decoy":
            return 1 - self._goal if self._step < 4 else self._goal
        if self.scenario == "lateral_switch":
            return (1 - self._goal) if (self._step // 2) % 2 == 0 and self._step < 6 else self._goal
        if self._step < 6:
            return None if self._step % 2 == 0 else 1 - self._goal
        return self._goal

    def _sense(self, action: DefenderAction) -> None:
        intended = self._intended_alert()
        self._last_alert = None
        self._last_forensic = None
        if intended is not None:
            detection = 0.72 if action == DefenderAction.OBSERVE else 0.9
            if self._rng.random() < detection:
                self._last_alert = intended
        if action == DefenderAction.TRACK:
            self._last_forensic = self._goal if self._rng.random() < 0.88 else 1 - self._goal
        if self._last_alert is not None:
            self._belief_log_odds += self._evidence_increment(self._last_alert, 0.7)
        if self._last_forensic is not None:
            self._belief_log_odds += self._evidence_increment(self._last_forensic, 0.88)

    def _observation(self) -> DefenderObservation:
        alerts = np.zeros(3, dtype=np.float32)
        forensics = np.zeros(2, dtype=np.float32)
        alerts[2 if self._last_alert is None else self._last_alert] = 1.0
        if self._last_forensic is not None:
            forensics[self._last_forensic] = 1.0
        probability_b = float(1 / (1 + np.exp(-self._belief_log_odds)))
        vector = np.concatenate(
            (
                alerts,
                forensics,
                np.asarray(
                    [
                        self._step / self.horizon,
                        min(self._step / max(1, self.horizon - 2), 1.0),
                        self._cumulative_cost,
                    ],
                    dtype=np.float32,
                ),
            )
        ).astype(np.float32)
        return DefenderObservation(
            observation=vector,
            belief=(1 - probability_b, probability_b),
            action_mask=(not self._done,) * 4,
            step=self._step,
            cumulative_cost=self._cumulative_cost,
            committed_goal=self._committed_goal,
            source=self.name,
        )

    def step(self, action: DefenderAction | int) -> DefenderTransition:
        if self._done:
            raise RuntimeError("Call reset() before stepping a finished episode")
        action = DefenderAction(int(action))
        cost = self.costs[action]
        committed = action in (DefenderAction.COMMIT_A, DefenderAction.COMMIT_B)
        if committed:
            self._committed_goal = int(action) - int(DefenderAction.COMMIT_A)
        else:
            self._sense(action)
        self._cumulative_cost += cost
        self._last_action = action
        self._step += 1
        terminated = bool(committed)
        truncated = bool(not terminated and self._step >= self.horizon)
        self._done = terminated or truncated
        success = bool(committed and self._committed_goal == self._goal) if self._done else None
        if truncated:
            success = False
        observation = self._observation()
        info: dict[str, Any] = {"scenario": self.scenario}
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
        self._done = True
