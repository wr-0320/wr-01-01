"""PettingZoo signal-deception game and its common Defender adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from deception_rl.external_envs.interface import (
    DefenderAction,
    DefenderObservation,
    DefenderTransition,
)


class SignalDeceptionParallelEnv(ParallelEnv):
    """A non-spatial two-agent deception game using the PettingZoo API.

    The Deceiver knows the hidden target and emits A, B, or masked signals.
    The Defender can observe, track for a noisy forensic cue, or commit.  The
    hidden goal never appears in the Defender observation.
    """

    metadata = {"name": "signal_deception_v0", "render_modes": []}

    def __init__(self, horizon: int = 8, signal_noise: float = 0.1, audit_accuracy: float = 0.85):
        if horizon < 2:
            raise ValueError("horizon must be at least two")
        if not 0 <= signal_noise < 0.5 or not 0.5 < audit_accuracy <= 1:
            raise ValueError("invalid signal or audit accuracy")
        self.horizon = int(horizon)
        self.signal_noise = float(signal_noise)
        self.audit_accuracy = float(audit_accuracy)
        self.possible_agents = ["deceiver", "defender"]
        self.agents: list[str] = []
        self._action_spaces = {
            "deceiver": spaces.Discrete(3),  # claim A, claim B, mask
            "defender": spaces.Discrete(4),
        }
        self._observation_spaces = {
            "deceiver": spaces.Dict(
                {
                    "goal": spaces.Discrete(2),
                    "step": spaces.Discrete(self.horizon + 1),
                    "defender_last_action": spaces.Discrete(5),
                }
            ),
            "defender": spaces.Dict(
                {
                    "signal_history": spaces.Box(-1, 2, (self.horizon,), dtype=np.int8),
                    "signal_mask": spaces.MultiBinary(self.horizon),
                    "audit_history": spaces.Box(-1, 1, (self.horizon,), dtype=np.int8),
                    "audit_mask": spaces.MultiBinary(self.horizon),
                    "step": spaces.Discrete(self.horizon + 1),
                }
            ),
        }
        self.np_random, _ = seeding.np_random(None)
        self._goal = 0
        self._step = 0
        self._last_defender_action = 4
        self._signals = np.full(self.horizon, -1, dtype=np.int8)
        self._signal_mask = np.zeros(self.horizon, dtype=np.int8)
        self._audits = np.full(self.horizon, -1, dtype=np.int8)
        self._audit_mask = np.zeros(self.horizon, dtype=np.int8)

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def reset(self, seed=None, options: Mapping[str, Any] | None = None):
        self.np_random, _ = seeding.np_random(seed)
        requested = (options or {}).get("goal")
        if requested is None:
            self._goal = int(self.np_random.integers(0, 2))
        elif requested in (0, "A"):
            self._goal = 0
        elif requested in (1, "B"):
            self._goal = 1
        else:
            raise ValueError("options['goal'] must be A, B, 0, or 1")
        self.agents = self.possible_agents.copy()
        self._step = 0
        self._last_defender_action = 4
        self._signals.fill(-1)
        self._signal_mask.fill(0)
        self._audits.fill(-1)
        self._audit_mask.fill(0)
        observations = self._observations()
        return observations, {agent: {} for agent in self.agents}

    def step(self, actions: Mapping[str, int]):
        if not self.agents:
            raise RuntimeError("Call reset() before stepping a finished episode")
        if set(actions) != set(self.agents):
            raise ValueError("Actions must be supplied for both agents")
        deceiver_action = int(actions["deceiver"])
        defender_action = int(actions["defender"])
        if not self._action_spaces["deceiver"].contains(deceiver_action):
            raise ValueError("Invalid Deceiver action")
        if not self._action_spaces["defender"].contains(defender_action):
            raise ValueError("Invalid Defender action")

        signal = deceiver_action
        if signal in (0, 1) and self.np_random.random() < self.signal_noise:
            signal = 1 - signal
        self._signals[self._step] = signal
        self._signal_mask[self._step] = 1
        if defender_action == DefenderAction.TRACK:
            cue = self._goal if self.np_random.random() < self.audit_accuracy else 1 - self._goal
            self._audits[self._step] = cue
            self._audit_mask[self._step] = 1

        committed = defender_action in (DefenderAction.COMMIT_A, DefenderAction.COMMIT_B)
        committed_goal = defender_action - int(DefenderAction.COMMIT_A) if committed else None
        success = bool(committed and committed_goal == self._goal)
        self._last_defender_action = defender_action
        self._step += 1
        terminated = bool(committed)
        truncated = bool(not terminated and self._step >= self.horizon)
        rewards = {"deceiver": 0.0, "defender": 0.0}
        if terminated:
            rewards["defender"] = 1.0 if success else -1.0
            rewards["deceiver"] = -rewards["defender"]
        elif truncated:
            rewards = {"deceiver": 1.0, "defender": -1.0}
        observations = self._observations()
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        terminal_info = (
            {"success": success, "true_goal": self._goal, "committed_goal": committed_goal}
            if terminated
            else {"success": False, "true_goal": self._goal, "committed_goal": None}
            if truncated
            else {}
        )
        infos = {agent: dict(terminal_info) for agent in self.agents}
        if terminated or truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def _observations(self):
        return {
            "deceiver": {
                "goal": self._goal,
                "step": self._step,
                "defender_last_action": self._last_defender_action,
            },
            "defender": {
                "signal_history": self._signals.copy(),
                "signal_mask": self._signal_mask.copy(),
                "audit_history": self._audits.copy(),
                "audit_mask": self._audit_mask.copy(),
                "step": self._step,
            },
        }

    def render(self):
        return None


class PettingZooDeceptionAdapter:
    """Expose the signal game through the shared Defender interface."""

    name = "pettingzoo_signal_deception"
    observation_size = 8
    action_space = spaces.Discrete(4)

    def __init__(
        self,
        attacker_strategy: str = "deceptive",
        horizon: int = 8,
        observe_cost: float = 0.01,
        track_cost: float = 0.03,
    ) -> None:
        if attacker_strategy not in {"honest", "deceptive", "adaptive"}:
            raise ValueError("Unknown attacker strategy")
        self.attacker_strategy = attacker_strategy
        self.horizon = int(horizon)
        self.costs = {
            DefenderAction.OBSERVE: float(observe_cost),
            DefenderAction.TRACK: float(track_cost),
            DefenderAction.COMMIT_A: 0.0,
            DefenderAction.COMMIT_B: 0.0,
        }
        if min(self.costs.values()) < 0:
            raise ValueError("Action costs must be nonnegative")
        self._env = SignalDeceptionParallelEnv(horizon=horizon)
        self._rng = np.random.default_rng()
        self._raw = None
        self._belief_log_odds = 0.0
        self._cumulative_cost = 0.0
        self._committed_goal = None
        self._last_action = None
        self._done = True

    def reset(self, seed=None, options: Mapping[str, Any] | None = None):
        self._rng = np.random.default_rng(seed)
        observations, _ = self._env.reset(seed=seed, options=options)
        self._raw = observations
        self._belief_log_odds = 0.0
        self._cumulative_cost = 0.0
        self._committed_goal = None
        self._last_action = None
        self._done = False
        return self._observation()

    def _attacker_action(self) -> int:
        hidden = self._raw["deceiver"]
        goal = int(hidden["goal"])
        step = int(hidden["step"])
        last_defender = int(hidden["defender_last_action"])
        if self.attacker_strategy == "honest":
            return goal
        if self.attacker_strategy == "deceptive":
            return 1 - goal if step < 4 else goal
        if last_defender == DefenderAction.TRACK:
            return 2 if step % 2 == 0 else 1 - goal
        return 1 - goal if step < 6 else goal

    @staticmethod
    def _evidence_increment(cue: int, reliability: float) -> float:
        magnitude = float(np.log(reliability / (1 - reliability)))
        return magnitude if cue == 1 else -magnitude

    def _update_belief(self, index: int) -> None:
        defender = self._raw["defender"]
        signal = int(defender["signal_history"][index])
        if signal in (0, 1):
            self._belief_log_odds += self._evidence_increment(signal, 0.7)
        if defender["audit_mask"][index]:
            cue = int(defender["audit_history"][index])
            self._belief_log_odds += self._evidence_increment(cue, 0.85)

    def _observation(self) -> DefenderObservation:
        defender = self._raw["defender"]
        step = int(defender["step"])
        signal_features = np.zeros(3, dtype=np.float32)
        audit_features = np.zeros(2, dtype=np.float32)
        if step:
            signal = int(defender["signal_history"][step - 1])
            signal_features[signal] = 1.0
            if defender["audit_mask"][step - 1]:
                audit_features[int(defender["audit_history"][step - 1])] = 1.0
        probability_b = float(1 / (1 + np.exp(-self._belief_log_odds)))
        vector = np.concatenate(
            (
                signal_features,
                audit_features,
                np.asarray(
                    [
                        step / self.horizon,
                        0.0 if self._last_action is None else int(self._last_action) / 3,
                        self._cumulative_cost,
                    ],
                    dtype=np.float32,
                ),
            )
        ).astype(np.float32)
        mask = (not self._done,) * 4
        return DefenderObservation(
            observation=vector,
            belief=(1 - probability_b, probability_b),
            action_mask=mask,
            step=step,
            cumulative_cost=self._cumulative_cost,
            committed_goal=self._committed_goal,
            source=self.name,
        )

    def step(self, action: DefenderAction | int) -> DefenderTransition:
        if self._done:
            raise RuntimeError("Call reset() before stepping a finished episode")
        action = DefenderAction(int(action))
        cost = self.costs[action]
        observations, _, terminations, truncations, infos = self._env.step(
            {"deceiver": self._attacker_action(), "defender": int(action)}
        )
        self._raw = observations
        self._update_belief(int(observations["defender"]["step"]) - 1)
        self._cumulative_cost += cost
        self._last_action = action
        if action in (DefenderAction.COMMIT_A, DefenderAction.COMMIT_B):
            self._committed_goal = int(action) - int(DefenderAction.COMMIT_A)
        terminated = bool(terminations["defender"])
        truncated = bool(truncations["defender"])
        self._done = terminated or truncated
        success = bool(infos["defender"]["success"]) if self._done else None
        observation = self._observation()
        return DefenderTransition(
            observation=observation,
            belief=observation.belief,
            action=action,
            cost=cost,
            success=success,
            terminated=terminated,
            truncated=truncated,
            info={
                "attacker_strategy": self.attacker_strategy,
                **infos["defender"],
            },
        )

    def close(self) -> None:
        self._env.close()
        self._done = True
