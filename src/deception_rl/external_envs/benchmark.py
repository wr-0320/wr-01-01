"""Stage 17.3 attack adapters layered over the frozen environments."""

from __future__ import annotations

import numpy as np

from deception_rl.envs import ForkedRoutesAdapter, GridWorldAdapter
from deception_rl.external_envs.cyber import CyberDeceptionEnv
from deception_rl.external_envs.navigation import NavigationDefenderAdapter
from deception_rl.external_envs.pettingzoo_adapter import PettingZooDeceptionAdapter
from experiments.stage15_attackers import AdaptiveDeceiverV2, RouteAwareFamily


ATTACK_SETTINGS = ("scripted", "adaptive", "unseen_family")


class BenchmarkNavigationAdapter(NavigationDefenderAdapter):
    """Use fixed Stage 15 attackers without changing either navigation map."""

    def __init__(self, backend: str, attack_setting: str) -> None:
        if backend not in {"gridworld", "forked_routes"}:
            raise ValueError("Unknown navigation backend")
        if attack_setting not in ATTACK_SETTINGS:
            raise ValueError("Unknown benchmark attack setting")
        adapter = GridWorldAdapter() if backend == "gridworld" else ForkedRoutesAdapter()
        super().__init__(adapter, "honest")
        self.backend = backend
        self.attack_setting = attack_setting
        self.name = backend

    def _make_attacker(self):
        if self.attack_setting == "scripted":
            return RouteAwareFamily("scripted_3", self.model)
        if self.attack_setting == "adaptive":
            return AdaptiveDeceiverV2(self.model)
        return RouteAwareFamily("stochastic_deception", self.model)

    def reset(self, seed=None, options=None):
        observation = super().reset(seed=seed, options=options)
        self._previous_public_action = None
        self._previous_visible = False
        return observation

    def step(self, action):
        if self.attack_setting == "adaptive" and self._previous_public_action is not None:
            self._attacker.observe_defender(
                self._previous_public_action,
                max(0, self._step - 1),
                self._previous_visible,
            )
        transition = super().step(action)
        self._previous_public_action = self.ACTION_MAP[transition.action]
        trajectory_mask = np.asarray(self._raw["defender"]["trajectory_mask"], dtype=bool)
        self._previous_visible = bool(trajectory_mask.any())
        return transition


class BenchmarkPettingZooAdapter(PettingZooDeceptionAdapter):
    """Map the benchmark's three semantic families to signal strategies."""

    def __init__(self, attack_setting: str) -> None:
        if attack_setting not in ATTACK_SETTINGS:
            raise ValueError("Unknown benchmark attack setting")
        super().__init__("adaptive" if attack_setting == "adaptive" else "deceptive")
        self.attack_setting = attack_setting
        self.name = "pettingzoo_deception"

    def _attacker_action(self) -> int:
        if self.attack_setting != "unseen_family":
            return super()._attacker_action()
        hidden = self._raw["deceiver"]
        goal = int(hidden["goal"])
        step = int(hidden["step"])
        if step >= 6:
            return goal
        draw = float(self._rng.random())
        if draw < 0.45:
            return 1 - goal
        if draw < 0.75:
            return goal
        return 2


class BenchmarkCyberAdapter(CyberDeceptionEnv):
    """Expose scripted, observation-adaptive, and stochastic cyber attacks."""

    def __init__(self, attack_setting: str) -> None:
        if attack_setting not in ATTACK_SETTINGS:
            raise ValueError("Unknown benchmark attack setting")
        scenario = "credential_decoy" if attack_setting == "scripted" else "lateral_switch"
        super().__init__(scenario)
        self.attack_setting = attack_setting
        self.name = "cyber_deception"

    def _intended_alert(self) -> int | None:
        if self.attack_setting == "scripted":
            return super()._intended_alert()
        if self.attack_setting == "adaptive":
            if self._step >= 6:
                return self._goal
            if self._last_action is None:
                return 1 - self._goal
            # React only to the completed previous Defender action.
            if int(self._last_action) == 1:  # track
                return None if self._step % 2 == 0 else 1 - self._goal
            return 1 - self._goal
        if self._step >= 7:
            return self._goal
        draw = float(self._rng.random())
        if draw < 0.45:
            return 1 - self._goal
        if draw < 0.75:
            return self._goal
        return None

