"""A simple Defender that chooses sensing or guard actions from belief."""

from __future__ import annotations

import numpy as np

from deception_rl.beliefs import BeliefEstimator, default_belief_estimator
from deception_rl.envs.gridworld import GUARD_A, GUARD_B, OBSERVE, TRACK


class BeliefBasedDefender:
    """Use belief entropy to trade off sensing against guarding a goal."""

    def __init__(
        self,
        estimator: BeliefEstimator | None = None,
        observe_entropy: float = 0.75,
        track_entropy: float = 0.95,
    ) -> None:
        if not 0 <= observe_entropy <= track_entropy <= 1:
            raise ValueError("Entropy thresholds must satisfy 0 <= observe <= track <= 1.")
        self.estimator = estimator if estimator is not None else default_belief_estimator()
        self.observe_entropy = observe_entropy
        self.track_entropy = track_entropy
        self.last_belief = {"A": 0.5, "B": 0.5}

    def estimate_belief(self, observation: dict[str, np.ndarray]) -> dict[str, float]:
        self.last_belief = self.estimator.estimate(
            observation["trajectory"],
            observation["trajectory_mask"],
            observation["target_positions"],
        )
        total = self.last_belief["A"] + self.last_belief["B"]
        if total <= 0 or not np.isfinite(total):
            raise ValueError("Belief estimator must return finite positive mass.")
        self.last_belief = {
            "A": self.last_belief["A"] / total,
            "B": self.last_belief["B"] / total,
        }
        return self.last_belief.copy()

    def act(self, observation: dict[str, np.ndarray]) -> int:
        belief = self.estimate_belief(observation)
        probabilities = np.array([belief["A"], belief["B"]], dtype=np.float64)
        positive = probabilities[probabilities > 0]
        entropy = float(-(positive * np.log2(positive)).sum())

        if entropy >= self.track_entropy:
            return TRACK
        if entropy >= self.observe_entropy:
            return OBSERVE
        return GUARD_A if belief["A"] >= belief["B"] else GUARD_B


class CommittedDefender:
    """Gather observations until a fixed step, then lock one guard decision."""

    def __init__(
        self,
        step_commit: int,
        defender: BeliefBasedDefender | None = None,
    ) -> None:
        if step_commit < 0:
            raise ValueError("step_commit must be non-negative.")
        self.step_commit = step_commit
        self.defender = defender or BeliefBasedDefender()
        self.reset()

    def reset(self) -> None:
        self._observation_step = 0
        self.decision: int | None = None
        self.belief_at_commitment: dict[str, float] | None = None

    def act(self, observation: dict[str, np.ndarray]) -> int:
        if self.decision is not None:
            return self.decision

        belief = self.defender.estimate_belief(observation)
        if self._observation_step >= self.step_commit:
            self.belief_at_commitment = belief.copy()
            self.decision = GUARD_A if belief["A"] >= belief["B"] else GUARD_B
            action = self.decision
        else:
            probabilities = np.array([belief["A"], belief["B"]], dtype=np.float64)
            positive = probabilities[probabilities > 0]
            entropy = float(-(positive * np.log2(positive)).sum())
            action = TRACK if entropy >= self.defender.track_entropy else OBSERVE

        self._observation_step += 1
        return action
