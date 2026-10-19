"""Classical collection and stopping baselines for the intelligence setting."""

from __future__ import annotations

from typing import Any

import numpy as np

from deception_rl.agents.intelligence_policies import IntelligenceBeliefEstimator
from deception_rl.envs.deceptive_intelligence import (
    BLUE_WAIT,
    NUM_HYPOTHESES,
    OPERATIONAL_HYPOTHESES,
    encode_commit_action,
    encode_track_action,
    encode_verify_action,
)


class BayesianFusionController:
    """Template-likelihood fusion with threshold stopping and active verification."""

    name = "bayesian_fusion"

    def __init__(self, max_cycles: int = 10, confidence_threshold: float = 0.24):
        self.max_cycles = int(max_cycles)
        self.confidence_threshold = float(confidence_threshold)
        self.estimator = IntelligenceBeliefEstimator()
        self.reset()

    def reset(self) -> None:
        self.info: dict[str, Any] | None = None

    def _update(self, observation: dict[str, Any]) -> np.ndarray:
        result = self.estimator.estimate(observation)
        belief = result.hypothesis_probabilities
        self.info = {
            "hypothesis_belief": belief.tolist(),
            "predicted_hypothesis": int(np.argmax(belief)),
            "confidence": result.confidence,
            "entropy": result.entropy,
            "deception_probability": result.deception_probability,
        }
        return belief

    def _verification(self, observation: dict[str, Any], belief: np.ndarray) -> int:
        templates = self.estimator.hypothesis_templates()
        ranking = np.argsort(belief)[::-1]
        difference = np.abs(templates[ranking[0]] - templates[ranking[1]])
        source, region = np.unravel_index(np.argmax(difference), difference.shape)
        if int(np.asarray(observation["remaining_verification"])[0]) > 0:
            return encode_verify_action(int(source), int(region))
        if int(np.asarray(observation["remaining_tracking"])[0]) > 0:
            return encode_track_action(
                OPERATIONAL_HYPOTHESES[int(ranking[0])].main_region
            )
        return BLUE_WAIT

    def act(self, observation: dict[str, Any]) -> int:
        belief = self._update(observation)
        predicted = int(np.argmax(belief))
        step = int(np.asarray(observation["current_step"])[0])
        if self.info["confidence"] >= self.confidence_threshold or step >= self.max_cycles - 1:
            return encode_commit_action(predicted)
        return self._verification(observation, belief)


class POMCPLikeController(BayesianFusionController):
    """One-step Monte Carlo belief rollout with explicit sensing assumptions.

    This is a compact POMCP-like baseline, not a solver for the full simulator.
    It samples latent hypotheses from the current belief and simulated categorical
    cues under fixed assumed sensing accuracies.
    """

    name = "pomcp_like"

    def __init__(self, max_cycles: int = 10, rollouts: int = 128, seed: int = 0):
        self.rollouts = int(rollouts)
        self.seed = int(seed)
        super().__init__(max_cycles=max_cycles)
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        super().reset()
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def _posterior_after_signal(
        prior: np.ndarray, signal: int, accuracy: float
    ) -> np.ndarray:
        likelihood = np.full(
            NUM_HYPOTHESES, (1 - accuracy) / (NUM_HYPOTHESES - 1)
        )
        likelihood[signal] = accuracy
        posterior = prior * likelihood
        return posterior / posterior.sum()

    def _rollout_value(self, belief: np.ndarray, accuracy: float, cost: float) -> float:
        values = []
        for _ in range(self.rollouts):
            truth = int(self.rng.choice(NUM_HYPOTHESES, p=belief))
            if self.rng.random() < accuracy:
                signal = truth
            else:
                alternatives = [index for index in range(NUM_HYPOTHESES) if index != truth]
                signal = int(self.rng.choice(alternatives))
            posterior = self._posterior_after_signal(belief, signal, accuracy)
            decision = int(np.argmax(posterior))
            values.append((10.0 if decision == truth else -10.0) - cost)
        return float(np.mean(values))

    def act(self, observation: dict[str, Any]) -> int:
        belief = self._update(observation)
        predicted = int(np.argmax(belief))
        step = int(np.asarray(observation["current_step"])[0])
        commit_value = 20.0 * float(belief[predicted]) - 10.0
        candidates = {"commit": commit_value}
        if int(np.asarray(observation["remaining_verification"])[0]) > 0:
            candidates["verify"] = self._rollout_value(belief, 0.72, 0.45)
        if int(np.asarray(observation["remaining_tracking"])[0]) > 0:
            candidates["track"] = self._rollout_value(belief, 0.64, 0.30)
        candidates["wait"] = self._rollout_value(belief, 0.54, 0.15)
        choice = max(candidates, key=candidates.get)
        self.info["action_utilities"] = candidates
        if choice == "commit" or step >= self.max_cycles - 1:
            return encode_commit_action(predicted)
        if choice == "verify":
            return self._verification(observation, belief)
        if choice == "track":
            region = OPERATIONAL_HYPOTHESES[predicted].main_region
            return encode_track_action(region)
        return BLUE_WAIT


class ActiveInferenceController(BayesianFusionController):
    """Choose observations by expected entropy reduction, risk, and resource cost."""

    name = "active_inference"

    def act(self, observation: dict[str, Any]) -> int:
        belief = self._update(observation)
        predicted = int(np.argmax(belief))
        step = int(np.asarray(observation["current_step"])[0])
        entropy = float(self.info["entropy"])
        deception = float(self.info["deception_probability"])
        remaining = self.max_cycles - step
        free_energy = {
            "commit": 10.0 * (1 - float(belief[predicted])) + 2.0 * deception,
            "wait": 3.0 * entropy + 0.15 + 0.12 / max(1, remaining),
        }
        if int(np.asarray(observation["remaining_verification"])[0]) > 0:
            free_energy["verify"] = 2.0 * entropy + 0.45 + 0.4 * deception
        if int(np.asarray(observation["remaining_tracking"])[0]) > 0:
            free_energy["track"] = 2.4 * entropy + 0.30 + 0.2 * deception
        choice = min(free_energy, key=free_energy.get)
        self.info["expected_free_energy"] = free_energy
        if choice == "commit" or step >= self.max_cycles - 1:
            return encode_commit_action(predicted)
        if choice == "verify":
            return self._verification(observation, belief)
        if choice == "track":
            return encode_track_action(
                OPERATIONAL_HYPOTHESES[predicted].main_region
            )
        return BLUE_WAIT

