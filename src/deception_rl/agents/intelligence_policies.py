"""Transparent red and blue baselines for deceptive intelligence experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from deception_rl.envs.deceptive_intelligence import (
    BLUE_COMMIT_START,
    BLUE_TRACK_START,
    BLUE_WAIT,
    NUM_HYPOTHESES,
    NUM_RED_ACTIONS,
    NUM_REGIONS,
    NUM_SOURCES,
    OPERATIONAL_HYPOTHESES,
    RED_CONCEAL,
    RED_HONEST,
    RED_MULTISOURCE_START,
    decode_track_action,
    decode_verify_action,
    encode_commit_action,
    encode_decoy_action,
    encode_track_action,
    encode_verify_action,
)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponential = np.exp(shifted)
    return exponential / exponential.sum()


@dataclass(frozen=True)
class IntelligenceBelief:
    hypothesis_probabilities: np.ndarray
    deception_probability: float
    confidence: float
    entropy: float


class IntelligenceBeliefEstimator:
    """Likelihood-style filter using only blue-visible report history.

    The filter compares weighted multi-source evidence with operational
    templates.  It is deliberately interpretable and serves as a baseline,
    rather than claiming calibrated Bayesian optimality.
    """

    def __init__(self, temperature: float = 0.20) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    @staticmethod
    def hypothesis_templates() -> np.ndarray:
        templates = np.full(
            (NUM_HYPOTHESES, NUM_SOURCES, NUM_REGIONS), 0.10, dtype=np.float32
        )
        objective_sensitivity = np.array(
            [[0.95, 0.55, 0.70], [0.45, 0.98, 0.58], [0.72, 0.52, 0.92]],
            dtype=np.float32,
        )
        deception_susceptibility = np.array([0.70, 0.82, 0.45], dtype=np.float32)
        for index, hypothesis in enumerate(OPERATIONAL_HYPOTHESES):
            templates[index, :, hypothesis.main_region] += (
                0.58 * objective_sensitivity[:, hypothesis.objective]
            )
            if hypothesis.feint_region is not None:
                templates[index, :, hypothesis.main_region] *= 0.72
                templates[index, :, hypothesis.feint_region] += (
                    0.48 * deception_susceptibility
                )
        return np.clip(templates, 0.0, 1.0)

    def estimate(self, observation: dict[str, Any]) -> IntelligenceBelief:
        reports = np.asarray(observation["reports"], dtype=np.float32)
        mask = np.asarray(observation["report_mask"], dtype=bool)
        current_step = int(np.asarray(observation["current_step"])[0])
        if current_step == 0 or not mask.any():
            probabilities = np.full(NUM_HYPOTHESES, 1.0 / NUM_HYPOTHESES)
            return IntelligenceBelief(
                probabilities, 0.5, float(probabilities.max()), 1.0
            )

        source_reliability = np.asarray(
            observation["source_reliability"], dtype=np.float32
        )
        activity = reports[..., 0]
        confidence = reports[..., 1]
        normalized_age = reports[..., 2]
        verified = reports[..., 3]
        weights = (
            mask
            * confidence
            * (1.0 - 0.65 * normalized_age)
            * source_reliability[None, :, None]
            * (1.0 + 0.50 * verified)
        )
        recent_start = max(0, current_step - 5)
        weights[:recent_start] = 0
        denominator = weights.sum(axis=0)
        evidence = np.divide(
            (activity * weights).sum(axis=0),
            denominator,
            out=np.full((NUM_SOURCES, NUM_REGIONS), 0.10, dtype=np.float32),
            where=denominator > 0,
        )
        evidence_weight = np.maximum(denominator, 0.05)
        templates = self.hypothesis_templates()
        squared_error = ((templates - evidence[None, ...]) ** 2 * evidence_weight).sum(
            axis=(1, 2)
        ) / evidence_weight.sum()
        probabilities = _softmax(-squared_error / self.temperature)
        deceptive = np.array(
            [hypothesis.feint_region is not None for hypothesis in OPERATIONAL_HYPOTHESES]
        )
        entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
        entropy /= float(np.log(NUM_HYPOTHESES))
        return IntelligenceBelief(
            hypothesis_probabilities=probabilities,
            deception_probability=float(probabilities[deceptive].sum()),
            confidence=float(probabilities.max()),
            entropy=entropy,
        )


class HonestRedPolicy:
    name = "honest"

    def reset(self) -> None:
        return None

    def act(self, observation: dict[str, Any]) -> int:
        return RED_HONEST


class ConcealmentRedPolicy:
    name = "concealment"

    def reset(self) -> None:
        return None

    def act(self, observation: dict[str, Any]) -> int:
        mask = np.asarray(observation["action_mask"], dtype=bool)
        return RED_CONCEAL if mask[RED_CONCEAL] else RED_HONEST


class DecoyRedPolicy:
    name = "decoy"

    def __init__(self, multisource: bool = False) -> None:
        self.multisource = bool(multisource)
        if self.multisource:
            self.name = "multi_source_deception"

    def reset(self) -> None:
        return None

    def _decoy_region(self, observation: dict[str, Any]) -> int:
        hypothesis = OPERATIONAL_HYPOTHESES[int(observation["hypothesis"])]
        if hypothesis.feint_region is not None:
            return hypothesis.feint_region
        return (hypothesis.main_region + 1) % NUM_REGIONS

    def act(self, observation: dict[str, Any]) -> int:
        region = self._decoy_region(observation)
        action = encode_decoy_action(region, multisource=self.multisource)
        mask = np.asarray(observation["action_mask"], dtype=bool)
        if mask[action]:
            return action
        fallback = encode_decoy_action(region, multisource=False)
        return fallback if mask[fallback] else RED_HONEST


class DelayedDeceptionRedPolicy(DecoyRedPolicy):
    name = "delayed_deception"

    def __init__(self, onset: int = 3) -> None:
        super().__init__(multisource=True)
        if onset < 1:
            raise ValueError("onset must be positive")
        self.onset = int(onset)
        self.name = "delayed_deception"

    def act(self, observation: dict[str, Any]) -> int:
        step = int(np.asarray(observation["step"])[0])
        return RED_HONEST if step < self.onset else super().act(observation)


class SwitchDeceptionRedPolicy(DecoyRedPolicy):
    name = "switch_deception"

    def __init__(self, switch_step: int = 3) -> None:
        super().__init__(multisource=False)
        if switch_step < 1:
            raise ValueError("switch_step must be positive")
        self.switch_step = int(switch_step)
        self.name = "switch_deception"

    def act(self, observation: dict[str, Any]) -> int:
        main = int(observation["main_region"])
        false_regions = [region for region in range(NUM_REGIONS) if region != main]
        step = int(np.asarray(observation["step"])[0])
        region = false_regions[int(step >= self.switch_step)]
        action = encode_decoy_action(region)
        return action if observation["action_mask"][action] else RED_HONEST


class AdaptiveDeceptionRedPolicy(DecoyRedPolicy):
    """Rule-based deception that moves away from the last blue inspection."""

    name = "adaptive_rule_based_deception"

    def __init__(self) -> None:
        super().__init__(multisource=True)
        self._current_region = 0
        self.name = "adaptive_rule_based_deception"

    def reset(self) -> None:
        self._current_region = 0

    def act(self, observation: dict[str, Any]) -> int:
        main = int(observation["main_region"])
        hypothesis = OPERATIONAL_HYPOTHESES[int(observation["hypothesis"])]
        if self._current_region == main:
            self._current_region = (
                hypothesis.feint_region
                if hypothesis.feint_region is not None
                else (main + 1) % NUM_REGIONS
            )
        inspected: int | None = None
        verification = decode_verify_action(int(observation["last_blue_action"]))
        if verification is not None:
            inspected = verification[1]
        else:
            inspected = decode_track_action(int(observation["last_blue_action"]))
        if inspected == self._current_region:
            candidates = [
                region
                for region in range(NUM_REGIONS)
                if region not in {main, inspected}
            ]
            if candidates:
                self._current_region = candidates[0]
        action = encode_decoy_action(self._current_region, multisource=True)
        mask = np.asarray(observation["action_mask"], dtype=bool)
        if mask[action]:
            return action
        fallback = encode_decoy_action(self._current_region)
        return fallback if mask[fallback] else RED_HONEST


class FixedTimeBluePolicy:
    name = "fixed_time"

    def __init__(
        self,
        commit_step: int = 4,
        estimator: IntelligenceBeliefEstimator | None = None,
    ) -> None:
        if commit_step < 1:
            raise ValueError("commit_step must be positive")
        self.commit_step = int(commit_step)
        self.estimator = estimator or IntelligenceBeliefEstimator()
        self.last_belief: IntelligenceBelief | None = None

    def reset(self) -> None:
        self.last_belief = None

    def act(self, observation: dict[str, Any]) -> int:
        self.last_belief = self.estimator.estimate(observation)
        step = int(np.asarray(observation["current_step"])[0])
        if step >= self.commit_step:
            return encode_commit_action(
                int(np.argmax(self.last_belief.hypothesis_probabilities))
            )
        return BLUE_WAIT


class AdaptiveVerificationBluePolicy(FixedTimeBluePolicy):
    """Verify discriminating evidence, then commit when confidence is adequate."""

    name = "adaptive_verification"

    def __init__(
        self,
        min_commit_step: int = 2,
        confidence_threshold: float = 0.20,
        max_cycles: int = 10,
        estimator: IntelligenceBeliefEstimator | None = None,
    ) -> None:
        super().__init__(commit_step=min_commit_step, estimator=estimator)
        if not 0 < confidence_threshold < 1:
            raise ValueError("confidence_threshold must be in (0, 1)")
        self.confidence_threshold = float(confidence_threshold)
        self.max_cycles = int(max_cycles)

    def act(self, observation: dict[str, Any]) -> int:
        belief = self.estimator.estimate(observation)
        self.last_belief = belief
        probabilities = belief.hypothesis_probabilities
        ranking = np.argsort(probabilities)[::-1]
        step = int(np.asarray(observation["current_step"])[0])
        predicted = int(ranking[0])
        if step >= self.commit_step and (
            belief.confidence >= self.confidence_threshold
            or step >= self.max_cycles - 1
        ):
            return encode_commit_action(predicted)

        action_mask = np.asarray(observation["action_mask"], dtype=bool)
        if int(np.asarray(observation["remaining_verification"])[0]) > 0:
            templates = self.estimator.hypothesis_templates()
            disagreement = np.abs(templates[ranking[0]] - templates[ranking[1]])
            source, region = np.unravel_index(np.argmax(disagreement), disagreement.shape)
            action = encode_verify_action(int(source), int(region))
            if action_mask[action]:
                return action

        if int(np.asarray(observation["remaining_tracking"])[0]) > 0:
            region = OPERATIONAL_HYPOTHESES[predicted].main_region
            action = encode_track_action(region)
            if action_mask[action]:
                return action
        return BLUE_WAIT


def policy_action_name(action: int, agent: str) -> str:
    if agent == "red":
        if not 0 <= action < NUM_RED_ACTIONS:
            return "invalid"
        if action == RED_HONEST:
            return "honest"
        if action == RED_CONCEAL:
            return "conceal"
        if action >= RED_MULTISOURCE_START:
            return f"multi_source:{action - RED_MULTISOURCE_START}"
        return f"decoy:{action - 2}"
    if action == BLUE_WAIT:
        return "wait"
    verification = decode_verify_action(action)
    if verification is not None:
        return f"verify:{verification[0]}:{verification[1]}"
    tracked = decode_track_action(action)
    if tracked is not None:
        return f"track:{tracked}"
    committed = action - BLUE_COMMIT_START
    return f"commit:{committed}" if 0 <= committed < NUM_HYPOTHESES else "invalid"

