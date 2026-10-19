"""Classical Bayesian and finite-horizon POMDP baselines for Stage 12."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from deception_rl.envs.gridworld import GUARD_A, GUARD_B, OBSERVE, TRACK


MOVES = np.asarray([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])


def binary_entropy(probability: float) -> float:
    p = float(np.clip(probability, 1e-12, 1 - 1e-12))
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


class TrajectoryBayesFilter:
    """Bayes filter under a soft shortest-path motion likelihood.

    Hidden state is the true goal. Timed position observations may have gaps;
    likelihoods marginalize all motion paths during the unobserved interval.
    """

    def __init__(self, rationality: float = 2.0) -> None:
        if rationality <= 0:
            raise ValueError("rationality must be positive")
        self.rationality = float(rationality)
        self.reset()

    def reset(self) -> None:
        self.belief = np.asarray([0.5, 0.5], dtype=np.float64)
        self.last_time: int | None = None
        self.last_position: np.ndarray | None = None
        self.deception_probability = 0.0
        self.processed_times: set[int] = set()

    def _step_distribution(
        self, distribution: dict[tuple[int, int], float], target: np.ndarray
    ) -> dict[tuple[int, int], float]:
        result: dict[tuple[int, int], float] = {}
        for state, mass in distribution.items():
            position = np.asarray(state, dtype=np.int64)
            candidates = np.clip(position + MOVES, 0, 6)
            current_distance = np.abs(target - position).sum()
            reductions = current_distance - np.abs(candidates - target).sum(axis=1)
            weights = np.exp(self.rationality * reductions)
            weights /= weights.sum()
            for candidate, probability in zip(candidates, weights):
                key = (int(candidate[0]), int(candidate[1]))
                result[key] = result.get(key, 0.0) + mass * float(probability)
        return result

    def _likelihood(
        self, start: np.ndarray, end: np.ndarray, gap: int, target: np.ndarray
    ) -> tuple[float, float]:
        distribution = {(int(start[0]), int(start[1])): 1.0}
        for _ in range(gap):
            distribution = self._step_distribution(distribution, target)
        observed = distribution.get((int(end[0]), int(end[1])), 0.0)
        peak = max(distribution.values())
        return max(observed, 1e-12), max(peak, 1e-12)

    def update(self, observation: dict[str, np.ndarray], timestep: int) -> np.ndarray:
        del timestep
        mask = np.asarray(observation["trajectory_mask"], dtype=bool)
        targets = np.asarray(observation["target_positions"], dtype=np.int64)
        trajectory = np.asarray(observation["trajectory"], dtype=np.int64)
        for time in np.flatnonzero(mask):
            time = int(time)
            if time in self.processed_times:
                continue
            position = trajectory[time]
            if self.last_time is not None and time > self.last_time:
                gap = time - self.last_time
                likelihoods = []
                normalized = []
                for target in targets:
                    likelihood, peak = self._likelihood(
                        self.last_position, position, gap, target
                    )
                    likelihoods.append(likelihood)
                    normalized.append(likelihood / peak)
                likelihoods = np.asarray(likelihoods)
                predictive_fit = float(np.dot(self.belief, np.asarray(normalized)))
                posterior = self.belief * likelihoods
                self.belief = posterior / posterior.sum()
                anomaly = 1.0 - np.clip(predictive_fit, 0.0, 1.0)
                self.deception_probability = float(
                    0.6 * self.deception_probability + 0.4 * anomaly
                )
            self.last_time = time
            self.last_position = position.copy()
            self.processed_times.add(time)
        return self.belief.copy()

    @property
    def entropy(self) -> float:
        return binary_entropy(float(self.belief[0]))


class BayesianDefender:
    """Sequential Bayes goal inference with observe-or-commit actions."""

    def __init__(self, confidence_threshold: float = 0.8, commit_deadline: int = 12):
        self.confidence_threshold = float(confidence_threshold)
        self.commit_deadline = int(commit_deadline)
        self.filter = TrajectoryBayesFilter()
        self.reset()

    def reset(self):
        self.filter.reset()
        self.goal = None
        self.commit_timestep = None
        self.info = None

    def act(self, observation, timestep):
        belief = self.filter.update(observation, timestep)
        confidence = float(belief.max())
        self.info = {
            "goal_belief": belief.tolist(),
            "confidence": confidence,
            "entropy": self.filter.entropy,
            "deception_probability": self.filter.deception_probability,
            "bayes_model": "soft_shortest_path",
        }
        if self.goal is not None:
            return self.goal
        if timestep >= self.commit_deadline or (
            timestep >= 5 and confidence >= self.confidence_threshold
        ):
            self.goal = int(np.argmax(belief))
            self.commit_timestep = timestep
            return self.goal
        return OBSERVE


class FiniteHorizonPOMDPDefender:
    """Small belief-state POMDP solved by exact finite-horizon recursion."""

    def __init__(
        self,
        horizon: int = 3,
        decision_deadline: int = 10,
        observe_accuracy: float = 0.68,
        track_accuracy: float = 0.82,
        observe_cost: float = 0.01,
        track_cost: float = 0.03,
        false_commitment_penalty: float = 0.5,
    ) -> None:
        if horizon < 1 or decision_deadline < 1:
            raise ValueError("horizon and decision_deadline must be positive")
        if not 0.5 < observe_accuracy <= track_accuracy < 1:
            raise ValueError("sensor accuracies must satisfy .5 < observe <= track < 1")
        self.horizon = int(horizon)
        self.decision_deadline = int(decision_deadline)
        self.accuracy = {OBSERVE: float(observe_accuracy), TRACK: float(track_accuracy)}
        self.cost = {OBSERVE: float(observe_cost), TRACK: float(track_cost)}
        self.false_commitment_penalty = float(false_commitment_penalty)
        self.filter = TrajectoryBayesFilter()
        self.reset()

    def reset(self):
        self.filter.reset()
        self.goal = None
        self.commit_timestep = None
        self.info = None
        self._value.cache_clear()

    def _commit_value(self, probability_a: float, goal: int) -> float:
        correct = probability_a if goal == 0 else 1 - probability_a
        return correct - self.false_commitment_penalty * (1 - correct)

    @staticmethod
    def _posterior(probability_a: float, accuracy: float, signal_a: bool) -> float:
        if signal_a:
            numerator = probability_a * accuracy
            denominator = numerator + (1 - probability_a) * (1 - accuracy)
        else:
            numerator = probability_a * (1 - accuracy)
            denominator = numerator + (1 - probability_a) * accuracy
        return numerator / denominator

    @lru_cache(maxsize=8192)
    def _value(self, rounded_probability_a: float, horizon: int) -> float:
        probability_a = float(rounded_probability_a)
        commit = max(
            self._commit_value(probability_a, 0),
            self._commit_value(probability_a, 1),
        )
        if horizon <= 0:
            return commit
        sensing = []
        for action in (OBSERVE, TRACK):
            accuracy = self.accuracy[action]
            probability_signal_a = (
                probability_a * accuracy + (1 - probability_a) * (1 - accuracy)
            )
            posterior_a = round(
                self._posterior(probability_a, accuracy, True), 6
            )
            posterior_b = round(
                self._posterior(probability_a, accuracy, False), 6
            )
            expected = (
                probability_signal_a * self._value(posterior_a, horizon - 1)
                + (1 - probability_signal_a) * self._value(posterior_b, horizon - 1)
                - self.cost[action]
            )
            sensing.append(expected)
        return max(commit, *sensing)

    def action_values(self, probability_a: float, horizon: int) -> dict[int, float]:
        values = {
            GUARD_A: self._commit_value(probability_a, 0),
            GUARD_B: self._commit_value(probability_a, 1),
        }
        for action in (OBSERVE, TRACK):
            accuracy = self.accuracy[action]
            probability_signal_a = (
                probability_a * accuracy + (1 - probability_a) * (1 - accuracy)
            )
            posterior_a = round(
                self._posterior(probability_a, accuracy, True), 6
            )
            posterior_b = round(
                self._posterior(probability_a, accuracy, False), 6
            )
            values[action] = (
                probability_signal_a * self._value(posterior_a, horizon - 1)
                + (1 - probability_signal_a) * self._value(posterior_b, horizon - 1)
                - self.cost[action]
            )
        return values

    def act(self, observation, timestep):
        belief = self.filter.update(observation, timestep)
        probability_a = float(belief[0])
        remaining = max(0, self.decision_deadline - timestep)
        horizon = min(self.horizon, remaining)
        values = self.action_values(probability_a, horizon)
        if remaining == 0:
            action = GUARD_A if probability_a >= 0.5 else GUARD_B
        else:
            action = max(values, key=values.get)
        self.info = {
            "goal_belief": belief.tolist(),
            "confidence": float(belief.max()),
            "entropy": self.filter.entropy,
            "deception_probability": self.filter.deception_probability,
            "pomdp_horizon": horizon,
            "action_values": {
                ("commit_A", "commit_B", "observe", "track")[key]: value
                for key, value in values.items()
            },
        }
        if self.goal is not None:
            return self.goal
        if action in (GUARD_A, GUARD_B):
            self.goal = action
            self.commit_timestep = timestep
            return action
        return action

