"""Environment-aware formal Bayesian and finite-horizon POMDP baselines.

These are new Stage 14 baselines.  They preserve the Stage 12 assumptions but
make the transition model explicit, perform filtering in log space, support
arbitrary adapters, and expose the complete decision model for reproduction.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from deception_rl.envs.adapters import GridWorldAdapter
from deception_rl.envs.gridworld import GUARD_A, GUARD_B, OBSERVE, TRACK
from deception_rl.envs.interface import DiscreteEnvironmentModel, Position


def binary_entropy(probability_a: float) -> float:
    probability_a = float(np.clip(probability_a, 1e-12, 1 - 1e-12))
    return float(
        -(probability_a * np.log2(probability_a) + (1 - probability_a) * np.log2(1 - probability_a))
    )


class FormalTrajectoryBayesFilter:
    """Exact finite-state Bayes filter under a Boltzmann-rational mover."""

    def __init__(
        self,
        model: DiscreteEnvironmentModel | None = None,
        rationality: float = 2.0,
        likelihood_floor: float = 1e-12,
    ) -> None:
        if rationality <= 0 or likelihood_floor <= 0:
            raise ValueError("rationality and likelihood_floor must be positive")
        self.model = model or GridWorldAdapter().model
        self.rationality = float(rationality)
        self.likelihood_floor = float(likelihood_floor)
        self._distances = tuple(self.model.distances_to(goal) for goal in self.model.goals)
        self.reset()

    def reset(self) -> None:
        self.log_belief = np.log(np.asarray([0.5, 0.5], dtype=np.float64))
        self.last_time: int | None = None
        self.last_position: Position | None = None
        self.processed_times: set[int] = set()
        self.deception_probability = 0.0
        self.last_predictive_likelihood = 1.0

    @property
    def belief(self) -> np.ndarray:
        shifted = self.log_belief - np.max(self.log_belief)
        values = np.exp(shifted)
        return values / values.sum()

    @property
    def entropy(self) -> float:
        return binary_entropy(float(self.belief[0]))

    @lru_cache(maxsize=32768)
    def _one_step(self, state: Position, goal_index: int) -> tuple[tuple[Position, float], ...]:
        distances = self._distances[goal_index]
        weights = []
        successors = []
        for action in range(len(self.model.action_deltas)):
            successor = self.model.transition(state, action)
            successors.append(successor)
            weights.append(np.exp(-self.rationality * distances[successor]))
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()
        aggregated: dict[Position, float] = {}
        for successor, probability in zip(successors, weights):
            aggregated[successor] = aggregated.get(successor, 0.0) + float(probability)
        return tuple(sorted(aggregated.items()))

    def transition_distribution(
        self, start: Position, gap: int, goal_index: int
    ) -> dict[Position, float]:
        if gap < 0:
            raise ValueError("gap must be nonnegative")
        distribution = {start: 1.0}
        for _ in range(gap):
            next_distribution: dict[Position, float] = {}
            for state, mass in distribution.items():
                for successor, probability in self._one_step(state, goal_index):
                    next_distribution[successor] = (
                        next_distribution.get(successor, 0.0) + mass * probability
                    )
            distribution = next_distribution
        return distribution

    def update(self, observation: dict[str, np.ndarray], timestep: int) -> np.ndarray:
        del timestep
        trajectory = np.asarray(observation["trajectory"], dtype=np.int64)
        mask = np.asarray(observation["trajectory_mask"], dtype=bool)
        for raw_time in np.flatnonzero(mask):
            time = int(raw_time)
            if time in self.processed_times:
                continue
            position = tuple(int(value) for value in trajectory[time])
            if position not in self.model.traversable:
                raise ValueError(f"Observed position {position} is outside the environment model")
            if self.last_time is not None and time > self.last_time:
                gap = time - self.last_time
                likelihoods = np.asarray(
                    [
                        max(
                            self.transition_distribution(self.last_position, gap, goal).get(position, 0.0),
                            self.likelihood_floor,
                        )
                        for goal in range(2)
                    ],
                    dtype=np.float64,
                )
                prior = self.belief
                predictive = float(np.dot(prior, likelihoods))
                peak = max(
                    max(self.transition_distribution(self.last_position, gap, goal).values())
                    for goal in range(2)
                )
                self.last_predictive_likelihood = predictive
                normalized_fit = float(np.clip(predictive / max(peak, self.likelihood_floor), 0, 1))
                self.deception_probability = float(
                    0.6 * self.deception_probability + 0.4 * (1 - normalized_fit)
                )
                self.log_belief += np.log(likelihoods)
                self.log_belief -= np.logaddexp.reduce(self.log_belief)
            self.last_time = time
            self.last_position = position
            self.processed_times.add(time)
        return self.belief.copy()


class FormalBayesianDefender:
    """Sequential Bayes decision rule with an explicit stopping boundary."""

    def __init__(
        self,
        model: DiscreteEnvironmentModel | None = None,
        confidence_threshold: float = 0.8,
        earliest_commit: int = 5,
        decision_deadline: int = 12,
    ) -> None:
        if not 0.5 < confidence_threshold < 1:
            raise ValueError("confidence_threshold must lie between .5 and 1")
        self.filter = FormalTrajectoryBayesFilter(model)
        self.confidence_threshold = float(confidence_threshold)
        self.earliest_commit = int(earliest_commit)
        self.decision_deadline = int(decision_deadline)
        self.reset()

    def reset(self) -> None:
        self.filter.reset()
        self.goal = None
        self.commit_timestep = None
        self.info = None

    def act(self, observation, timestep):
        belief = self.filter.update(observation, timestep)
        confidence = float(belief.max())
        log_odds = float(np.log(max(belief[0], 1e-12) / max(belief[1], 1e-12)))
        self.info = {
            "goal_belief": belief.tolist(),
            "confidence": confidence,
            "entropy": self.filter.entropy,
            "deception_probability": self.filter.deception_probability,
            "log_bayes_factor_A_over_B": log_odds,
            "decision_rule": "posterior_stopping_boundary",
        }
        if self.goal is not None:
            return self.goal
        should_commit = timestep >= self.decision_deadline or (
            timestep >= self.earliest_commit and confidence >= self.confidence_threshold
        )
        if should_commit:
            self.goal = int(np.argmax(belief))
            self.commit_timestep = timestep
            return self.goal
        return OBSERVE


class FormalFiniteHorizonPOMDPDefender:
    """Exact finite-horizon value iteration on the binary belief simplex."""

    def __init__(
        self,
        model: DiscreteEnvironmentModel | None = None,
        horizon: int = 5,
        decision_deadline: int = 10,
        observe_accuracy: float = 0.68,
        track_accuracy: float = 0.82,
        observe_cost: float = 0.01,
        track_cost: float = 0.03,
        interception_reward: float = 1.0,
        false_commitment_penalty: float = 0.5,
    ) -> None:
        if horizon < 1 or decision_deadline < 1:
            raise ValueError("horizon and decision_deadline must be positive")
        if not 0.5 < observe_accuracy <= track_accuracy < 1:
            raise ValueError("sensor accuracies must satisfy .5 < observe <= track < 1")
        self.filter = FormalTrajectoryBayesFilter(model)
        self.horizon = int(horizon)
        self.decision_deadline = int(decision_deadline)
        self.accuracy = {OBSERVE: float(observe_accuracy), TRACK: float(track_accuracy)}
        self.cost = {OBSERVE: float(observe_cost), TRACK: float(track_cost)}
        self.interception_reward = float(interception_reward)
        self.false_commitment_penalty = float(false_commitment_penalty)
        self.reset()

    def reset(self) -> None:
        self.filter.reset()
        self.goal = None
        self.commit_timestep = None
        self.info = None
        self._value.cache_clear()

    def _commit_value(self, probability_a: float, goal: int) -> float:
        correct = probability_a if goal == 0 else 1 - probability_a
        return self.interception_reward * correct - self.false_commitment_penalty * (1 - correct)

    @staticmethod
    def posterior(probability_a: float, accuracy: float, signal_a: bool) -> float:
        likelihood_a = accuracy if signal_a else 1 - accuracy
        likelihood_b = 1 - accuracy if signal_a else accuracy
        numerator = probability_a * likelihood_a
        return numerator / (numerator + (1 - probability_a) * likelihood_b)

    @lru_cache(maxsize=65536)
    def _value(self, rounded_probability_a: float, horizon: int) -> float:
        values = self.action_values(float(rounded_probability_a), horizon, recurse=False)
        return max(values.values())

    def action_values(
        self, probability_a: float, horizon: int, recurse: bool = True
    ) -> dict[int, float]:
        probability_a = float(np.clip(probability_a, 1e-9, 1 - 1e-9))
        values = {
            GUARD_A: self._commit_value(probability_a, 0),
            GUARD_B: self._commit_value(probability_a, 1),
        }
        if horizon <= 0:
            return values
        for action in (OBSERVE, TRACK):
            accuracy = self.accuracy[action]
            signal_a_probability = probability_a * accuracy + (1 - probability_a) * (1 - accuracy)
            posterior_a = round(self.posterior(probability_a, accuracy, True), 8)
            posterior_b = round(self.posterior(probability_a, accuracy, False), 8)
            child_horizon = horizon - 1
            if recurse:
                value_a = self._value(posterior_a, child_horizon)
                value_b = self._value(posterior_b, child_horizon)
            else:
                child_a = self.action_values(posterior_a, child_horizon, recurse=True)
                child_b = self.action_values(posterior_b, child_horizon, recurse=True)
                value_a = max(child_a.values())
                value_b = max(child_b.values())
            values[action] = (
                signal_a_probability * value_a
                + (1 - signal_a_probability) * value_b
                - self.cost[action]
            )
        return values

    def act(self, observation, timestep):
        belief = self.filter.update(observation, timestep)
        probability_a = float(belief[0])
        remaining = max(0, self.decision_deadline - timestep)
        horizon = min(self.horizon, remaining)
        values = self.action_values(probability_a, horizon)
        action = (
            GUARD_A if probability_a >= 0.5 else GUARD_B
        ) if remaining == 0 else max(values, key=values.get)
        names = {GUARD_A: "commit_A", GUARD_B: "commit_B", OBSERVE: "observe", TRACK: "track"}
        self.info = {
            "goal_belief": belief.tolist(),
            "confidence": float(belief.max()),
            "entropy": self.filter.entropy,
            "deception_probability": self.filter.deception_probability,
            "pomdp_horizon": horizon,
            "action_values": {names[key]: value for key, value in values.items()},
            "decision_rule": "exact_finite_horizon_belief_tree",
        }
        if self.goal is not None:
            return self.goal
        if action in (GUARD_A, GUARD_B):
            self.goal = action
            self.commit_timestep = timestep
        return action

