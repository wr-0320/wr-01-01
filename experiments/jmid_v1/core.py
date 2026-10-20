"""Small controlled models for informative missingness and dependent evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GROUPS = ((0, 1), (2, 3), (4, 5))
METHODS = ("M1_naive", "M2_missing", "M3_dependence", "M4_joint")
CONDITIONS = {
    "A_random_independent": (0.0, 0.0),
    "B_strategic_independent": (1.0, 0.0),
    "C_random_dependent": (0.0, 0.9),
    "D_strategic_dependent": (1.0, 0.9),
}


@dataclass(frozen=True)
class Dataset:
    z: np.ndarray
    y: np.ndarray
    selected_signal: np.ndarray


def generate_dataset(
    episodes: int,
    seed: int,
    *,
    missing_strength: float,
    dependence_strength: float,
) -> Dataset:
    """Generate reports encoded as -1 (missing), 0, or 1."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0.0 <= missing_strength <= 1.0:
        raise ValueError("missing_strength must be in [0, 1]")
    if not 0.0 <= dependence_strength <= 1.0:
        raise ValueError("dependence_strength must be in [0, 1]")

    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=episodes, dtype=np.int8)
    signal_probability = np.where(z == 1, 0.82, 0.18)
    selected = np.empty((episodes, 6), dtype=np.int8)

    for left, right in GROUPS:
        shared = rng.binomial(1, signal_probability).astype(np.int8)
        for source in (left, right):
            independent = rng.binomial(1, signal_probability).astype(np.int8)
            use_shared = rng.random(episodes) < dependence_strength
            selected[:, source] = np.where(use_shared, shared, independent)

    reports = np.bitwise_xor(
        selected, rng.binomial(1, 0.10, size=selected.shape).astype(np.int8)
    )

    base_observation = 0.78
    strategic_by_state = np.where(z == 1, 0.68, 0.88)
    observation_probability = (
        (1.0 - missing_strength) * base_observation
        + missing_strength * strategic_by_state
    )
    observed = np.empty(selected.shape, dtype=bool)
    availability_coupling = missing_strength * dependence_strength * 0.85
    for left, right in GROUPS:
        independent = rng.random((episodes, 2)) < observation_probability[:, None]
        shared = rng.random(episodes) < observation_probability
        couple = rng.random(episodes) < availability_coupling
        observed[:, left] = np.where(couple, shared, independent[:, 0])
        observed[:, right] = np.where(couple, shared, independent[:, 1])
    y = np.where(observed, reports, -1).astype(np.int8)
    return Dataset(z=z, y=y, selected_signal=selected)


def _logsumexp_two(logits: np.ndarray) -> np.ndarray:
    maximum = logits.max(axis=1, keepdims=True)
    return maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1))


class _BaseModel:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    def _fit_prior(self, z: np.ndarray) -> None:
        counts = np.bincount(z, minlength=2).astype(float) + self.alpha
        self.log_prior = np.log(counts / counts.sum())

    def _probabilities(self, log_likelihood: np.ndarray) -> np.ndarray:
        logits = log_likelihood + self.log_prior[None, :]
        log_norm = _logsumexp_two(logits)
        return np.exp(logits[:, 1] - log_norm)


class NaiveModel(_BaseModel):
    def fit(self, data: Dataset) -> "NaiveModel":
        self._fit_prior(data.z)
        self.probability = np.empty((2, 6), dtype=float)
        for state in (0, 1):
            for source in range(6):
                values = data.y[(data.z == state) & (data.y[:, source] >= 0), source]
                self.probability[state, source] = (
                    values.sum() + self.alpha
                ) / (len(values) + 2.0 * self.alpha)
        return self

    def predict_proba(self, y: np.ndarray) -> np.ndarray:
        result = np.zeros((len(y), 2), dtype=float)
        for state in (0, 1):
            for source in range(6):
                observed = y[:, source] >= 0
                value = y[observed, source]
                p = self.probability[state, source]
                result[observed, state] += value * np.log(p) + (1 - value) * np.log(1 - p)
        return self._probabilities(result)


class MissingAwareModel(_BaseModel):
    def fit(self, data: Dataset) -> "MissingAwareModel":
        self._fit_prior(data.z)
        self.probability = np.empty((2, 6, 3), dtype=float)
        encoded = data.y + 1
        for state in (0, 1):
            for source in range(6):
                counts = np.bincount(
                    encoded[data.z == state, source], minlength=3
                ).astype(float) + self.alpha
                self.probability[state, source] = counts / counts.sum()
        return self

    def predict_proba(self, y: np.ndarray) -> np.ndarray:
        encoded = y + 1
        result = np.zeros((len(y), 2), dtype=float)
        rows = np.arange(len(y))
        for state in (0, 1):
            for source in range(6):
                result[:, state] += np.log(
                    self.probability[state, source, encoded[:, source]]
                )
        return self._probabilities(result)


class DependenceAwareModel(_BaseModel):
    def fit(self, data: Dataset) -> "DependenceAwareModel":
        self._fit_prior(data.z)
        self.tables = np.empty((2, len(GROUPS), 2, 2), dtype=float)
        for state in (0, 1):
            for group_index, (left, right) in enumerate(GROUPS):
                selected = data.y[data.z == state][:, (left, right)]
                complete = selected[(selected >= 0).all(axis=1)]
                counts = np.full((2, 2), self.alpha, dtype=float)
                for a, b in complete:
                    counts[a, b] += 1.0
                self.tables[state, group_index] = counts / counts.sum()
        return self

    def predict_proba(self, y: np.ndarray) -> np.ndarray:
        result = np.zeros((len(y), 2), dtype=float)
        for state in (0, 1):
            for group_index, (left, right) in enumerate(GROUPS):
                table = self.tables[state, group_index]
                for row_index, (a, b) in enumerate(y[:, (left, right)]):
                    if a >= 0 and b >= 0:
                        probability = table[a, b]
                    elif a >= 0:
                        probability = table[a, :].sum()
                    elif b >= 0:
                        probability = table[:, b].sum()
                    else:
                        probability = 1.0
                    result[row_index, state] += np.log(probability)
        return self._probabilities(result)


class JointModel(_BaseModel):
    def fit(self, data: Dataset) -> "JointModel":
        self._fit_prior(data.z)
        self.tables = np.empty((2, len(GROUPS), 3, 3), dtype=float)
        encoded = data.y + 1
        for state in (0, 1):
            for group_index, (left, right) in enumerate(GROUPS):
                counts = np.full((3, 3), self.alpha, dtype=float)
                for a, b in encoded[data.z == state][:, (left, right)]:
                    counts[a, b] += 1.0
                self.tables[state, group_index] = counts / counts.sum()
        return self

    def predict_proba(self, y: np.ndarray) -> np.ndarray:
        encoded = y + 1
        result = np.zeros((len(y), 2), dtype=float)
        rows = np.arange(len(y))
        for state in (0, 1):
            for group_index, (left, right) in enumerate(GROUPS):
                result[:, state] += np.log(
                    self.tables[state, group_index, encoded[:, left], encoded[:, right]]
                )
        return self._probabilities(result)


def fit_models(data: Dataset, alpha: float = 1.0) -> dict[str, _BaseModel]:
    return {
        "M1_naive": NaiveModel(alpha).fit(data),
        "M2_missing": MissingAwareModel(alpha).fit(data),
        "M3_dependence": DependenceAwareModel(alpha).fit(data),
        "M4_joint": JointModel(alpha).fit(data),
    }


def metrics(z: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    prediction = (clipped >= 0.5).astype(np.int8)
    confidence = np.maximum(clipped, 1.0 - clipped)
    wrong = prediction != z
    brier = np.mean((clipped - z) ** 2)
    nll = -np.mean(z * np.log(clipped) + (1 - z) * np.log(1 - clipped))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        member = (clipped >= edges[index]) & (
            clipped < edges[index + 1] if index < 9 else clipped <= edges[index + 1]
        )
        if member.any():
            ece += member.mean() * abs(clipped[member].mean() - z[member].mean())
    return {
        "accuracy": float(np.mean(prediction == z)),
        "brier": float(brier),
        "nll": float(nll),
        "ece": float(ece),
        "wrong_confidence": float(confidence[wrong].mean()) if wrong.any() else 0.0,
        "high_confidence_error_rate": float(np.mean(wrong & (confidence >= 0.9))),
    }


def diagnostics(data: Dataset) -> dict[str, float]:
    missing = data.y < 0
    result = {
        "missing_rate": float(missing.mean()),
        "missing_rate_z0": float(missing[data.z == 0].mean()),
        "missing_rate_z1": float(missing[data.z == 1].mean()),
    }
    agreements = []
    correlations = []
    missing_correlations = []
    for left, right in GROUPS:
        for state in (0, 1):
            state_rows = data.z == state
            complete = state_rows & (data.y[:, left] >= 0) & (data.y[:, right] >= 0)
            if complete.any():
                a = data.y[complete, left].astype(float)
                b = data.y[complete, right].astype(float)
                agreements.append(float(np.mean(a == b)))
                if a.std() > 0 and b.std() > 0:
                    correlations.append(float(np.corrcoef(a, b)[0, 1]))
            ma = (data.y[state_rows, left] < 0).astype(float)
            mb = (data.y[state_rows, right] < 0).astype(float)
            if ma.std() > 0 and mb.std() > 0:
                missing_correlations.append(float(np.corrcoef(ma, mb)[0, 1]))
    result["pair_agreement"] = float(np.mean(agreements))
    result["pair_correlation"] = float(np.mean(correlations))
    result["missing_pair_correlation"] = float(np.mean(missing_correlations))
    return result

