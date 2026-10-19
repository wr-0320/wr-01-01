"""Calibrated cell-compromise probability models with no runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from deception_rl.envs.deceptive_intelligence import NUM_REGIONS, NUM_SOURCES
from deception_rl.intelligence.ride import HonestIntelligenceReasoner
from deception_rl.intelligence.ride_v2.reliability import RELIABILITY_FEATURES
from deception_rl.intelligence.ride_v2.source_region_reliability import (
    SourceRegionReliabilityEstimate,
    extract_source_region_evidence,
)


EPSILON = 1e-7
CONTINUOUS_FEATURES = tuple(RELIABILITY_FEATURES) + tuple(
    f"recent_change_{name}" for name in RELIABILITY_FEATURES
)
DESIGN_FEATURES = CONTINUOUS_FEATURES + tuple(
    f"source_{index}" for index in range(NUM_SOURCES)
) + tuple(f"region_{index}" for index in range(NUM_REGIONS))


def _design_row(
    current: np.ndarray,
    previous: np.ndarray | None,
    source: int,
    region: int,
) -> np.ndarray:
    change = np.zeros_like(current) if previous is None else np.abs(current - previous)
    identity = np.zeros(NUM_SOURCES + NUM_REGIONS, dtype=float)
    identity[source] = 1.0
    identity[NUM_SOURCES + region] = 1.0
    return np.concatenate((current[source, region], change[source, region], identity))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-value))


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(probability / (1.0 - probability))


def build_cell_dataset(samples: Iterable[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten visible cells and add per-cell recent changes and identity features."""

    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in samples:
        by_episode.setdefault(str(row["episode_id"]), []).append(row)
    features: list[np.ndarray] = []
    labels: list[int] = []
    for episode_rows in by_episode.values():
        previous: np.ndarray | None = None
        for row in sorted(episode_rows, key=lambda item: int(item["step"])):
            current = np.asarray(row["cell_features"], dtype=float)
            compromised = np.asarray(row["compromised_cells"], dtype=int)
            reported = np.asarray(row["reported_cells"], dtype=bool)
            if current.shape != (NUM_SOURCES, NUM_REGIONS, len(RELIABILITY_FEATURES)):
                raise ValueError("cell features have an unexpected shape")
            if compromised.shape != (NUM_SOURCES, NUM_REGIONS):
                raise ValueError("cell labels have an unexpected shape")
            if reported.shape != (NUM_SOURCES, NUM_REGIONS):
                raise ValueError("reported-cell mask has an unexpected shape")
            for source, region in np.argwhere(reported):
                features.append(_design_row(current, previous, int(source), int(region)))
                labels.append(int(compromised[source, region]))
            previous = current
    if not features:
        raise ValueError("cell dataset contains no visible reports")
    return np.stack(features), np.asarray(labels, dtype=int)


def _fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    l2: float,
    max_iterations: int = 100,
) -> np.ndarray:
    design = np.column_stack((np.ones(len(matrix)), np.asarray(matrix, dtype=float)))
    labels = np.asarray(labels, dtype=float)
    weights = np.asarray(weights, dtype=float)
    coefficient = np.zeros(design.shape[1], dtype=float)
    prevalence = float(np.average(labels, weights=weights))
    coefficient[0] = float(_logit(np.array([prevalence]))[0])
    penalty = np.eye(design.shape[1], dtype=float) * float(l2)
    penalty[0, 0] = 0.0
    for _ in range(max_iterations):
        probability = _sigmoid(design @ coefficient)
        curvature = np.maximum(probability * (1.0 - probability), 1e-5) * weights
        gradient = design.T @ (weights * (probability - labels)) + penalty @ coefficient
        hessian = design.T @ (curvature[:, None] * design) + penalty
        try:
            update = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            update = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        coefficient -= update
        if float(np.max(np.abs(update))) < 1e-8:
            break
    return coefficient


@dataclass(frozen=True)
class LogisticCellModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    balanced: bool

    @classmethod
    def fit(
        cls,
        matrix: np.ndarray,
        labels: np.ndarray,
        *,
        balanced: bool,
        l2: float = 1.0,
    ) -> "LogisticCellModel":
        matrix = np.asarray(matrix, dtype=float)
        labels = np.asarray(labels, dtype=int)
        mean = matrix[:, : len(CONTINUOUS_FEATURES)].mean(axis=0)
        scale = matrix[:, : len(CONTINUOUS_FEATURES)].std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        standardized = matrix.copy()
        standardized[:, : len(CONTINUOUS_FEATURES)] = (
            standardized[:, : len(CONTINUOUS_FEATURES)] - mean
        ) / scale
        weights = np.ones(len(labels), dtype=float)
        if balanced:
            prevalence = float(labels.mean())
            if not 0.0 < prevalence < 1.0:
                raise ValueError("balanced fitting requires both classes")
            weights[labels == 1] = 0.5 / prevalence
            weights[labels == 0] = 0.5 / (1.0 - prevalence)
        coefficient = _fit_logistic(standardized, labels, weights, l2=l2)
        return cls(mean, scale, coefficient, balanced)

    def decision_function(self, matrix: np.ndarray) -> np.ndarray:
        standardized = np.asarray(matrix, dtype=float).copy()
        standardized[:, : len(CONTINUOUS_FEATURES)] = (
            standardized[:, : len(CONTINUOUS_FEATURES)] - self.mean
        ) / self.scale
        return self.coefficient[0] + standardized @ self.coefficient[1:]

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        return _sigmoid(self.decision_function(matrix))

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(DESIGN_FEATURES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficient": self.coefficient.tolist(),
            "balanced": self.balanced,
        }


@dataclass(frozen=True)
class PlattCalibrator:
    intercept: float
    slope: float

    @classmethod
    def fit(cls, score: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        coefficient = _fit_logistic(
            np.asarray(score, dtype=float).reshape(-1, 1),
            np.asarray(labels, dtype=int),
            np.ones(len(labels), dtype=float),
            l2=1e-4,
        )
        return cls(float(coefficient[0]), float(coefficient[1]))

    def predict(self, score: np.ndarray) -> np.ndarray:
        return _sigmoid(self.intercept + self.slope * np.asarray(score, dtype=float))


@dataclass(frozen=True)
class IsotonicCalibrator:
    upper_scores: np.ndarray
    probabilities: np.ndarray

    @classmethod
    def fit(cls, score: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        score = np.asarray(score, dtype=float)
        labels = np.asarray(labels, dtype=float)
        order = np.argsort(score, kind="stable")
        unique, first, counts = np.unique(score[order], return_index=True, return_counts=True)
        sums = np.add.reduceat(labels[order], first)
        blocks = [
            [float(unique[index]), float(counts[index]), float(sums[index])]
            for index in range(len(unique))
        ]
        index = 0
        while index < len(blocks) - 1:
            if blocks[index][2] / blocks[index][1] <= blocks[index + 1][2] / blocks[index + 1][1]:
                index += 1
                continue
            blocks[index][0] = blocks[index + 1][0]
            blocks[index][1] += blocks[index + 1][1]
            blocks[index][2] += blocks[index + 1][2]
            del blocks[index + 1]
            index = max(index - 1, 0)
        return cls(
            np.asarray([block[0] for block in blocks]),
            np.asarray([block[2] / block[1] for block in blocks]),
        )

    def predict(self, score: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(self.upper_scores, np.asarray(score), side="left")
        return self.probabilities[np.clip(indices, 0, len(self.probabilities) - 1)]


class CalibratedCellReliabilityFilter:
    """Apply a frozen cell classifier to blue-visible report histories."""

    def __init__(
        self,
        model: LogisticCellModel,
        calibrator: PlattCalibrator | IsotonicCalibrator | None = None,
    ) -> None:
        self.model = model
        self.calibrator = calibrator

    def estimate(
        self,
        observation: dict[str, Any],
        reasoner: HonestIntelligenceReasoner,
    ) -> SourceRegionReliabilityEstimate:
        evidence, time_indices = extract_source_region_evidence(observation, reasoner)
        report_mask = np.asarray(observation["report_mask"], dtype=bool)
        history = np.ones((len(evidence), NUM_SOURCES, NUM_REGIONS), dtype=float)
        previous: np.ndarray | None = None
        for row_index, (current, timestep) in enumerate(zip(evidence, time_indices)):
            if timestep < 0:
                previous = current
                continue
            visible = np.argwhere(report_mask[int(timestep)])
            if len(visible):
                matrix = np.stack(
                    [
                        _design_row(current, previous, int(source), int(region))
                        for source, region in visible
                    ]
                )
                score = self.model.decision_function(matrix)
                compromised = (
                    self.model.predict_proba(matrix)
                    if self.calibrator is None
                    else self.calibrator.predict(score)
                )
                for (source, region), probability in zip(visible, compromised):
                    history[row_index, int(source), int(region)] = 1.0 - float(probability)
            previous = current
        global_history = history.mean(axis=2)
        return SourceRegionReliabilityEstimate(
            current=history[-1],
            history=history,
            global_history=global_history,
            time_indices=time_indices,
            evidence=evidence,
        )


def _rank_auc(probability: np.ndarray, labels: np.ndarray) -> float:
    positive = probability[labels == 1]
    negative = np.sort(probability[labels == 0])
    if not len(positive) or not len(negative):
        return float("nan")
    lower = np.searchsorted(negative, positive, side="left")
    upper = np.searchsorted(negative, positive, side="right")
    return float(np.mean((lower + upper) / 2.0) / len(negative))


def _average_precision(probability: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-probability, kind="stable")
    ordered = labels[order]
    positives = int(ordered.sum())
    if not positives:
        return float("nan")
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / positives)


def calibration_bins(
    probability: np.ndarray, labels: np.ndarray, *, bins: int = 10
) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    output = []
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        if selected.any():
            output.append(
                {
                    "bin": index,
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "count": int(selected.sum()),
                    "mean_probability": float(probability[selected].mean()),
                    "observed_frequency": float(labels[selected].mean()),
                }
            )
    return output


def probability_metrics(probability: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    labels = np.asarray(labels, dtype=int)
    prevalence = float(labels.mean())
    brier = float(np.mean((probability - labels) ** 2))
    no_skill_brier = prevalence * (1.0 - prevalence)
    bins = calibration_bins(probability, labels)
    ece = float(
        sum(row["count"] * abs(row["mean_probability"] - row["observed_frequency"]) for row in bins)
        / len(labels)
    )
    prediction = probability >= 0.5
    true_positive = int(((labels == 1) & prediction).sum())
    false_positive = int(((labels == 0) & prediction).sum())
    false_negative = int(((labels == 1) & ~prediction).sum())
    f1_denominator = 2 * true_positive + false_positive + false_negative
    calibration = PlattCalibrator.fit(_logit(probability), labels)
    return {
        "samples": len(labels),
        "compromised_prevalence": prevalence,
        "brier": brier,
        "no_skill_brier": no_skill_brier,
        "brier_skill": 1.0 - brier / no_skill_brier if no_skill_brier else float("nan"),
        "ece": ece,
        "auroc": _rank_auc(probability, labels),
        "average_precision": _average_precision(probability, labels),
        "minority_recall_at_0_5": true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0,
        "f1_at_0_5": 2 * true_positive / f1_denominator if f1_denominator else 0.0,
        "calibration_intercept": calibration.intercept,
        "calibration_slope": calibration.slope,
    }
