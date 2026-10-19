"""Semi-supervised latent reliable/compromised mechanism estimation for RCL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from deception_rl.envs.deceptive_intelligence import (
    NUM_HYPOTHESES,
    NUM_REGIONS,
    NUM_SOURCES,
)
from deception_rl.intelligence.ride_v2.fixed_reliability import FixedCellReliability
from deception_rl.intelligence.ride_v2.reliability_conditioned import (
    ReliabilityConditionedLikelihood,
)


@dataclass(frozen=True)
class LatentMechanismResult:
    """Fitted RCL parameters and optimization diagnostics."""

    likelihood: ReliabilityConditionedLikelihood
    reliability: FixedCellReliability
    iterations: int
    converged: bool
    observed_log_likelihood: float
    labeled_fraction: float


def _log_normal(value: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return -0.5 * np.sum(((value - mean) / scale) ** 2, axis=-1) - np.sum(
        np.log(scale), axis=-1
    )


def _flatten(samples: Iterable[dict[str, Any]]) -> dict[str, np.ndarray]:
    values: list[np.ndarray] = []
    hypotheses: list[int] = []
    sources: list[int] = []
    regions: list[int] = []
    labels: list[int] = []
    for row in samples:
        visible = np.asarray(row["reported_cells"], dtype=bool)
        label_mask = np.asarray(
            row.get("mechanism_label_mask", np.zeros_like(visible)), dtype=bool
        )
        if label_mask.shape != visible.shape or np.any(label_mask & ~visible):
            raise ValueError("mechanism label mask must select only visible cells")
        compromised = np.asarray(
            row.get("compromised_cells", np.zeros_like(visible)), dtype=int
        )
        reports = np.asarray(row["reported_values"], dtype=float)
        for source, region in np.argwhere(visible):
            source = int(source)
            region = int(region)
            values.append(reports[source, region, :2].copy())
            hypotheses.append(int(row["hypothesis_label"]))
            sources.append(source)
            regions.append(region)
            labels.append(int(compromised[source, region]) if label_mask[source, region] else -1)
    if not values:
        raise ValueError("latent-mechanism samples contain no visible reports")
    return {
        "x": np.asarray(values, dtype=float),
        "h": np.asarray(hypotheses, dtype=int),
        "s": np.asarray(sources, dtype=int),
        "r": np.asarray(regions, dtype=int),
        "label": np.asarray(labels, dtype=int),
    }


def _weighted_parameters(
    x: np.ndarray,
    weights: np.ndarray,
    fallback_mean: np.ndarray,
    fallback_scale: np.ndarray,
    *,
    scale_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    total = float(weights.sum())
    if total < 2.0:
        return fallback_mean.copy(), fallback_scale.copy()
    mean = np.sum(weights[:, None] * x, axis=0) / total
    variance = np.sum(weights[:, None] * (x - mean) ** 2, axis=0) / total
    return mean, np.maximum(np.sqrt(variance), scale_floor)


def fit_latent_mechanism_rcl(
    samples: Iterable[dict[str, Any]],
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    scale_floor: float = 0.08,
    initial_compromise_rate: float = 0.15,
) -> LatentMechanismResult:
    """Fit RCL with cell mechanism labels clamped where available.

    ``mechanism_label_mask`` is the only switch that permits reading a cell's
    ``compromised_cells`` value. Consequently, changing hidden labels cannot
    change a 0%-label fit. The operational-hypothesis label remains supervised.
    """

    data = _flatten(samples)
    x, h, source, region, labels = (
        data["x"],
        data["h"],
        data["s"],
        data["r"],
        data["label"],
    )
    global_mean = x.mean(axis=0)
    global_scale = np.maximum(x.std(axis=0), scale_floor)

    reliable_mean = np.empty((NUM_HYPOTHESES, NUM_SOURCES, NUM_REGIONS, 2))
    reliable_scale = np.empty_like(reliable_mean)
    for hypothesis in range(NUM_HYPOTHESES):
        for s in range(NUM_SOURCES):
            for r in range(NUM_REGIONS):
                selected = (h == hypothesis) & (source == s) & (region == r)
                reliable_mean[hypothesis, s, r] = (
                    x[selected].mean(axis=0) if selected.any() else global_mean
                )
                reliable_scale[hypothesis, s, r] = (
                    np.maximum(x[selected].std(axis=0), scale_floor)
                    if selected.sum() >= 2
                    else global_scale
                )

    compromised_mean = np.empty((NUM_SOURCES, NUM_REGIONS, 2))
    compromised_scale = np.empty_like(compromised_mean)
    reliability = np.full((NUM_SOURCES, NUM_REGIONS), 1.0 - initial_compromise_rate)
    for s in range(NUM_SOURCES):
        for r in range(NUM_REGIONS):
            selected = (source == s) & (region == r)
            indices = np.flatnonzero(selected)
            labeled = indices[labels[indices] >= 0]
            if len(labeled):
                compromised_count = float((labels[labeled] == 1).sum())
                reliability[s, r] = 1.0 - (compromised_count + 1.0) / (len(labeled) + 2.0)
            labeled_compromised = indices[labels[indices] == 1]
            if len(labeled_compromised) >= 2:
                seed_values = x[labeled_compromised]
            elif len(indices):
                residual = np.sum(
                    ((x[indices] - reliable_mean[h[indices], s, r]) / global_scale) ** 2,
                    axis=1,
                )
                take = max(2, int(np.ceil(initial_compromise_rate * len(indices))))
                seed_values = x[indices[np.argsort(residual)[-take:]]]
            else:
                seed_values = x
            compromised_mean[s, r] = seed_values.mean(axis=0)
            compromised_scale[s, r] = np.maximum(seed_values.std(axis=0), scale_floor)

    previous = -np.inf
    converged = False
    observed_log_likelihood = -np.inf
    for iteration in range(1, max_iterations + 1):
        reliable_log = _log_normal(
            x,
            reliable_mean[h, source, region],
            reliable_scale[h, source, region],
        ) + np.log(np.clip(reliability[source, region], 1e-6, 1 - 1e-6))
        compromised_log = _log_normal(
            x,
            compromised_mean[source, region],
            compromised_scale[source, region],
        ) + np.log(np.clip(1.0 - reliability[source, region], 1e-6, 1 - 1e-6))
        denominator = np.logaddexp(reliable_log, compromised_log)
        gamma = np.exp(reliable_log - denominator)
        gamma[labels == 0] = 1.0
        gamma[labels == 1] = 0.0
        observed_log_likelihood = float(denominator[labels < 0].sum())
        observed_log_likelihood += float(reliable_log[labels == 0].sum())
        observed_log_likelihood += float(compromised_log[labels == 1].sum())

        for s in range(NUM_SOURCES):
            for r in range(NUM_REGIONS):
                cell = (source == s) & (region == r)
                cell_count = int(cell.sum())
                reliability[s, r] = (float(gamma[cell].sum()) + 1.0) / (cell_count + 2.0)
                c_mean, c_scale = _weighted_parameters(
                    x[cell],
                    1.0 - gamma[cell],
                    global_mean,
                    global_scale,
                    scale_floor=scale_floor,
                )
                compromised_mean[s, r] = c_mean
                compromised_scale[s, r] = c_scale
                for hypothesis in range(NUM_HYPOTHESES):
                    group = cell & (h == hypothesis)
                    r_mean, r_scale = _weighted_parameters(
                        x[group],
                        gamma[group],
                        global_mean,
                        global_scale,
                        scale_floor=scale_floor,
                    )
                    reliable_mean[hypothesis, s, r] = r_mean
                    reliable_scale[hypothesis, s, r] = r_scale

        if np.isfinite(previous) and abs(observed_log_likelihood - previous) <= tolerance * (
            1.0 + abs(previous)
        ):
            converged = True
            break
        previous = observed_log_likelihood

    likelihood = ReliabilityConditionedLikelihood(
        reliable_mean,
        reliable_scale,
        compromised_mean,
        compromised_scale,
    )
    return LatentMechanismResult(
        likelihood=likelihood,
        reliability=FixedCellReliability(reliability),
        iterations=iteration,
        converged=converged,
        observed_log_likelihood=observed_log_likelihood,
        labeled_fraction=float(np.mean(labels >= 0)),
    )


def fit_unlabeled_single_mechanism(
    samples: Iterable[dict[str, Any]], *, scale_floor: float = 0.08
) -> ReliabilityConditionedLikelihood:
    """Fit one H-conditioned Gaussian to every report without mechanism labels."""

    rows = list(samples)
    data = _flatten(
        [
            dict(
                row,
                mechanism_label_mask=np.zeros_like(row["reported_cells"], dtype=bool),
            )
            for row in rows
        ]
    )
    x, h, source, region = data["x"], data["h"], data["s"], data["r"]
    global_mean = x.mean(axis=0)
    global_scale = np.maximum(x.std(axis=0), scale_floor)
    mean = np.empty((NUM_HYPOTHESES, NUM_SOURCES, NUM_REGIONS, 2))
    scale = np.empty_like(mean)
    for hypothesis in range(NUM_HYPOTHESES):
        for s in range(NUM_SOURCES):
            for r in range(NUM_REGIONS):
                selected = (h == hypothesis) & (source == s) & (region == r)
                values = x[selected]
                mean[hypothesis, s, r] = values.mean(axis=0) if len(values) else global_mean
                scale[hypothesis, s, r] = (
                    np.maximum(values.std(axis=0), scale_floor)
                    if len(values) >= 2
                    else global_scale
                )
    return ReliabilityConditionedLikelihood(
        mean,
        scale,
        np.broadcast_to(global_mean, (NUM_SOURCES, NUM_REGIONS, 2)).copy(),
        np.broadcast_to(global_scale, (NUM_SOURCES, NUM_REGIONS, 2)).copy(),
    )


def mechanism_responsibilities(
    result: LatentMechanismResult, samples: Iterable[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    """Return offline compromised probabilities and labels for diagnostics only."""

    rows = list(samples)
    data = _flatten(
        [
            dict(
                row,
                mechanism_label_mask=np.zeros_like(row["reported_cells"], dtype=bool),
            )
            for row in rows
        ]
    )
    model = result.likelihood
    x, h, source, region = data["x"], data["h"], data["s"], data["r"]
    rho = np.clip(result.reliability.reliable[source, region], 1e-6, 1 - 1e-6)
    reliable_log = _log_normal(
        x, model.reliable_mean[h, source, region], model.reliable_scale[h, source, region]
    ) + np.log(rho)
    compromised_log = _log_normal(
        x, model.compromised_mean[source, region], model.compromised_scale[source, region]
    ) + np.log(1.0 - rho)
    probability = np.exp(compromised_log - np.logaddexp(reliable_log, compromised_log))
    truth = []
    for row in rows:
        visible = np.asarray(row["reported_cells"], dtype=bool)
        compromised = np.asarray(row["compromised_cells"], dtype=int)
        truth.extend(compromised[visible].tolist())
    return probability, np.asarray(truth, dtype=int)
