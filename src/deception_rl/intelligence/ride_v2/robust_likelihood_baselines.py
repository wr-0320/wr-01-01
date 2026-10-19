"""Robust reliable-emission baselines for the isolated RCL comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES
from deception_rl.intelligence.adapter import audit_blue_observation
from deception_rl.intelligence.ride_v2.reliability_conditioned import (
    ReliabilityConditionedLikelihood,
)


@dataclass(frozen=True)
class _ReliableEmissionLikelihood:
    reliable_mean: np.ndarray
    reliable_scale: np.ndarray
    temperature: float = 1.0

    def __post_init__(self) -> None:
        mean = np.asarray(self.reliable_mean, dtype=float)
        scale = np.asarray(self.reliable_scale, dtype=float)
        if mean.shape != scale.shape or mean.ndim != 4 or mean.shape[0] != NUM_HYPOTHESES:
            raise ValueError("reliable emission arrays have unexpected shapes")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("reliable emission arrays must be finite")
        if np.any(scale <= 0.0):
            raise ValueError("reliable scales must be positive")
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be positive and finite")
        object.__setattr__(self, "reliable_mean", mean.copy())
        object.__setattr__(self, "reliable_scale", scale.copy())

    @classmethod
    def from_emissions(cls, emissions: ReliabilityConditionedLikelihood, **kwargs):
        return cls(emissions.reliable_mean, emissions.reliable_scale, **kwargs)

    def _cell_score(self, value: np.ndarray, source: int, region: int) -> np.ndarray:
        raise NotImplementedError

    def _cell_weight(self, estimate, row_index: int, source: int, region: int) -> float:
        return 1.0

    def logits(self, observation: dict[str, Any], predictor, reasoner) -> np.ndarray:
        audit_blue_observation(observation)
        estimate = predictor.estimate(observation, reasoner)
        reports = np.asarray(observation["reports"], dtype=float)
        mask = np.asarray(observation["report_mask"], dtype=bool)
        logits = np.full(NUM_HYPOTHESES, -np.log(NUM_HYPOTHESES), dtype=float)
        for row_index, timestep in enumerate(estimate.time_indices):
            if timestep < 0:
                continue
            for source, region in np.argwhere(mask[int(timestep)]):
                source = int(source)
                region = int(region)
                value = reports[int(timestep), source, region, :2]
                weight = self._cell_weight(estimate, row_index, source, region)
                logits += weight * self._cell_score(value, source, region)
        return logits / self.temperature


@dataclass(frozen=True)
class StudentTReliableLikelihood(_ReliableEmissionLikelihood):
    """Independent Student-t reliable emissions with a shared degrees of freedom."""

    degrees_of_freedom: float = 4.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.isfinite(self.degrees_of_freedom) or self.degrees_of_freedom <= 2.0:
            raise ValueError("degrees_of_freedom must be finite and greater than two")

    def _cell_score(self, value, source, region):
        scale = self.reliable_scale[:, source, region]
        residual = (value - self.reliable_mean[:, source, region]) / scale
        nu = float(self.degrees_of_freedom)
        return -np.sum(np.log(scale), axis=-1) - 0.5 * (nu + 1.0) * np.sum(
            np.log1p((residual**2) / nu), axis=-1
        )

    def with_temperature(self, temperature: float):
        return StudentTReliableLikelihood(
            self.reliable_mean,
            self.reliable_scale,
            float(temperature),
            self.degrees_of_freedom,
        )


@dataclass(frozen=True)
class HuberReliableLikelihood(_ReliableEmissionLikelihood):
    """Huber pseudo-likelihood applied to standardized reliable residuals."""

    delta: float = 1.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.isfinite(self.delta) or self.delta <= 0.0:
            raise ValueError("delta must be positive and finite")

    def _cell_score(self, value, source, region):
        scale = self.reliable_scale[:, source, region]
        residual = np.abs((value - self.reliable_mean[:, source, region]) / scale)
        loss = np.where(
            residual <= self.delta,
            0.5 * residual**2,
            self.delta * (residual - 0.5 * self.delta),
        )
        return -np.sum(np.log(scale), axis=-1) - np.sum(loss, axis=-1)

    def with_temperature(self, temperature: float):
        return HuberReliableLikelihood(
            self.reliable_mean,
            self.reliable_scale,
            float(temperature),
            self.delta,
        )


@dataclass(frozen=True)
class ReliabilityWeightedGaussianLikelihood(_ReliableEmissionLikelihood):
    """Gaussian reliable evidence attenuated by a training-only reliability prior."""

    weight_power: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.isfinite(self.weight_power) or self.weight_power <= 0.0:
            raise ValueError("weight_power must be positive and finite")

    def _cell_score(self, value, source, region):
        return ReliabilityConditionedLikelihood._log_normal(
            value,
            self.reliable_mean[:, source, region],
            self.reliable_scale[:, source, region],
        )

    def _cell_weight(self, estimate, row_index, source, region):
        reliability = float(estimate.history[row_index, source, region])
        return float(np.clip(reliability, 0.0, 1.0) ** self.weight_power)

    def with_temperature(self, temperature: float):
        return ReliabilityWeightedGaussianLikelihood(
            self.reliable_mean,
            self.reliable_scale,
            float(temperature),
            self.weight_power,
        )
