"""Interpretable progress-based belief baseline over observed trajectories."""

from __future__ import annotations

import numpy as np


class TrajectoryBeliefEstimator:
    """Convert observed movement progress into a normalized goal distribution.

    This is a lightweight baseline, not a calibrated probabilistic opponent
    model. A learned estimator can implement the same `estimate` interface.
    """

    def __init__(self, temperature: float = 2.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature

    def estimate(
        self,
        trajectory: np.ndarray,
        trajectory_mask: np.ndarray,
        target_positions: np.ndarray,
    ) -> dict[str, float]:
        positions = np.asarray(trajectory, dtype=np.int64)
        mask = np.asarray(trajectory_mask, dtype=bool)
        targets = np.asarray(target_positions, dtype=np.int64)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("trajectory must have shape (time, 2).")
        if mask.shape != (positions.shape[0],):
            raise ValueError("trajectory_mask must have one entry per timestep.")
        if targets.shape != (2, 2):
            raise ValueError("target_positions must have shape (2, 2).")

        visible_times = np.flatnonzero(mask)
        scores = np.zeros(2, dtype=np.float64)
        for previous_time, current_time in zip(visible_times, visible_times[1:]):
            elapsed = max(int(current_time - previous_time), 1)
            previous_position = positions[previous_time]
            current_position = positions[current_time]
            for goal in (0, 1):
                previous_distance = np.abs(targets[goal] - previous_position).sum()
                current_distance = np.abs(targets[goal] - current_position).sum()
                scores[goal] += (previous_distance - current_distance) / elapsed

        logits = scores / self.temperature
        weights = np.exp(logits - np.max(logits))
        probabilities = weights / weights.sum()
        return {"A": float(probabilities[0]), "B": float(probabilities[1])}
