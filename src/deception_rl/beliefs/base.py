"""Replaceable interface for estimating intent from Defender-visible data."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class BeliefEstimator(Protocol):
    def estimate(
        self,
        trajectory: np.ndarray,
        trajectory_mask: np.ndarray,
        target_positions: np.ndarray,
    ) -> dict[str, float]:
        """Return a normalized distribution over the public goals A and B."""
