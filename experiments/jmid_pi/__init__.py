"""Exact finite-alphabet checks for the JMID partial-identification study."""

from .verification import (
    MODEL_A,
    MODEL_B,
    sharp_likelihood_ratio_bounds,
)

__all__ = ["MODEL_A", "MODEL_B", "sharp_likelihood_ratio_bounds"]
