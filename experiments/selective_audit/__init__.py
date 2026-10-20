"""Exact theory checks for the selective-inference direction audit."""

from .risk_bounds import (
    perfect_clean_classifier_risk_floor,
    worst_case_contaminated_selective_risk,
)

__all__ = [
    "perfect_clean_classifier_risk_floor",
    "worst_case_contaminated_selective_risk",
]
