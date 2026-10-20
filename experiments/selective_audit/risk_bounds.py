"""Elementary risk bounds used to screen the candidate direction.

This module contains no learned selector, classifier, or verification policy.
"""

from __future__ import annotations

from fractions import Fraction


def worst_case_contaminated_selective_risk(
    clean_coverage: Fraction,
    clean_error_mass: Fraction,
    contamination: Fraction,
) -> Fraction:
    """Exact worst-case selective risk under unrestricted Huber contamination."""
    zero, one = Fraction(0), Fraction(1)
    if not zero < clean_coverage <= one:
        raise ValueError("clean coverage must lie in (0, 1]")
    if not zero <= clean_error_mass <= clean_coverage:
        raise ValueError("clean error mass must lie in [0, clean coverage]")
    if not zero <= contamination < one:
        raise ValueError("contamination must lie in [0, 1)")
    numerator = (one - contamination) * clean_error_mass + contamination
    denominator = (one - contamination) * clean_coverage + contamination
    return numerator / denominator


def perfect_clean_classifier_risk_floor(
    clean_coverage: Fraction, contamination: Fraction
) -> Fraction:
    return worst_case_contaminated_selective_risk(
        clean_coverage, Fraction(0), contamination
    )
