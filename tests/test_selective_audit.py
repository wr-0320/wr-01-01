import csv
from fractions import Fraction as F
from pathlib import Path

import pytest

from experiments.selective_audit.risk_bounds import (
    perfect_clean_classifier_risk_floor,
    worst_case_contaminated_selective_risk,
)


def test_unrestricted_contamination_has_exact_selective_risk_bound():
    assert worst_case_contaminated_selective_risk(F(1, 2), F(1, 20), F(1, 10)) == F(29, 110)


def test_rejecting_more_can_raise_the_adversarial_risk_floor():
    epsilon = F(1, 10)
    assert perfect_clean_classifier_risk_floor(F(1), epsilon) == epsilon
    assert perfect_clean_classifier_risk_floor(F(1, 2), epsilon) == F(2, 11)
    assert perfect_clean_classifier_risk_floor(F(1, 4), epsilon) == F(4, 13)


def test_unknown_unbounded_contamination_precludes_nontrivial_control():
    coverage = F(3, 5)
    risks = [perfect_clean_classifier_risk_floor(coverage, F(n, 100)) for n in (10, 50, 90, 99)]
    assert risks == sorted(risks)
    assert risks[-1] > F(49, 50)


def test_invalid_risk_inputs_are_rejected():
    with pytest.raises(ValueError):
        worst_case_contaminated_selective_risk(F(1, 2), F(3, 4), F(1, 10))


def test_frozen_competitor_matrix_has_no_unverified_fulltext_claims():
    matrix = Path(__file__).parents[1] / "docs" / "selective_inference_competitor_matrix.csv"
    with matrix.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 15
    assert len({row["id"] for row in rows}) == 15
    assert {row["status"] for row in rows} == {"metadata_only"}
    assert all(not row["assumption"] and not row["guarantee"] for row in rows)
