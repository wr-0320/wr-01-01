import numpy as np
import pytest

from deception_rl.intelligence.ride_v2.granularity_baselines import (
    GroupedFixedReliability,
)


def samples():
    visible = np.ones((3, 3), dtype=bool)
    first = np.zeros((3, 3), dtype=int)
    second = np.zeros((3, 3), dtype=int)
    first[0, 0] = 1
    second[0, 1] = 1
    second[1, 2] = 1
    return [
        {"reported_cells": visible, "compromised_cells": first},
        {"reported_cells": visible, "compromised_cells": second},
    ]


def test_global_prior_is_shared_and_beta_smoothed():
    model = GroupedFixedReliability.fit(samples(), granularity="global")
    expected = 1.0 - (3.0 + 1.0) / (18.0 + 2.0)
    np.testing.assert_allclose(model.reliable, expected)


def test_source_prior_shares_regions_but_not_sources():
    model = GroupedFixedReliability.fit(samples(), granularity="source")
    np.testing.assert_allclose(model.reliable[:, 0], model.reliable[:, 1])
    np.testing.assert_allclose(model.reliable[:, 1], model.reliable[:, 2])
    assert model.reliable[0, 0] < model.reliable[2, 0]


def test_source_region_prior_matches_cellwise_beta_smoothing():
    model = GroupedFixedReliability.fit(samples(), granularity="source_region")
    assert model.reliable[0, 0] == pytest.approx(0.5)
    assert model.reliable[2, 2] == pytest.approx(0.75)


def test_invalid_granularity_is_rejected():
    with pytest.raises(ValueError, match="granularity"):
        GroupedFixedReliability.fit(samples(), granularity="region")

