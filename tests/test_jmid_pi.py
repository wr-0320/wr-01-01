from fractions import Fraction as F

import pytest

from experiments.jmid_pi.verification import (
    MODEL_A,
    MODEL_B,
    exact_witness,
    sharp_likelihood_ratio_bounds,
)


def test_exact_nonidentifiability_witness():
    witness = exact_witness()
    expected_observed = {"***": F(3, 10), "000": F(7, 20), "111": F(7, 20)}
    assert witness["model_a_observed"] == expected_observed
    assert witness["model_b_observed"] == expected_observed
    assert witness["model_a_posterior_111"] == F(6, 7)
    assert witness["model_b_posterior_111"] == F(5, 7)
    assert witness["model_a_posterior_000"] == F(1, 7)
    assert witness["model_b_posterior_000"] == F(2, 7)
    assert witness["different_graphs"] is True
    assert witness["both_mnar"] is True


def test_each_latent_construction_is_a_valid_probability_model():
    for model in (MODEL_A, MODEL_B):
        assert sum(model.observed_marginal().values()) == 1
        for state in (0, 1):
            conditional = model.conditional_observed(state)
            assert sum(conditional.values()) == 1
            assert all(F(0) <= value <= F(1) for value in conditional.values())


def test_sharp_bounds_include_global_prior_constraint():
    observed = {"a": F(7, 20), "b": F(7, 20), "c": F(3, 10)}
    envelopes = {
        "a": (F(1), F(9)),
        "b": (F(1, 9), F(1, 4)),
        "c": (F(2, 3), F(1)),
    }
    bounds = sharp_likelihood_ratio_bounds(observed, F(1, 2), envelopes)
    # Local bounds for a would be [1/2, 9/10]. The known prior and the upper
    # masses allowed for b,c sharpen its lower endpoint to 4/5.
    assert bounds["a"] == (F(4, 5), F(9, 10))


def test_sharp_bounds_reject_infeasible_envelopes():
    observed = {"a": F(1, 2), "b": F(1, 2)}
    with pytest.raises(ValueError, match="upper likelihood-ratio bounds"):
        sharp_likelihood_ratio_bounds(
            observed,
            F(3, 4),
            {"a": (F(0), F(1, 3)), "b": (F(0), F(1, 3))},
        )
