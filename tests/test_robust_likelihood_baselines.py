import numpy as np
import pytest

from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES
from deception_rl.intelligence.ride_v2.robust_likelihood_baselines import (
    HuberReliableLikelihood,
    ReliabilityWeightedGaussianLikelihood,
    StudentTReliableLikelihood,
)


def parameters():
    mean = np.zeros((NUM_HYPOTHESES, 3, 3, 2), dtype=float)
    mean[1] = 0.5
    mean[2] = 1.0
    scale = np.full_like(mean, 0.2)
    return mean, scale


def test_robust_scores_are_finite_and_keep_hypothesis_axis():
    mean, scale = parameters()
    models = (
        StudentTReliableLikelihood(mean, scale, degrees_of_freedom=4.0),
        HuberReliableLikelihood(mean, scale, delta=1.5),
        ReliabilityWeightedGaussianLikelihood(mean, scale),
    )
    for model in models:
        score = model._cell_score(np.array([0.2, 0.3]), 0, 0)
        assert score.shape == (NUM_HYPOTHESES,)
        assert np.isfinite(score).all()
        assert model.with_temperature(2.0).temperature == 2.0


def test_robust_hyperparameters_are_validated():
    mean, scale = parameters()
    with pytest.raises(ValueError):
        StudentTReliableLikelihood(mean, scale, degrees_of_freedom=2.0)
    with pytest.raises(ValueError):
        HuberReliableLikelihood(mean, scale, delta=0.0)
    with pytest.raises(ValueError):
        ReliabilityWeightedGaussianLikelihood(mean, scale, weight_power=0.0)


def test_student_and_huber_limit_extreme_residual_growth():
    mean, scale = parameters()
    student = StudentTReliableLikelihood(mean, scale, degrees_of_freedom=4.0)
    huber = HuberReliableLikelihood(mean, scale, delta=1.5)
    near_student = student._cell_score(np.array([1.0, 1.0]), 0, 0)[0]
    far_student = student._cell_score(np.array([10.0, 10.0]), 0, 0)[0]
    near_huber = huber._cell_score(np.array([1.0, 1.0]), 0, 0)[0]
    far_huber = huber._cell_score(np.array([10.0, 10.0]), 0, 0)[0]
    assert far_student < near_student
    assert far_huber < near_huber
    assert far_student > -100.0
    assert far_huber > -200.0
