from copy import deepcopy

import numpy as np

from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES
from deception_rl.intelligence.ride_v2.latent_mechanism import (
    fit_latent_mechanism_rcl,
)


def _samples():
    rows = []
    rng = np.random.default_rng(17)
    for hypothesis in range(NUM_HYPOTHESES):
        for repeat in range(12):
            compromised = np.zeros((3, 3), dtype=int)
            compromised[repeat % 3, (repeat // 3) % 3] = int(repeat % 4 == 0)
            values = np.zeros((3, 3, 4), dtype=float)
            values[..., :2] = hypothesis / NUM_HYPOTHESES + rng.normal(0, 0.03, (3, 3, 2))
            values[compromised.astype(bool), :2] = 0.9 + rng.normal(
                0, 0.03, (int(compromised.sum()), 2)
            )
            rows.append(
                {
                    "hypothesis_label": hypothesis,
                    "reported_cells": np.ones((3, 3), dtype=bool),
                    "reported_values": values,
                    "compromised_cells": compromised,
                    "mechanism_label_mask": np.zeros((3, 3), dtype=bool),
                }
            )
    return rows


def test_zero_label_fit_is_invariant_to_hidden_mechanism_truth():
    original = _samples()
    changed = deepcopy(original)
    for row in changed:
        row["compromised_cells"] = 1 - row["compromised_cells"]
    first = fit_latent_mechanism_rcl(original, max_iterations=20)
    second = fit_latent_mechanism_rcl(changed, max_iterations=20)
    np.testing.assert_allclose(first.reliability.reliable, second.reliability.reliable)
    np.testing.assert_allclose(first.likelihood.reliable_mean, second.likelihood.reliable_mean)
    np.testing.assert_allclose(first.likelihood.compromised_mean, second.likelihood.compromised_mean)
    assert first.labeled_fraction == 0.0


def test_labeled_cells_are_clamped_and_parameters_are_finite():
    rows = _samples()
    for row in rows:
        row["mechanism_label_mask"] = np.asarray(row["reported_cells"]).copy()
    result = fit_latent_mechanism_rcl(rows, max_iterations=20)
    assert result.labeled_fraction == 1.0
    assert np.isfinite(result.observed_log_likelihood)
    assert np.all((result.reliability.reliable > 0) & (result.reliability.reliable < 1))
    assert np.all(result.likelihood.reliable_scale >= 0.08)
    assert np.all(result.likelihood.compromised_scale >= 0.08)


def test_latent_mechanism_runner_smoke(tmp_path):
    import json

    from experiments.run_intelligence_latent_mechanism import run

    destination = tmp_path / "latent_mechanism"
    result = run(
        destination,
        seeds=(7901,),
        episodes=1,
        train_episodes=1,
        neural_train_episodes=1,
        neural_validation_episodes=1,
        epochs=1,
        smoke=True,
    )
    audit = json.loads((destination / "audit.json").read_text())
    assert audit["test_episodes"] == 77
    assert not audit["runtime_mechanism_truth_used"]
    assert not result["third_innovation_established"]


def test_latent_mechanism_formal_runner_smoke(tmp_path):
    import json

    from experiments.run_intelligence_latent_mechanism_formal import run

    destination = tmp_path / "latent_mechanism_formal"
    result = run(destination, seeds=(8901,), episodes=1, smoke=True)
    audit = json.loads((destination / "information_boundary_audit.json").read_text())
    assert audit["test_episodes"] == 49
    assert audit["algorithm_frozen_after_development"]
    assert not audit["runtime_mechanism_truth_used"]
    assert not result["formal_primary_supported"]
