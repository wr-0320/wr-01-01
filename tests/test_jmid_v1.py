import csv

import numpy as np

from experiments.jmid_v1.core import diagnostics, fit_models, generate_dataset
from experiments.jmid_v1.run import run


def test_controls_change_missingness_and_dependence():
    random_independent = generate_dataset(
        20_000, 11, missing_strength=0.0, dependence_strength=0.0
    )
    strategic_dependent = generate_dataset(
        20_000, 11, missing_strength=1.0, dependence_strength=0.9
    )
    base = diagnostics(random_independent)
    hard = diagnostics(strategic_dependent)
    assert hard["missing_rate_z1"] > hard["missing_rate_z0"] + 0.15
    assert hard["pair_correlation"] > base["pair_correlation"] + 0.20
    assert hard["missing_pair_correlation"] > base["missing_pair_correlation"] + 0.50


def test_dependence_control_preserves_state_conditional_missing_margins():
    strategic_independent = diagnostics(generate_dataset(
        30_000, 18, missing_strength=1.0, dependence_strength=0.0
    ))
    strategic_dependent = diagnostics(generate_dataset(
        30_000, 18, missing_strength=1.0, dependence_strength=0.9
    ))
    for key in ("missing_rate_z0", "missing_rate_z1"):
        assert abs(strategic_independent[key] - strategic_dependent[key]) < 0.015


def test_all_models_return_finite_probabilities():
    train = generate_dataset(800, 22, missing_strength=1.0, dependence_strength=0.9)
    test = generate_dataset(120, 23, missing_strength=1.0, dependence_strength=0.9)
    models = fit_models(train)
    assert set(models) == {"M1_naive", "M2_missing", "M3_dependence", "M4_joint"}
    for model in models.values():
        probability = model.predict_proba(test.y)
        assert probability.shape == (120,)
        assert np.isfinite(probability).all()
        assert ((probability > 0) & (probability < 1)).all()


def test_smoke_runner_pairs_methods_and_refuses_overwrite(tmp_path):
    output = tmp_path / "jmid"
    result = run(output, seeds=(77,), train_episodes=200, test_episodes=250, smoke=True)
    assert result == {"condition_rows": 16, "shift_rows": 20}
    with (output / "condition_per_seed.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 16
    assert len({(r["seed"], r["condition"]) for r in rows}) == 4
    try:
        run(output, seeds=(77,), smoke=True)
    except FileExistsError:
        pass
    else:
        raise AssertionError("runner overwrote an existing result directory")

