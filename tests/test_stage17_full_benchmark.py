import hashlib

import numpy as np

from deception_rl.external_baselines import RIDeDefenderAdapter, RuleActiveDefender
from deception_rl.external_envs import DefenderAction
from experiments.run_stage17_full_benchmark import (
    ATTACK_SETTINGS,
    ENVIRONMENTS,
    METHODS,
    RIDE_CHECKPOINT,
    benchmark_specs,
    build_results,
    convergence_rows,
    evaluate_episode,
)


def test_full_matrix_has_four_environments_and_three_attack_settings():
    specs = benchmark_specs()
    assert len(specs) == 12
    assert {spec.environment for spec in specs} == set(ENVIRONMENTS)
    assert {spec.attack_family for spec in specs} == set(ATTACK_SETTINGS)
    for spec in specs:
        env = spec.factory()
        observation = env.reset(seed=7, options={"goal": "A"})
        assert observation.observation.shape == (8,)
        assert observation.action_mask == (True, True, True, True)
        transition = env.step(DefenderAction.TRACK)
        assert transition.action == DefenderAction.TRACK
        env.close()


def test_rule_and_ride_adapters_preserve_checkpoint_and_run_all_backends():
    before = hashlib.sha256(RIDE_CHECKPOINT.read_bytes()).hexdigest()
    for spec in benchmark_specs():
        for method, policy in (
            ("rule_active", RuleActiveDefender()),
            ("ride", RIDeDefenderAdapter(RIDE_CHECKPOINT)),
        ):
            row = evaluate_episode(
                spec, method, policy, seed=0, episode=0, episode_seed=100
            )
            assert row["method"] == method
            assert row["environment"] == spec.environment
            assert 0 <= row["calibration"] <= 1
            assert row["interception"] in (0, 1)
    assert hashlib.sha256(RIDE_CHECKPOINT.read_bytes()).hexdigest() == before


def test_result_files_aggregate_five_seeds_and_500_episodes(tmp_path):
    rows = []
    for seed in range(5):
        for episode in range(100):
            rows.append(
                {
                    "seed": seed,
                    "method": METHODS[0],
                    "environment": ENVIRONMENTS[0],
                    "attack_family": ATTACK_SETTINGS[0],
                    "interception": episode % 2,
                    "regret": 1 - (episode % 2) + 0.01,
                    "observation_cost": 0.01,
                    "calibration": 0.25,
                    "probability_B": 0.5,
                    "true_goal": "A" if episode % 2 == 0 else "B",
                    "negative_log_likelihood": float(np.log(2)),
                }
            )
    main, per_family, calibration, regret = build_results(rows, tmp_path)
    assert main[0]["seeds"] == 5
    assert main[0]["episodes"] == 500
    assert per_family[0]["episodes"] == 500
    assert calibration[0]["expected_calibration_error"] == 0
    assert regret[0]["episodes"] == 500
    for name in (
        "main_results.csv",
        "per_family_results.csv",
        "calibration_results.csv",
        "regret_analysis.csv",
    ):
        assert (tmp_path / name).exists()


def test_convergence_is_reported_without_claiming_it_unconditionally():
    curves = {
        seed: [
            {"timesteps": (index + 1) * 1000, "evaluation_success": 0.5 + seed * 0.01}
            for index in range(10)
        ]
        for seed in range(5)
    }
    rows = convergence_rows(curves)
    assert len(rows) == 5
    assert all(row["converged_at_0.02_tolerance"] == 1 for row in rows)
