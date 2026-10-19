import json

import numpy as np
import pytest

from deception_rl.agents.intelligence_policies import DecoyRedPolicy
from deception_rl.envs.deceptive_intelligence import (
    BLUE_WAIT,
    DeceptiveIntelligenceEnv,
    decode_verify_action,
)
from deception_rl.intelligence.adapter import (
    audit_blue_observation,
    encode_intelligence_sequence,
)
from deception_rl.intelligence.ride import (
    HonestIntelligenceReasoner,
    IntelligenceRIDeModel,
    RIDeIntelligenceController,
    RIDeVariant,
)
from deception_rl.intelligence.ride_v2 import (
    DynamicReliabilityController,
    DynamicSourceReliabilityFilter,
    HierarchicalSourceRegionReliabilityFilter,
    MultiSourceEntropyVerify,
    MultiSourceRandomVerify,
    OracleSourceReliability,
    OracleVerifyPolicy,
    SourceRegionReliabilityController,
    extract_reliability_evidence,
    extract_source_region_evidence,
)
from deception_rl.intelligence.ride_v2.reliability import (
    ReliabilityEstimate,
    condition_observation_on_reliability,
)
from deception_rl.intelligence.ride_v2.calibrated_reliability import (
    CalibratedCellReliabilityFilter,
    LogisticCellModel,
    PlattCalibrator,
    build_cell_dataset,
    probability_metrics,
)
from deception_rl.intelligence.ride_v2.source_region_reliability import (
    SourceRegionReliabilityEstimate,
    condition_observation_on_source_region_reliability,
)
from experiments.run_intelligence_ride import collect_prefix_records
from experiments.run_intelligence_ride_v2_headroom import run
from experiments.run_intelligence_ride_v2_reliability import run as run_reliability
from experiments.run_intelligence_ride_v2_source_region import run as run_source_region
from experiments.run_intelligence_calibrated_cell_reliability import (
    run as run_calibrated_reliability,
)
from experiments.run_intelligence_calibrated_reliability_fusion import (
    run as run_calibrated_fusion,
)


def _inference(max_cycles=5):
    records = collect_prefix_records(
        ("honest",), 1, seed=930_000, split="ride_v2_test", max_cycles=max_cycles
    )
    return RIDeIntelligenceController(
        IntelligenceRIDeModel(hidden_size=8),
        HonestIntelligenceReasoner(records),
        None,
        RIDeVariant(
            "test_multi_source",
            use_deception_head=False,
            use_counterfactual=False,
            use_calibration=False,
            adaptive_commitment=False,
        ),
        max_cycles=max_cycles,
        fixed_commit_step=4,
    )


def test_oracle_reliability_is_separate_from_blue_observation():
    env = DeceptiveIntelligenceEnv(max_cycles=5)
    observations, _ = env.reset(seed=4, options={"hypothesis": 3})
    red_action = DecoyRedPolicy(multisource=True).act(observations["red"])
    diagnostic = OracleSourceReliability().inspect(env, red_action, None)
    observations, *_ = env.step({"red": red_action, "blue": BLUE_WAIT})
    filtered = OracleSourceReliability.filter_compromised_reports(
        observations["blue"], [diagnostic]
    )

    assert diagnostic.privileged_diagnostic_only is True
    assert diagnostic.compromised_cells.any()
    assert set(filtered) == set(observations["blue"])
    assert audit_blue_observation(filtered)["truth_fields_present"] is False
    encode_intelligence_sequence(filtered)

    contaminated = dict(filtered)
    contaminated["privileged_diagnostic_only"] = True
    with pytest.raises(ValueError, match="Unsafe blue observation"):
        encode_intelligence_sequence(contaminated)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MultiSourceRandomVerify(_inference(), seed=17),
        lambda: MultiSourceEntropyVerify(_inference()),
    ],
)
def test_active_baselines_use_three_distinct_verifications_before_commit(factory):
    env = DeceptiveIntelligenceEnv(max_cycles=5)
    controller = factory()
    controller.reset()
    observations, _ = env.reset(seed=12, options={"hypothesis": 0})
    cells = []
    for _ in range(3):
        action = controller.act(observations["blue"])
        cell = decode_verify_action(action)
        assert cell is not None
        cells.append(cell)
        observations, *_ = env.step({"red": 0, "blue": action})
    assert len(set(cells)) == 3


def test_oracle_rollout_does_not_mutate_live_simulator():
    env = DeceptiveIntelligenceEnv(max_cycles=5)
    observations, _ = env.reset(seed=22, options={"hypothesis": 5})
    red_action = DecoyRedPolicy(multisource=True).act(observations["red"])
    controller = OracleVerifyPolicy(_inference(), seed=91, rollouts=2)
    reports_before = env._reports.copy()
    random_before = repr(env.np_random.bit_generator.state)
    budget_before = env._verification_budget

    action = controller.act_with_oracle(
        observations["blue"], env, red_action, []
    )

    assert decode_verify_action(action) is not None
    np.testing.assert_array_equal(env._reports, reports_before)
    assert repr(env.np_random.bit_generator.state) == random_before
    assert env._verification_budget == budget_before
    assert env._step_count == 0


def test_dynamic_reliability_filter_uses_visible_five_factor_evidence():
    inference = _inference()
    env = DeceptiveIntelligenceEnv(max_cycles=5)
    observations, _ = env.reset(seed=31, options={"hypothesis": 2})
    observations, *_ = env.step({"red": 0, "blue": 0})
    evidence, times = extract_reliability_evidence(
        observations["blue"], inference.reasoner
    )
    assert evidence.shape == (1, 3, 5)
    assert times.tolist() == [0]

    samples = []
    for episode in range(2):
        for step in range(2):
            compromised = np.array(
                [(episode + step + source) % 2 for source in range(3)]
            )
            features = np.full((3, 5), 0.8 - 0.6 * compromised[:, None])
            samples.append(
                {
                    "episode_id": str(episode),
                    "step": step,
                    "features": features,
                    "compromised": compromised,
                }
            )
    reliability_filter = DynamicSourceReliabilityFilter.fit(samples)
    estimate = reliability_filter.estimate(observations["blue"], inference.reasoner)
    assert estimate.current.shape == (3,)
    assert np.all((estimate.current >= 0) & (estimate.current <= 1))

    controller = DynamicReliabilityController(inference, reliability_filter)
    action = controller.act(observations["blue"])
    assert action >= 0
    assert len(controller.info["dynamic_source_reliability"]) == 3


def test_hierarchical_source_region_filter_preserves_cell_granularity():
    inference = _inference()
    env = DeceptiveIntelligenceEnv(max_cycles=5)
    observations, _ = env.reset(seed=41, options={"hypothesis": 3})
    observations, *_ = env.step({"red": 0, "blue": 0})
    evidence, times = extract_source_region_evidence(
        observations["blue"], inference.reasoner
    )
    assert evidence.shape == (1, 3, 3, 5)
    assert times.tolist() == [0]

    samples = []
    for episode in range(2):
        for step in range(2):
            source_state = np.array(
                [(episode + step + source) % 2 for source in range(3)]
            )
            compromised_cells = np.zeros((3, 3), dtype=int)
            for source, state in enumerate(source_state):
                if state:
                    compromised_cells[source, (episode + step) % 3] = 1
            cell_features = np.full((3, 3, 5), 0.8)
            cell_features[compromised_cells.astype(bool)] = 0.2
            samples.append(
                {
                    "episode_id": str(episode),
                    "step": step,
                    "features": np.full((3, 5), 0.8 - 0.6 * source_state[:, None]),
                    "compromised": source_state,
                    "cell_features": cell_features,
                    "compromised_cells": compromised_cells,
                }
            )
    reliability_filter = HierarchicalSourceRegionReliabilityFilter.fit(samples)
    estimate = reliability_filter.estimate(observations["blue"], inference.reasoner)
    assert estimate.current.shape == (3, 3)
    assert estimate.global_history.shape[-1] == 3

    controller = SourceRegionReliabilityController(inference, reliability_filter)
    action = controller.act(observations["blue"])
    assert action >= 0
    assert np.asarray(controller.info["source_region_reliability"]).shape == (3, 3)


def test_reliability_conditioning_keeps_missing_report_values_zero():
    env = DeceptiveIntelligenceEnv(max_cycles=5)
    observations, _ = env.reset(seed=51, options={"hypothesis": 1})
    blue = observations["blue"]
    source_estimate = ReliabilityEstimate(
        current=np.full(3, 0.2),
        history=np.full((1, 3), 0.2),
        time_indices=np.array([-1]),
        evidence=np.full((1, 3, 5), 0.5),
    )
    conditioned_source = condition_observation_on_reliability(
        blue, source_estimate
    )
    np.testing.assert_array_equal(conditioned_source["reports"], blue["reports"])

    cell_estimate = SourceRegionReliabilityEstimate(
        current=np.full((3, 3), 0.2),
        history=np.full((1, 3, 3), 0.2),
        global_history=np.full((1, 3), 0.2),
        time_indices=np.array([-1]),
        evidence=np.full((1, 3, 3, 5), 0.5),
    )
    conditioned_cell = condition_observation_on_source_region_reliability(
        blue, cell_estimate
    )
    np.testing.assert_array_equal(conditioned_cell["reports"], blue["reports"])


def test_calibrated_cell_dataset_scores_only_reported_cells():
    cell_features = np.zeros((3, 3, 5), dtype=float)
    compromised = np.zeros((3, 3), dtype=int)
    compromised[1, 2] = 1
    reported = np.zeros((3, 3), dtype=bool)
    reported[0, 0] = True
    reported[1, 2] = True
    matrix, labels = build_cell_dataset(
        [
            {
                "episode_id": "one",
                "step": 0,
                "cell_features": cell_features,
                "compromised_cells": compromised,
                "reported_cells": reported,
            }
        ]
    )
    assert matrix.shape == (2, 16)
    assert labels.tolist() == [0, 1]


def test_balanced_logistic_can_be_calibrated_on_disjoint_data():
    generator = np.random.default_rng(73)
    matrix = generator.normal(size=(500, 16))
    labels = (matrix[:, 0] + 0.5 * matrix[:, 1] > 1.4).astype(int)
    model = LogisticCellModel.fit(matrix[:300], labels[:300], balanced=True)
    calibrator = PlattCalibrator.fit(
        model.decision_function(matrix[300:400]), labels[300:400]
    )
    probability = calibrator.predict(model.decision_function(matrix[400:]))
    metrics = probability_metrics(probability, labels[400:])
    assert metrics["brier_skill"] > 0
    assert metrics["auroc"] > 0.9


def test_calibrated_filter_leaves_unreported_cells_fully_reliable():
    inference = _inference()
    generator = np.random.default_rng(79)
    matrix = generator.normal(size=(200, 16))
    labels = (matrix[:, 0] > 1.0).astype(int)
    model = LogisticCellModel.fit(matrix, labels, balanced=False)
    reliability_filter = CalibratedCellReliabilityFilter(model)
    env = DeceptiveIntelligenceEnv(max_cycles=5)
    observations, _ = env.reset(seed=81, options={"hypothesis": 2})
    observations, *_ = env.step({"red": 0, "blue": 0})
    estimate = reliability_filter.estimate(observations["blue"], inference.reasoner)
    visible = np.asarray(observations["blue"]["report_mask"], dtype=bool)[0]
    assert np.all(estimate.current[~visible] == 1.0)
    assert np.all((estimate.current[visible] >= 0) & (estimate.current[visible] <= 1))


def test_development_runner_writes_headroom_and_boundary_outputs(tmp_path):
    result = run(
        tmp_path,
        seeds=(901,),
        episodes_per_family_seed=1,
        epochs=1,
        train_episodes_per_hypothesis=1,
        validation_episodes_per_hypothesis=1,
        hidden_size=8,
        max_cycles=5,
        oracle_rollouts=1,
    )
    assert result["formal_results_replaced"] is False
    for name in (
        "oracle_headroom.md",
        "voi_analysis.csv",
        "family_headroom.csv",
        "paired_seed_headroom.csv",
        "information_boundary_audit.json",
        "go_no_go.json",
    ):
        assert (tmp_path / name).exists()
    audit = json.loads((tmp_path / "information_boundary_audit.json").read_text())
    assert audit["formal_results_unchanged"] is True
    assert audit["oracle_fields_inserted_into_blue_observation"] is False
    assert audit["terminal_truth_used_by_ordinary_defender"] is False
    assert audit["ordinary_policy_calls"] > 0
    assert audit["privileged_oracle_calls"] > 0


def test_reliability_development_runner_keeps_labels_out_of_runtime(tmp_path):
    result = run_reliability(
        tmp_path,
        seeds=(907,),
        episodes_per_family_seed=1,
        reliability_train_episodes_per_hypothesis=1,
        reliability_validation_episodes_per_hypothesis=1,
        epochs=1,
        train_episodes_per_hypothesis=1,
        validation_episodes_per_hypothesis=1,
        hidden_size=8,
        max_cycles=5,
    )
    assert result["formal_results_replaced"] is False
    assert result["oracle_headroom_results_replaced"] is False
    audit = json.loads((tmp_path / "information_boundary_audit.json").read_text())
    assert audit["simulator_labels_used_at_runtime"] is False
    assert audit["simulator_labels_used_for_offline_filter_fit"] is True
    assert audit["formal_results_unchanged"] is True
    assert audit["headroom_results_unchanged"] is True
    assert (tmp_path / "reliability_development_report.md").exists()


def test_source_region_runner_uses_new_seeds_and_preserves_frozen_outputs(tmp_path):
    result = run_source_region(
        tmp_path,
        seeds=(1001,),
        episodes_per_family_seed=1,
        reliability_train_episodes_per_hypothesis=1,
        reliability_validation_episodes_per_hypothesis=1,
        epochs=1,
        train_episodes_per_hypothesis=1,
        validation_episodes_per_hypothesis=1,
        hidden_size=8,
        max_cycles=5,
    )
    assert result["formal_results_replaced"] is False
    assert result["prior_development_results_replaced"] is False
    audit = json.loads((tmp_path / "information_boundary_audit.json").read_text())
    assert audit["simulator_labels_used_at_runtime"] is False
    assert audit["simulator_cell_labels_used_for_offline_fit"] is True
    assert audit["formal_results_unchanged"] is True
    assert audit["headroom_results_unchanged"] is True
    assert audit["source_reliability_results_unchanged"] is True
    assert audit["source_region_v1_results_unchanged"] is True
    assert (tmp_path / "source_region_development_report.md").exists()


def test_calibrated_reliability_runner_is_isolated_from_fusion(tmp_path):
    result = run_calibrated_reliability(
        tmp_path,
        seeds=(1717,),
        train_episodes_per_hypothesis=2,
        calibration_episodes_per_hypothesis=1,
        evaluation_episodes_per_hypothesis=1,
        max_cycles=5,
    )
    assert result["belief_fusion_evaluated"] is False
    audit = json.loads((tmp_path / "information_boundary_audit.json").read_text())
    assert audit["reported_cells_only"] is True
    assert audit["predictor_connected_to_belief_fusion"] is False
    assert audit["predictor_connected_to_verify"] is False
    assert audit["source_region_corrected_results_unchanged"] is True
    assert (tmp_path / "calibrated_reliability_report.md").exists()


def test_calibrated_fusion_runner_uses_no_verification(tmp_path):
    result = run_calibrated_fusion(
        tmp_path,
        seeds=(2121,),
        episodes_per_family_seed=1,
        reliability_train_episodes_per_hypothesis=2,
        reliability_calibration_episodes_per_hypothesis=1,
        epochs=1,
        train_episodes_per_hypothesis=1,
        validation_episodes_per_hypothesis=1,
        hidden_size=8,
        max_cycles=5,
    )
    assert result["verification_enabled"] is False
    audit = json.loads((tmp_path / "information_boundary_audit.json").read_text())
    assert audit["simulator_labels_used_at_runtime"] is False
    assert audit["calibrated_predictor_results_unchanged"] is True
    assert (tmp_path / "calibrated_fusion_report.md").exists()
