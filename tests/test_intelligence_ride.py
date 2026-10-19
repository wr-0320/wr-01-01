import csv
import json

import numpy as np
import pytest
import torch

from deception_rl.envs.deceptive_intelligence import (
    NUM_BLUE_ACTIONS,
    NUM_HYPOTHESES,
    DeceptiveIntelligenceEnv,
)
from deception_rl.intelligence.adapter import (
    INTELLIGENCE_FEATURE_SIZE,
    audit_blue_observation,
    encode_intelligence_sequence,
)
from deception_rl.intelligence.ride import (
    HonestIntelligenceReasoner,
    IntelligenceRIDeModel,
    RIDeIntelligenceController,
    RIDeVariant,
    TemperatureCalibrator,
)
from experiments.run_intelligence_ride import collect_prefix_records, run


def test_adapter_uses_only_blue_visible_fields_and_rejects_truth_injection():
    env = DeceptiveIntelligenceEnv()
    observations, _ = env.reset(seed=0, options={"hypothesis": 4})
    audit = audit_blue_observation(observations["blue"])
    assert not audit["truth_fields_present"]
    sequence = encode_intelligence_sequence(observations["blue"])
    assert sequence.shape == (1, INTELLIGENCE_FEATURE_SIZE)

    contaminated = dict(observations["blue"])
    contaminated["true_hypothesis"] = 4
    with pytest.raises(ValueError, match="Unsafe blue observation"):
        encode_intelligence_sequence(contaminated)


def test_adapter_encodes_temporal_reports_without_offline_labels():
    env = DeceptiveIntelligenceEnv()
    observations, _ = env.reset(seed=3, options={"hypothesis": 2})
    observations, *_ = env.step({"red": 0, "blue": 0})
    observations, *_ = env.step({"red": 0, "blue": 0})
    first = encode_intelligence_sequence(observations["blue"])
    harness_record = {
        "sequence": first,
        "hypothesis_label": 2,
        "deception_label": 0,
    }
    assert harness_record["sequence"].shape == (2, INTELLIGENCE_FEATURE_SIZE)
    np.testing.assert_array_equal(
        harness_record["sequence"],
        encode_intelligence_sequence(observations["blue"]),
    )


def test_two_head_model_predicts_joint_beliefs_from_variable_length_sequences():
    model = IntelligenceRIDeModel(hidden_size=8)
    sequences = torch.rand(3, 4, INTELLIGENCE_FEATURE_SIZE)
    hypothesis, deception = model(sequences, torch.tensor([4, 2, 3]))
    assert hypothesis.shape == (3, NUM_HYPOTHESES)
    assert deception.shape == (3, 2)


def test_honest_reasoner_builds_every_hypothesis_counterfactual():
    records = collect_prefix_records(
        ("honest",), 1, seed=10, split="reasoner", max_cycles=5
    )
    reasoner = HonestIntelligenceReasoner(records)
    sequence = records[0]["sequence"]
    belief = np.full(NUM_HYPOTHESES, 1 / NUM_HYPOTHESES)
    result = reasoner.score(sequence, belief)
    assert len(result["distance_by_hypothesis"]) == NUM_HYPOTHESES
    assert len(result["honest_likelihood_by_hypothesis"]) == NUM_HYPOTHESES
    assert 0 <= result["counterfactual_deception_score"] <= 1


def test_temperature_calibration_is_fitted_without_changing_model_inputs():
    calibrator = TemperatureCalibrator()
    hypothesis_logits = torch.randn(12, NUM_HYPOTHESES)
    deception_logits = torch.randn(12, 2)
    hypothesis_labels = torch.arange(12) % NUM_HYPOTHESES
    deception_labels = torch.arange(12) % 2
    metrics = calibrator.fit(
        hypothesis_logits,
        deception_logits,
        hypothesis_labels,
        deception_labels,
        steps=3,
    )
    calibrated = calibrator(hypothesis_logits, deception_logits)
    assert calibrated[0].shape == hypothesis_logits.shape
    assert metrics["hypothesis_temperature"] > 0
    assert metrics["deception_temperature"] > 0


def test_full_ride_controller_returns_valid_action_from_blue_observation_only():
    records = collect_prefix_records(
        ("honest",), 1, seed=20, split="controller", max_cycles=5
    )
    reasoner = HonestIntelligenceReasoner(records)
    controller = RIDeIntelligenceController(
        IntelligenceRIDeModel(hidden_size=8),
        reasoner,
        None,
        RIDeVariant("test_full_ride", use_calibration=False),
        max_cycles=5,
    )
    env = DeceptiveIntelligenceEnv(max_cycles=5)
    observations, _ = env.reset(seed=1, options={"hypothesis": 5})
    action = controller.act(observations["blue"])
    assert 0 <= action < NUM_BLUE_ACTIONS
    assert "hypothesis_belief" in controller.info
    assert "counterfactual_deception_score" in controller.info


def test_smoke_ablation_writes_family_metrics_and_boundary_audit(tmp_path):
    summary = run(
        tmp_path,
        epochs=1,
        train_episodes_per_hypothesis=1,
        validation_episodes_per_hypothesis=1,
        evaluation_episodes_per_hypothesis=1,
        hidden_size=8,
        max_cycles=5,
    )
    assert summary
    with (tmp_path / "summary_by_deception_family.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (tmp_path / "ablation_overall.csv").open(newline="") as handle:
        ablation_rows = list(csv.DictReader(handle))
    assert {row["controller"] for row in rows} >= {
        "single_report",
        "multi_source_belief",
        "belief_plus_deception",
        "belief_deception_counterfactual",
        "full_ride",
        "bayesian_fusion",
        "pomcp_like",
        "active_inference",
    }
    required_metrics = {
        "interception",
        "false_commitment",
        "decision_latency",
        "information_cost",
        "regret",
        "calibration_error",
    }
    assert required_metrics <= set(rows[0])
    with (tmp_path / "episodes.csv").open(newline="") as handle:
        episode_rows = list(csv.DictReader(handle))
    assert {"hypothesis_nll", "deception_nll", "deception_label"} <= set(
        episode_rows[0]
    )
    assert len(ablation_rows) == 15
    assert {row["scope"] for row in ablation_rows} == {"overall", "seen", "unseen"}
    audit = json.loads((tmp_path / "information_boundary_audit.json").read_text())
    assert audit["evaluated_policy_inputs"] > 0
    assert audit["terminal_truth_used_as_policy_input"] is False
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["large_scale_training"] is False
    assert config["legacy_results_replaced"] is False

