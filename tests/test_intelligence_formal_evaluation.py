import numpy as np
import pytest

from experiments.run_intelligence_formal_evaluation import (
    PRIMARY_METRICS,
    aggregate_across_seeds,
    binary_auroc,
    binary_f1,
    deception_metrics,
    expected_calibration_error,
    read_numeric_csv,
    run,
)


def test_binary_detection_metrics_distinguish_rank_and_threshold_quality():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    assert binary_auroc(labels, probabilities) == 1.0
    assert binary_f1(labels, probabilities) == 1.0
    assert expected_calibration_error(probabilities, labels) >= 0


def test_deception_metrics_use_honest_controls_and_deceptive_cases():
    rows = [
        {
            "deception_label": label,
            "deception_probability": probability,
            "deception_nll": 0.0,
        }
        for label, probability in ((0, 0.1), (0, 0.2), (1, 0.8), (1, 0.9))
    ]
    result = deception_metrics(rows)
    assert result["deception_auroc"] == 1.0
    assert result["deception_f1"] == 1.0
    assert result["deception_brier"] < 0.05


def test_seed_aggregation_reports_t_interval_and_episode_total():
    rows = []
    for seed in range(5):
        row = {"method": "full_ride", "scope": "overall", "episodes": 700}
        for metric in PRIMARY_METRICS:
            row[metric] = 0.5 + seed * 0.01
        rows.append(row)
    result = aggregate_across_seeds(rows, ("method", "scope"))[0]
    assert result["seeds"] == 5
    assert result["episodes"] == 3500
    assert result["correct_commitment_ci95_low"] < result["correct_commitment_mean"]
    assert result["correct_commitment_ci95_high"] > result["correct_commitment_mean"]


def test_formal_protocol_rejects_smaller_post_hoc_matrix(tmp_path):
    with pytest.raises(ValueError, match="seeds 0,1,2,3,4"):
        run(tmp_path, seeds=(0,), episodes_per_cell=100)
    with pytest.raises(ValueError, match="100 episodes per cell"):
        run(tmp_path, episodes_per_cell=10)


def test_numeric_csv_reader_preserves_identifiers(tmp_path):
    table = tmp_path / "summary.csv"
    table.write_text(
        "method,scope,correct_commitment_mean\nfull_ride,overall,0.72\n",
        encoding="utf-8",
    )
    assert read_numeric_csv(table) == [
        {
            "method": "full_ride",
            "scope": "overall",
            "correct_commitment_mean": 0.72,
        }
    ]

