import csv

import torch

from experiments.run_deception_family_generalization import run_family_generalization
from models.intent_gru import IntentGRU


def _checkpoint(path):
    model = IntentGRU(hidden_size=8)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_size": 8,
            "num_layers": 1,
            "grid_size": 7,
            "max_time": 30,
        },
        path,
    )


def test_family_evaluation_keeps_step6_and_step7_separate(tmp_path):
    model_a = tmp_path / "model_a.pt"
    model_b = tmp_path / "model_b.pt"
    _checkpoint(model_a)
    _checkpoint(model_b)
    outputs = run_family_generalization(
        model_a_checkpoint=model_a,
        model_b_checkpoint=model_b,
        output_dir=tmp_path / "family-results",
        episodes_per_scenario=2,
        step_commit=5,
        seed=2,
    )

    with outputs["summary"].open(newline="", encoding="utf-8") as csv_file:
        summary_rows = list(csv.DictReader(csv_file))
    with outputs["episodes"].open(newline="", encoding="utf-8") as csv_file:
        episode_rows = list(csv.DictReader(csv_file))

    extrapolation_steps = {
        row["deception_steps"]
        for row in summary_rows
        if row["deception_family"] == "scripted_extrapolation"
    }
    assert extrapolation_steps == {"6", "7"}
    assert len(summary_rows) == 10
    assert len(episode_rows) == 20
    assert {row["deception_family"] for row in summary_rows} == {
        "scripted_extrapolation",
        "delayed_deception",
        "detour_deception",
        "multi_switch_deception",
    }