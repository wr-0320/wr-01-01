import csv

from experiments.run_exposure_diversity import run_exposure_diversity
from experiments.plot_exposure_diversity import plot_exposure_diversity


def test_exposure_diversity_controls_and_outputs_curve(tmp_path):
    result_dir = tmp_path / "exposure"
    outputs = run_exposure_diversity(
        output_dir=result_dir,
        base_episodes_per_policy=2,
        scripted_low_per_variant=1,
        scripted_high_per_variant=4,
        seed_count=1,
        test_episodes_per_scenario=2,
        step_commit=5,
        seed=7,
        epochs=2,
        patience=2,
        batch_size=16,
        device="cpu",
    )
    figure_path = plot_exposure_diversity(
        outputs["diversity"], tmp_path / "figures" / "robustness.png"
    )

    with outputs["training_metrics"].open(newline="", encoding="utf-8") as csv_file:
        training_rows = {row["model"]: row for row in csv.DictReader(csv_file)}
    assert training_rows["C1"]["deception_training_episodes"] == "4"
    assert training_rows["C2"]["deception_training_episodes"] == "16"
    assert training_rows["C2"]["training_family_count"] == "1"
    assert training_rows["C3"]["deception_training_episodes"] == "16"
    assert training_rows["C3"]["training_family_count"] == "4"
    with outputs["diversity"].open(newline="", encoding="utf-8") as csv_file:
        diversity_rows = {row["model"]: row for row in csv.DictReader(csv_file)}
    assert diversity_rows["C2"]["training_repeats"] == "3"
    assert "robustness_scripted_extrapolation" in diversity_rows["C2"]
    assert "robustness_new_families" in diversity_rows["C2"]
    assert figure_path.is_file() and figure_path.stat().st_size > 0
