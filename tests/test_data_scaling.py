import csv

from experiments.run_data_scaling import run_data_scaling


def test_data_scaling_generates_all_variants_and_seed_groups(tmp_path):
    outputs = run_data_scaling(
        output_dir=tmp_path / "scaled",
        episodes_per_policy=4,
        seed_count=2,
        test_episodes=2,
        training_deception_steps=(1, 2),
        unseen_deception_steps=(3,),
        step_commit=3,
        seed=21,
        epochs=2,
        patience=2,
        batch_size=16,
        device="cpu",
    )

    with outputs["training_data"].open(newline="", encoding="utf-8") as csv_file:
        training_rows = list(csv.DictReader(csv_file))
    with outputs["summary"].open(newline="", encoding="utf-8") as csv_file:
        summary_rows = list(csv.DictReader(csv_file))

    assert len(training_rows) == 16
    assert {row["seed_group"] for row in training_rows} == {"0", "1"}
    scripted_rows = [
        row for row in training_rows if row["deceiver_policy"] == "scripted_deception"
    ]
    assert {row["deception_steps"] for row in scripted_rows} == {"1", "2"}
    assert len(summary_rows) == 6
    assert {row["deception_steps"] for row in summary_rows} == {"1", "2", "3"}
    assert all(outputs[name].is_file() for name in ("model_a", "model_b", "episodes"))
