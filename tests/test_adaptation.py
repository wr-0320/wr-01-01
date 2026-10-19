import csv

from experiments.run_adaptation import run_adaptation


def test_adaptation_generates_policy_data_models_and_comparison_csvs(tmp_path):
    outputs = run_adaptation(
        output_dir=tmp_path / "adaptation",
        train_episodes_per_policy=2,
        test_episodes=2,
        seen_deception_steps=2,
        unseen_deception_steps=3,
        step_commit=2,
        seed=13,
        epochs=3,
        patience=2,
    )

    with outputs["training_data"].open(newline="", encoding="utf-8") as csv_file:
        training_rows = list(csv.DictReader(csv_file))
    with outputs["summary"].open(newline="", encoding="utf-8") as csv_file:
        summary_rows = list(csv.DictReader(csv_file))
    with outputs["episodes"].open(newline="", encoding="utf-8") as csv_file:
        episode_rows = list(csv.DictReader(csv_file))

    assert {row["deceiver_policy"] for row in training_rows} == {
        "honest_shortest_path",
        "random_policy",
        "scripted_deception",
    }
    scripted_training_rows = [
        row for row in training_rows if row["deceiver_policy"] == "scripted_deception"
    ]
    assert {row["deception_steps"] for row in scripted_training_rows} == {"2"}
    assert {row["training_policies"] for row in summary_rows} == {
        "honest_shortest_path+random_policy",
        "honest_shortest_path+random_policy+scripted_deception",
    }
    assert {row["condition"] for row in summary_rows} == {
        "seen_scripted_deception",
        "unseen_scripted_deception",
    }
    assert len(summary_rows) == 4
    assert len(episode_rows) == 8
    assert all(outputs[name].is_file() for name in ("model_a", "model_b", "training_metrics"))
