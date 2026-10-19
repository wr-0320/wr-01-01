import csv

from experiments.evaluate_paper import _from_family_generalization, _from_h3
from experiments.plot_paper import plot_paper, wilson_interval


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_h3_reference_cost_is_only_attributed_to_scripted_policy(tmp_path):
    source = tmp_path / "h3.csv"
    shared = {
        "episode": 0,
        "true_goal": "A",
        "belief_at_commitment": '{"A":0.8,"B":0.2}',
        "decision_correct": 1,
        "deceiver_success": 0,
        "deception_cost": 4,
    }
    _write_csv(
        source,
        [
            {**shared, "deceiver_policy": "honest_shortest_path"},
            {**shared, "deceiver_policy": "scripted_deception"},
        ],
    )

    rows = _from_h3(source, "Stage 3", "rule")

    assert rows[0]["deception_cost"] == ""
    assert rows[1]["deception_cost"] == 4.0


def test_family_rows_normalize_to_unseen_robustness(tmp_path):
    source = tmp_path / "families.csv"
    _write_csv(
        source,
        [
            {
                "model": "ModelB",
                "deception_family": "detour_deception",
                "condition": "unseen_family_detour_deception",
                "deception_parameters": "lateral_detour_steps=1",
                "deception_steps": "",
                "episode": 0,
                "true_goal": "A",
                "belief_accuracy": 1.0,
                "deceiver_success": 0,
                "commitment_accuracy": 1,
            }
        ],
    )

    [row] = _from_family_generalization(source, {"ModelA": 0, "ModelB": 4000})

    assert row["stage"] == "Stage 6B"
    assert row["robustness_under_unseen_deception"] == 1.0
    assert row["deception_exposure"] == 4000


def test_wilson_interval_bounds_binary_rate():
    lower, upper = wilson_interval(50, 100)

    assert lower < 0.5 < upper
    assert wilson_interval(0, 10)[1] < 0.5


def test_plot_script_writes_all_requested_figures(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    summary_rows = []
    episode_rows = []
    for model in ("ModelA", "ModelB"):
        for step in range(1, 6):
            condition = (
                f"seen_scripted_deception_steps_{step}"
                if step < 5
                else "unseen_scripted_deception_steps_5"
            )
            accuracy = 0.4 + 0.1 * (step - 1)
            success = 1 - accuracy
            summary_rows.append(
                {
                    "stage": "Stage 6",
                    "experiment": "defender_adaptation",
                    "model": model,
                        "deception_family": "",
                        "deception_parameters": "",
                    "deception_steps": step,
                    "condition": condition,
                    "commitment_accuracy": accuracy,
                    "deception_success_rate": success,
                    "robustness_under_unseen_deception": accuracy if step == 5 else "",
                    "deception_exposure": 4000 if model == "ModelB" else 0,
                }
            )
            episode_rows.append(
                {
                    "stage": "Stage 6",
                    "model": model,
                        "deception_family": "",
                        "deception_parameters": "",
                        "deception_steps": step,
                    "condition": condition,
                    "commitment_accuracy": int(accuracy >= 0.5),
                    "deceiver_success": int(success >= 0.5),
                }
            )
        summary_rows.append(
            {
                "stage": "Stage 5",
                "experiment": "defender_adaptation",
                "model": model,
                    "deception_family": "",
                    "deception_parameters": "",
                "deception_steps": 3,
                "condition": "unseen_scripted_deception",
                "commitment_accuracy": 0.5,
                "deception_success_rate": 0.5,
                "robustness_under_unseen_deception": 0.5,
                "deception_exposure": 40 if model == "ModelB" else 0,
            }
        )
        episode_rows.append(
            {
                "stage": "Stage 5",
                "model": model,
                    "deception_family": "",
                    "deception_parameters": "",
                    "deception_steps": 3,
                "condition": "unseen_scripted_deception",
                "commitment_accuracy": 1,
                "deceiver_success": 0,
            }
        )
        family_cases = [
            ("scripted_extrapolation", "6"),
            ("scripted_extrapolation", "7"),
            ("delayed_deception", ""),
            ("detour_deception", ""),
            ("multi_switch_deception", ""),
        ]
        for family, steps in family_cases:
            condition = f"unseen_family_{family}_{steps or 'default'}"
            summary_rows.append(
                {
                    "stage": "Stage 6B",
                    "experiment": "unseen_deception_families",
                    "model": model,
                    "deception_family": family,
                    "deception_parameters": steps,
                    "deception_steps": steps,
                    "condition": condition,
                    "commitment_accuracy": 0.5,
                    "deception_success_rate": 0.5,
                    "robustness_under_unseen_deception": 0.5,
                    "deception_exposure": 4000 if model == "ModelB" else 0,
                }
            )
            episode_rows.append(
                {
                    "stage": "Stage 6B",
                    "model": model,
                    "deception_family": family,
                    "deception_parameters": steps,
                    "deception_steps": steps,
                    "condition": condition,
                    "commitment_accuracy": 1,
                    "deceiver_success": 0,
                }
            )
    _write_csv(evaluation_dir / "evaluation_summary.csv", summary_rows)
    _write_csv(evaluation_dir / "evaluation_episodes.csv", episode_rows)

    figures = plot_paper(evaluation_dir, tmp_path / "figures")

    assert len(figures) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in figures)
