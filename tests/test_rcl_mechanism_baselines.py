import csv
import json

from experiments.run_rcl_mechanism_baselines import METHODS, run


def test_mechanism_baseline_smoke_is_paired_and_records_selection(tmp_path):
    output = tmp_path / "mechanisms"
    result = run(
        output,
        seeds=(9991,),
        episodes=1,
        likelihood_episodes=1,
        calibration_episodes=1,
        smoke=True,
    )
    assert result["episodes"] == 7 * len(METHODS)
    with (output / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    scenarios = {}
    for row in rows:
        key = (row["training_seed"], row["deception_family"], row["seed"])
        scenarios.setdefault(key, set()).add(row["controller"])
    assert all(value == set(METHODS) for value in scenarios.values())
    parameters = json.loads((output / "parameters_seed9991.json").read_text())
    assert parameters["selected_student_dof"] in (3.0, 5.0, 10.0, 30.0)
    assert parameters["selected_huber_delta"] in (1.0, 1.345, 2.0, 3.0)
    assert parameters["reliability_weight_power"] == 1.0
    assert (output / "report.md").is_file()
