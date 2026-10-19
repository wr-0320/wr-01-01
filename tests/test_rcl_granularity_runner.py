import csv

from experiments.run_rcl_granularity_development import METHODS, run


def test_granularity_smoke_is_paired_and_complete(tmp_path):
    output = tmp_path / "granularity"
    result = run(
        output,
        seeds=(9911,),
        episodes=1,
        likelihood_episodes=1,
        calibration_episodes=1,
        smoke=True,
    )
    with (output / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert result["episodes"] == 7 * len(METHODS)
    assert {row["controller"] for row in rows} == set(METHODS)
    scenarios = {}
    for row in rows:
        key = (row["training_seed"], row["deception_family"], row["seed"])
        scenarios.setdefault(key, set()).add(row["controller"])
    assert all(methods == set(METHODS) for methods in scenarios.values())
    assert (output / "paired_differences.csv").is_file()

