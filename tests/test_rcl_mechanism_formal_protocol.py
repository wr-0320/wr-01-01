from experiments.run_rcl_mechanism_baselines import METHODS, formal_decision, write_report
from experiments.run_rcl_mechanism_formal import FORMAL_SEEDS, OUTPUT


def test_formal_seed_and_output_reservations_are_frozen():
    assert FORMAL_SEEDS == tuple(range(13001, 13011))
    assert "outputs/formal/rcl_mechanism_confirmation_v1" in OUTPUT.as_posix()


def test_formal_decision_requires_every_predeclared_condition():
    rows = []
    for metric, low, high in (
        ("hypothesis_nll", -0.04, -0.01),
        ("hypothesis_brier", -0.02, -0.001),
        ("correct_commitment", -0.019, 0.03),
    ):
        rows.append(
            {
                "comparator": "huber",
                "metric": metric,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    assert formal_decision(rows)["decision"] == "CONFIRMED"
    rows[1]["ci95_high"] = 0.0
    assert formal_decision(rows)["decision"] == "NOT_CONFIRMED"


def test_formal_report_is_marked_as_confirmation(tmp_path):
    summary = [
        {
            "method": method,
            "scope": "overall",
            "correct_commitment_mean": 0.8,
            "hypothesis_brier_mean": 0.2,
            "hypothesis_nll_mean": 0.3,
            "hypothesis_ece_mean": 0.04,
        }
        for method in METHODS
    ]
    paired = []
    for comparator in METHODS:
        if comparator == "global_dual_mixture":
            continue
        for metric in (
            "correct_commitment",
            "hypothesis_brier",
            "hypothesis_nll",
            "hypothesis_ece",
        ):
            paired.append(
                {
                    "comparator": comparator,
                    "metric": metric,
                    "difference": -0.01,
                    "ci95_low": -0.02 if metric != "correct_commitment" else -0.019,
                    "ci95_high": -0.001,
                }
            )
    decision = formal_decision(paired)
    write_report(tmp_path, summary, paired, formal=True, decision=decision)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# Dual-mechanism mixture formal confirmation v1")
    assert "frozen ten-seed formal confirmation result" in report
    assert "three-seed development evidence" not in report
    assert "Decision: **CONFIRMED**" in report
