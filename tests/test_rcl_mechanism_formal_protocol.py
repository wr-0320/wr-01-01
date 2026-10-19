from experiments.run_rcl_mechanism_baselines import formal_decision
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
