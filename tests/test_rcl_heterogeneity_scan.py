import csv
import json

import numpy as np

from experiments.run_rcl_heterogeneity_scan import METHODS, PROFILES, run


def test_heterogeneity_scan_smoke_is_complete_and_audited(tmp_path):
    output = tmp_path / "scan"
    result = run(
        output,
        seeds=(9299,),
        episodes=1,
        likelihood_episodes=1,
        calibration_episodes=1,
        smoke=True,
    )
    assert result["episodes"] == len(PROFILES) * len(METHODS) * 7
    with (output / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["profile"] for row in rows} == set(PROFILES)
    assert {row["controller"] for row in rows} == set(METHODS)
    with (output / "heterogeneity_audit.csv").open(newline="", encoding="utf-8") as handle:
        audit = list(csv.DictReader(handle))
    assert len(audit) == len(PROFILES) * 2 * 9
    assert np.allclose(
        [float(row["configured_activation_mean"]) for row in audit], 0.6
    )
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    assert decision["formal_claim_established"] is False
