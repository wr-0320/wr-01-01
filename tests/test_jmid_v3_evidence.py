import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "development" / "jmid_v3"


def _rows(name):
    with (EVIDENCE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_v3_archive_matches_frozen_code_and_protocol():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    assert manifest["code_commit"] == "b8c729df361d68b9a7604e7a59f6aa1c8edb653f"
    assert manifest["checkout_dirty_at_start"] is False
    assert manifest["seeds"] == list(range(4401, 4411))
    assert len(manifest["candidate_graphs"]) == 105
    protocol = ROOT / manifest["protocol_path"]
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == manifest["protocol_sha256"]


def test_v3_archived_decision_and_paired_evidence_are_internally_consistent():
    decision = json.loads((EVIDENCE / "decision.json").read_text())
    assert decision["status"] == "GO_PROTOCOL_DESIGN"
    assert decision["rerun_from_saved_training_split"] is True
    rows = _rows("per_seed_metrics.csv")
    assert len(rows) == 60
    grouped = defaultdict(dict)
    for row in rows:
        grouped[int(row["seed"])][row["method"]] = row
    assert set(grouped) == set(range(4401, 4411))
    for methods in grouped.values():
        assert len(methods) == 6
        assert len({row["episode_id_sha256"] for row in methods.values()}) == 1
        for target in ("Learned_M4", "Graph_Averaged_M4"):
            assert float(methods[target]["brier"]) < float(methods["Learned_M23"]["brier"])
            assert float(methods[target]["nll"]) < float(methods["Learned_M23"]["nll"])


def test_v3_graph_recovery_explains_identical_joint_results():
    graph_rows = _rows("graph_diagnostics.csv")
    assert len(graph_rows) == 10
    assert all(row["m4_exact_recovery"] == "1" for row in graph_rows)
    assert all(row["training_rerun_reproduced"] == "1" for row in graph_rows)
    assert min(float(row["true_graph_weight"]) for row in graph_rows) > 0.999999999
    metrics = _rows("per_seed_metrics.csv")
    by_seed = defaultdict(dict)
    for row in metrics:
        by_seed[row["seed"]][row["method"]] = row
    compared = ("accuracy", "brier", "nll", "ece", "wrong_confidence", "high_confidence_error_rate")
    for methods in by_seed.values():
        for name in compared:
            assert methods["Oracle_M4"][name] == methods["Learned_M4"][name]
            assert np.isclose(
                float(methods["Oracle_M4"][name]),
                float(methods["Graph_Averaged_M4"][name]),
                rtol=0.0,
                atol=1e-9,
            )
