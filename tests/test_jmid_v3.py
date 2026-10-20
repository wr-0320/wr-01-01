import csv
import inspect
import json

import numpy as np
import pytest

from experiments.jmid_v3.generator import ObservedDataset, diagnostics, generate_v3
from experiments.jmid_v3.graphs import edge_distance, enumerate_matchings, graph_id
from experiments.jmid_v3.models import DependenceModel, JointModel
from experiments.jmid_v3.run import (
    graph_weights, run, select_graph, validation_nlls,
)


def test_all_105_matchings_are_unique_and_distances_are_correct():
    graphs = enumerate_matchings()
    assert len(graphs) == len(set(graphs)) == 105
    assert [graph_id(g) for g in graphs] == sorted(graph_id(g) for g in graphs)
    truth = graphs[0]
    distances = {edge_distance(graph, truth) for graph in graphs}
    assert distances == {0.0, 0.5, 0.75, 1.0}
    for graph in graphs:
        shared = len(set(graph) & set(truth))
        assert edge_distance(graph, truth) == 1.0 - shared / 4


def test_generator_preserves_content_missingness_and_directed_copy():
    truth = enumerate_matchings()[17]
    data = generate_v3(30_000, 101, truth)
    report = diagnostics(data)
    assert data.observed.y.shape == (30_000, 8)
    assert data.true_graph == truth
    assert report["missing_given_raw1"] > report["missing_given_raw0"] + 0.25
    assert abs(report["copy_rate"] - 0.85) < 0.015
    assert report["within_state_report_correlation"] > 0.55


def test_learned_scoring_accepts_observed_training_data_only():
    signature = inspect.signature(validation_nlls)
    assert list(signature.parameters) == ["model_class", "structure_fit", "validation", "candidates"]
    truth = enumerate_matchings()[3]
    generated = generate_v3(300, 102, truth)
    observed = generated.observed
    fit = ObservedDataset(observed.episode_id[:220], observed.z[:220], observed.y[:220])
    validation = ObservedDataset(observed.episode_id[220:], observed.z[220:], observed.y[220:])
    candidates = enumerate_matchings()
    m3 = validation_nlls(DependenceModel, fit, validation, candidates)
    m4 = validation_nlls(JointModel, fit, validation, candidates)
    assert m3.shape == m4.shape == (105,)
    assert np.isfinite(m3).all() and np.isfinite(m4).all()


def test_selection_and_averaging_are_frozen_validation_rules():
    candidates = enumerate_matchings()
    scores = np.linspace(0.4, 0.8, len(candidates)); scores[9] = 0.1
    index, graph = select_graph(scores, candidates)
    weights = graph_weights(scores, 1250)
    assert index == 9 and graph == candidates[9]
    assert np.isclose(weights.sum(), 1.0) and (weights >= 0).all()
    assert int(np.argmax(weights)) == 9
    tied = np.ones(len(candidates))
    assert select_graph(tied, candidates)[0] == 0


def test_smoke_run_is_paired_aggregates_mismatch_and_refuses_overwrite(tmp_path):
    output = tmp_path / "v3"
    result = run(
        output, seeds=(88,), train_episodes=180, validation_episodes=45,
        test_episodes=200, smoke=True,
    )
    assert result == {
        "per_seed_rows": 6, "validation_rows": 105,
        "mismatch_rows": 3, "decision": "SMOKE_ONLY",
    }
    with (output / "per_seed_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert len({r["episode_id_sha256"] for r in rows}) == 1
    assert len({r["episode_count"] for r in rows}) == 1
    with (output / "fixed_mismatch_per_seed.csv").open(newline="", encoding="utf-8") as handle:
        mismatch = list(csv.DictReader(handle))
    assert {float(r["injected_mismatch_distance"]) for r in mismatch} == {0.5, 0.75, 1.0}
    assert all(int(r["candidate_count"]) > 0 for r in mismatch)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["candidate_graphs"]) == 105
    assert len(manifest["protocol_sha256"]) == 64
    assert manifest["seeds"] == [88]
    with (output / "graph_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        graph_rows = list(csv.DictReader(handle))
    assert graph_rows[0]["training_rerun_reproduced"] == "1"
    with pytest.raises(FileExistsError):
        run(output, seeds=(88,), smoke=True)
