"""Run the single frozen JMID v3 graph-uncertainty diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import subprocess
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.jmid_v1.core import metrics
from .generator import ObservedDataset, diagnostics, generate_v3
from .graphs import Graph, edge_distance, enumerate_matchings, graph_id
from .models import (
    DependenceModel,
    FixedComposition,
    JointModel,
    MissingAwareModel,
    NaiveModel,
    binary_nll,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs" / "jmid_graph_uncertainty_protocol_v3.md"
DEFAULT_OUTPUT = ROOT / "outputs" / "development" / "jmid_v3"
SEEDS = tuple(range(4401, 4411))
METHODS = (
    "M2_missing",
    "Learned_M3",
    "Learned_M23",
    "Oracle_M4",
    "Learned_M4",
    "Graph_Averaged_M4",
)
PRIMARY_DEPLOYABLE = ("Learned_M4", "Graph_Averaged_M4")
METRIC_NAMES = (
    "accuracy", "brier", "nll", "ece",
    "wrong_confidence", "high_confidence_error_rate",
)
T_CRITICAL_DF9 = 2.2621571628540993


def _subset(data: ObservedDataset, start: int, stop: int) -> ObservedDataset:
    return ObservedDataset(
        episode_id=data.episode_id[start:stop], z=data.z[start:stop], y=data.y[start:stop]
    )


def validation_nlls(
    model_class, structure_fit: ObservedDataset, validation: ObservedDataset,
    candidates: tuple[Graph, ...],
) -> np.ndarray:
    """Score graph candidates using observed training data only."""
    return np.asarray([
        binary_nll(validation.z, model_class(graph).fit(structure_fit).predict_proba(validation.y))
        for graph in candidates
    ])


def select_graph(scores: np.ndarray, candidates: tuple[Graph, ...]) -> tuple[int, Graph]:
    if scores.shape != (len(candidates),):
        raise ValueError("one validation score is required per candidate")
    index = int(np.argmin(scores))  # candidate order is the frozen tie breaker
    return index, candidates[index]


def graph_weights(scores: np.ndarray, validation_episodes: int) -> np.ndarray:
    if validation_episodes <= 0:
        raise ValueError("validation_episodes must be positive")
    shifted = -float(validation_episodes) * (scores - float(scores.min()))
    shifted -= float(shifted.max())
    weights = np.exp(shifted)
    return weights / weights.sum()


def _git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip())
    return commit, dirty


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, selected in grouped.items():
        record = dict(zip(keys, values)); record["seeds"] = len(selected)
        for name in METRIC_NAMES:
            array = np.asarray([float(row[name]) for row in selected])
            record[f"{name}_mean"] = float(array.mean())
            record[f"{name}_sd"] = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        output.append(record)
    return output


def _paired(rows: list[dict], baseline: str = "Learned_M23") -> list[dict]:
    by_method = {method: {int(r["seed"]): r for r in rows if r["method"] == method} for method in METHODS}
    output = []
    for method in ("Oracle_M4", *PRIMARY_DEPLOYABLE):
        shared = sorted(set(by_method[method]) & set(by_method[baseline]))
        for metric_name in ("accuracy", "brier", "nll"):
            differences = np.asarray([
                float(by_method[method][seed][metric_name])
                - float(by_method[baseline][seed][metric_name]) for seed in shared
            ])
            mean = float(differences.mean())
            if len(differences) == 10:
                half = T_CRITICAL_DF9 * float(differences.std(ddof=1)) / math.sqrt(10)
                low, high = mean - half, mean + half
            else:
                low = high = float("nan")
            output.append({
                "method": method, "baseline": baseline, "metric": metric_name,
                "seeds": len(shared), "mean_difference": mean,
                "ci95_low": low, "ci95_high": high,
                "lower_seed_count": int((differences < 0).sum()),
                "higher_seed_count": int((differences > 0).sum()),
            })
    return output


def _decision(paired: list[dict], full_run: bool, training_rerun_reproduced: bool) -> dict:
    if not full_run:
        return {"status": "SMOKE_ONLY", "reason": "No frozen decision is made from a smoke run."}
    indexed = {(r["method"], r["metric"]): r for r in paired}
    passed = []
    mean_better = []
    fewer_than_six = []
    for method in PRIMARY_DEPLOYABLE:
        brier, nll, accuracy = (indexed[(method, name)] for name in ("brier", "nll", "accuracy"))
        method_pass = (
            brier["ci95_high"] < 0 and nll["ci95_high"] < 0
            and brier["lower_seed_count"] >= 8 and nll["lower_seed_count"] >= 8
            and accuracy["ci95_low"] > -0.01 and training_rerun_reproduced
        )
        if method_pass:
            passed.append(method)
        if brier["mean_difference"] < 0 and nll["mean_difference"] < 0:
            mean_better.append(method)
        if brier["lower_seed_count"] < 6 or nll["lower_seed_count"] < 6:
            fewer_than_six.append(method)
    oracle = {name: indexed[("Oracle_M4", name)] for name in ("accuracy", "brier", "nll")}
    oracle_advantage = oracle["brier"]["mean_difference"] < 0 and oracle["nll"]["mean_difference"] < 0
    if passed:
        status = "GO_PROTOCOL_DESIGN"
        reason = f"Deployable gate passed by: {', '.join(passed)}."
    elif not mean_better or len(fewer_than_six) == len(PRIMARY_DEPLOYABLE):
        status = "STOP_COMPLEX_JOINT_TABLE"
        reason = "Neither deployable method met the frozen minimum empirical rule."
    elif oracle_advantage:
        status = "KNOWN_GRAPH_ONLY"
        reason = "Oracle improved mean probability metrics, while deployable methods failed the continue gate."
    else:
        status = "INCONCLUSIVE"
        reason = "The result matches none of the frozen GO, STOP, or known-graph-only rules."
    return {
        "status": status, "reason": reason, "passed_deployable_methods": passed,
        "oracle_mean_probability_advantage": oracle_advantage,
        "rerun_from_saved_training_split": training_rerun_reproduced,
        "note": "Exploratory diagnostic; this decision does not establish novelty or formal evidence.",
    }


def _true_graph(seed: int, candidates: tuple[Graph, ...]) -> Graph:
    rng = np.random.default_rng(seed + 88173)
    return candidates[int(rng.integers(len(candidates)))]


def run(
    output: Path = DEFAULT_OUTPUT, *, seeds=SEEDS, train_episodes: int = 5000,
    validation_episodes: int = 1250, test_episodes: int = 5000, smoke: bool = False,
) -> dict:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    candidates = enumerate_matchings()
    if len(candidates) != 105:
        raise RuntimeError("frozen graph space must contain 105 candidates")
    if smoke:
        train_episodes = min(train_episodes, 240)
        validation_episodes = min(validation_episodes, 60)
        test_episodes = min(test_episodes, 280)
    if not 0 < validation_episodes < train_episodes:
        raise ValueError("validation split must be nonempty and smaller than training data")

    commit, dirty = _git_state()
    if not smoke and dirty:
        raise RuntimeError("full v3 evidence run requires a clean committed checkout")
    fit_episodes = train_episodes - validation_episodes
    per_seed, graph_rows, validation_rows = [], [], []
    mismatch_rows, diagnostic_rows, runtime_rows = [], [], []

    for seed in seeds:
        started = time.perf_counter(); tracemalloc.start()
        truth = _true_graph(int(seed), candidates)
        train_generated = generate_v3(train_episodes, int(seed) * 10 + 1, truth)
        test_generated = generate_v3(test_episodes, int(seed) * 10 + 2, truth)
        train = train_generated.observed; test = test_generated.observed
        structure_fit = _subset(train, 0, fit_episodes)
        validation = _subset(train, fit_episodes, train_episodes)

        m3_scores = validation_nlls(DependenceModel, structure_fit, validation, candidates)
        m4_scores = validation_nlls(JointModel, structure_fit, validation, candidates)
        m3_index, m3_graph = select_graph(m3_scores, candidates)
        m4_index, m4_graph = select_graph(m4_scores, candidates)
        weights = graph_weights(m4_scores, validation_episodes)
        highest_index = int(np.argmax(weights))
        # Frozen reproducibility check: repeat every graph-selection and averaging
        # calculation from the saved training split, before any test prediction.
        m3_scores_rerun = validation_nlls(DependenceModel, structure_fit, validation, candidates)
        m4_scores_rerun = validation_nlls(JointModel, structure_fit, validation, candidates)
        m3_rerun_index, _ = select_graph(m3_scores_rerun, candidates)
        m4_rerun_index, _ = select_graph(m4_scores_rerun, candidates)
        weights_rerun = graph_weights(m4_scores_rerun, validation_episodes)
        rerun_reproduced = bool(
            m3_rerun_index == m3_index and m4_rerun_index == m4_index
            and np.array_equal(m3_scores_rerun, m3_scores)
            and np.array_equal(m4_scores_rerun, m4_scores)
            and np.array_equal(weights_rerun, weights)
        )
        if not rerun_reproduced:
            raise RuntimeError("training-only graph selection/averaging was not reproducible")

        m1 = NaiveModel().fit(train)
        m2 = MissingAwareModel().fit(train)
        full_m3 = DependenceModel(m3_graph).fit(train)
        m23 = FixedComposition(m1, m2, full_m3)
        full_m4 = [JointModel(graph).fit(train) for graph in candidates]
        all_m4_predictions = np.asarray([model.predict_proba(test.y) for model in full_m4])
        predictions = {
            "M2_missing": m2.predict_proba(test.y),
            "Learned_M3": full_m3.predict_proba(test.y),
            "Learned_M23": m23.predict_proba(test.y),
            "Oracle_M4": all_m4_predictions[candidates.index(truth)],
            "Learned_M4": all_m4_predictions[m4_index],
            "Graph_Averaged_M4": np.sum(weights[:, None] * all_m4_predictions, axis=0),
        }
        expected_ids = tuple(test.episode_id.tolist())
        for method in METHODS:
            per_seed.append({
                "seed": seed, "method": method, "episode_count": len(expected_ids),
                "episode_id_sha256": hashlib.sha256(test.episode_id.tobytes()).hexdigest(),
                **metrics(test.z, predictions[method]),
            })

        for index, graph in enumerate(candidates):
            validation_rows.append({
                "seed": seed, "candidate_index": index, "graph_id": graph_id(graph),
                "distance_from_truth": edge_distance(graph, truth),
                "m3_validation_nll": float(m3_scores[index]),
                "m4_validation_nll": float(m4_scores[index]),
                "m4_weight": float(weights[index]),
            })
        for distance in (0.5, 0.75, 1.0):
            selected = [i for i, graph in enumerate(candidates) if edge_distance(graph, truth) == distance]
            values = [metrics(test.z, all_m4_predictions[index]) for index in selected]
            mismatch_rows.append({
                "seed": seed, "injected_mismatch_distance": distance,
                "candidate_count": len(selected),
                **{name: float(np.mean([value[name] for value in values])) for name in METRIC_NAMES},
            })
        true_index = candidates.index(truth)
        graph_rows.append({
            "seed": seed, "true_graph": graph_id(truth),
            "learned_m3_graph": graph_id(m3_graph),
            "learned_m4_graph": graph_id(m4_graph),
            "highest_weight_graph": graph_id(candidates[highest_index]),
            "m3_recovery_error": edge_distance(m3_graph, truth),
            "m4_recovery_error": edge_distance(m4_graph, truth),
            "m3_exact_recovery": int(m3_index == true_index),
            "m4_exact_recovery": int(m4_index == true_index),
            "true_graph_weight": float(weights[true_index]),
            "expected_graph_error": float(sum(
                weights[i] * edge_distance(graph, truth) for i, graph in enumerate(candidates)
            )),
            "training_rerun_reproduced": int(rerun_reproduced),
        })
        diagnostic_rows.append({"seed": seed, **diagnostics(test_generated)})
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        runtime_rows.append({
            "seed": seed, "wall_seconds": time.perf_counter() - started,
            "fitted_table_models": 2 * len(candidates) + 3 + len(candidates),
            "graph_candidates": len(candidates),
            "python_tracemalloc_peak_bytes": peak,
            "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        })

    summary = _summarize(per_seed, ("method",))
    mismatch_summary = _summarize(mismatch_rows, ("injected_mismatch_distance",))
    paired = _paired(per_seed)
    full_run = (
        not smoke and tuple(seeds) == SEEDS and train_episodes == 5000
        and validation_episodes == 1250 and test_episodes == 5000
    )
    all_training_reruns_reproduced = all(
        row["training_rerun_reproduced"] == 1 for row in graph_rows
    )
    decision = _decision(paired, full_run, all_training_reruns_reproduced)
    config = {
        "status": "frozen_exploratory_diagnostic" if full_run else "smoke",
        "seeds": list(seeds), "train_episodes": train_episodes,
        "structure_fit_episodes": fit_episodes, "validation_episodes": validation_episodes,
        "test_episodes": test_episodes, "methods": METHODS,
        "candidate_count": len(candidates), "copy_probability": 0.85,
        "composition": "logit(M2)+logit(M3)-logit(M1)",
        "graph_weight_rule": "exp[-N_validation*(NLL-NLL_min)]",
    }
    manifest = {
        "code_commit": commit, "checkout_dirty_at_start": dirty,
        "protocol_path": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": _sha256(PROTOCOL),
        "seeds": list(seeds), "candidate_graphs": [graph_id(graph) for graph in candidates],
        "train_episode_ids": [0, train_episodes - 1],
        "structure_fit_episode_ids": [0, fit_episodes - 1],
        "validation_episode_ids": [fit_episodes, train_episodes - 1],
        "test_episode_ids": [0, test_episodes - 1],
    }
    for filename, rows in (
        ("per_seed_metrics.csv", per_seed), ("summary.csv", summary),
        ("paired_differences.csv", paired), ("graph_diagnostics.csv", graph_rows),
        ("validation_scores.csv", validation_rows),
        ("fixed_mismatch_per_seed.csv", mismatch_rows),
        ("fixed_mismatch_summary.csv", mismatch_summary),
        ("generator_diagnostics.csv", diagnostic_rows), ("runtime.csv", runtime_rows),
    ):
        _write_csv(output / filename, rows)
    for filename, value in (("config.json", config), ("manifest.json", manifest), ("decision.json", decision)):
        (output / filename).write_text(json.dumps(value, indent=2), encoding="utf-8")
    _write_report(output / "report.md", summary, paired, graph_rows, mismatch_summary, runtime_rows, decision, smoke)
    return {
        "per_seed_rows": len(per_seed), "validation_rows": len(validation_rows),
        "mismatch_rows": len(mismatch_rows), "decision": decision["status"],
    }


def _write_report(path, summary, paired, graph_rows, mismatch_summary, runtime_rows, decision, smoke):
    lines = [
        "# JMID v3 smoke result" if smoke else "# JMID v3 frozen exploratory diagnostic",
        "", "Wiring evidence only." if smoke else "Ten-seed diagnostic evidence; not formal confirmation.", "",
        "## Predictive results", "",
        "| Method | Accuracy | Brier | NLL | ECE | Wrong confidence | High-confidence error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = next(r for r in summary if r["method"] == method)
        lines.append(
            f"| {method} | {row['accuracy_mean']:.4f} | {row['brier_mean']:.4f} | "
            f"{row['nll_mean']:.4f} | {row['ece_mean']:.4f} | "
            f"{row['wrong_confidence_mean']:.4f} | {row['high_confidence_error_rate_mean']:.4f} |"
        )
    lines += ["", "## Paired differences versus Learned M23", "",
              "| Method | Metric | Mean difference | 95% CI | Lower seeds |", "|---|---|---:|---:|---:|"]
    for row in paired:
        lines.append(
            f"| {row['method']} | {row['metric']} | {row['mean_difference']:.6f} | "
            f"[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] | {row['lower_seed_count']}/{row['seeds']} |"
        )
    lines += ["", "## Graph recovery", "",
              f"- Learned M3 exact recovery: {np.mean([r['m3_exact_recovery'] for r in graph_rows]):.3f}.",
              f"- Learned M4 exact recovery: {np.mean([r['m4_exact_recovery'] for r in graph_rows]):.3f}.",
              f"- Mean true-graph weight: {np.mean([r['true_graph_weight'] for r in graph_rows]):.6f}.",
              f"- Mean posterior expected graph error: {np.mean([r['expected_graph_error'] for r in graph_rows]):.4f}.",
              "", "## Fixed mismatch diagnostic", ""]
    for row in mismatch_summary:
        lines.append(
            f"- distance {row['injected_mismatch_distance']}: Brier {row['brier_mean']:.4f}; "
            f"NLL {row['nll_mean']:.4f}; accuracy {row['accuracy_mean']:.4f}."
        )
    lines += ["", "## Complexity", "",
              f"- Mean wall time per seed: {np.mean([r['wall_seconds'] for r in runtime_rows]):.3f} seconds.",
              f"- Fitted table models per seed: {runtime_rows[0]['fitted_table_models']}.",
              f"- Maximum recorded Python peak: {max(r['python_tracemalloc_peak_bytes'] for r in runtime_rows)} bytes.",
              "", "## Frozen decision", "", f"**{decision['status']}**", "", decision["reason"], "",
              "This exploratory outcome does not establish novelty or formal evidence."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(args.output, smoke=args.smoke)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
