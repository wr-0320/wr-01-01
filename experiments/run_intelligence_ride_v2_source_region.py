"""Development evaluation of hierarchical source-region reliability filtering."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES
from deception_rl.intelligence.ride import HonestIntelligenceReasoner
from deception_rl.intelligence.ride_v2.active import MultiSourceEntropyVerify
from deception_rl.intelligence.ride_v2.oracle import OracleReliabilityController
from deception_rl.intelligence.ride_v2.reliability import (
    DynamicReliabilityController,
    DynamicSourceReliabilityFilter,
)
from deception_rl.intelligence.ride_v2.source_region_reliability import (
    HierarchicalSourceRegionReliabilityFilter,
    SourceRegionReliabilityController,
)
from experiments.run_intelligence_ride import (
    TRAIN_FAMILIES,
    collect_prefix_records,
    fit_calibrator,
    train_model,
)
from experiments.run_intelligence_ride_v2_headroom import (
    COMMIT_STEP,
    DEVELOPMENT_RED_FAMILIES,
    FORMAL_OUTPUT,
    VERIFICATION_BUDGET,
    directory_manifest,
    evaluate_development_episode,
    summarize,
    write_csv,
)
from experiments.run_intelligence_ride_v2_reliability import (
    DEFAULT_OUTPUT as SOURCE_RELIABILITY_OUTPUT,
    HEADROOM_OUTPUT,
    RELIABILITY_TRAIN_FAMILIES,
    _inference,
    collect_reliability_samples,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REGION_V1_OUTPUT = (
    ROOT / "outputs" / "intelligence_ride_v2_source_region_development"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "intelligence_ride_v2_source_region_corrected_development"
)
SOURCE_REGION_SEEDS = (1111, 1212, 1313)
METHODS = (
    "multi_source",
    "ride_v1",
    "multi_source_entropy_verify",
    "source_reliability_entropy_verify",
    "source_region_reliability",
    "source_region_reliability_entropy_verify",
    "oracle_cell_reliability",
)


def build_controllers(
    model: Any,
    reasoner: Any,
    calibrator: Any,
    source_filter: DynamicSourceReliabilityFilter,
    source_region_filter: HierarchicalSourceRegionReliabilityFilter,
    *,
    max_cycles: int,
) -> dict[str, Any]:
    source_active = DynamicReliabilityController(
        _inference(
            model,
            reasoner,
            calibrator,
            "source_active_inference",
            full_ride=False,
            max_cycles=max_cycles,
        ),
        source_filter,
        active_verification=True,
        verification_budget=VERIFICATION_BUDGET,
        commit_step=COMMIT_STEP,
    )
    source_active.name = "source_reliability_entropy_verify"
    oracle = OracleReliabilityController(
        _inference(
            model,
            reasoner,
            calibrator,
            "oracle_cell_inference",
            full_ride=False,
            max_cycles=max_cycles,
        ),
        commit_step=COMMIT_STEP,
    )
    oracle.name = "oracle_cell_reliability"
    controllers = {
        "multi_source": _inference(
            model,
            reasoner,
            calibrator,
            "multi_source",
            full_ride=False,
            max_cycles=max_cycles,
        ),
        "ride_v1": _inference(
            model,
            reasoner,
            calibrator,
            "ride_v1",
            full_ride=True,
            max_cycles=max_cycles,
        ),
        "multi_source_entropy_verify": MultiSourceEntropyVerify(
            _inference(
                model,
                reasoner,
                calibrator,
                "entropy_inference",
                full_ride=False,
                max_cycles=max_cycles,
            ),
            verification_budget=VERIFICATION_BUDGET,
            commit_step=COMMIT_STEP,
        ),
        "source_reliability_entropy_verify": source_active,
        "source_region_reliability": SourceRegionReliabilityController(
            _inference(
                model,
                reasoner,
                calibrator,
                "source_region_inference",
                full_ride=False,
                max_cycles=max_cycles,
            ),
            source_region_filter,
            active_verification=False,
            commit_step=COMMIT_STEP,
        ),
        "source_region_reliability_entropy_verify": SourceRegionReliabilityController(
            _inference(
                model,
                reasoner,
                calibrator,
                "source_region_active_inference",
                full_ride=False,
                max_cycles=max_cycles,
            ),
            source_region_filter,
            active_verification=True,
            verification_budget=VERIFICATION_BUDGET,
            commit_step=COMMIT_STEP,
        ),
        "oracle_cell_reliability": oracle,
    }
    if tuple(controllers) != METHODS:
        raise AssertionError("source-region method order drifted")
    return controllers


def source_region_filter_metrics(
    source_region_filter: HierarchicalSourceRegionReliabilityFilter,
    samples: list[dict[str, Any]],
) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        grouped[str(row["episode_id"])].append(row)
    probabilities = []
    labels = []
    for episode_rows in grouped.values():
        ordered = sorted(episode_rows, key=lambda row: int(row["step"]))
        source_evidence = np.stack([np.asarray(row["features"]) for row in ordered])
        cell_evidence = np.stack([np.asarray(row["cell_features"]) for row in ordered])
        global_history = source_region_filter.global_filter.estimate_evidence(
            source_evidence
        )
        reliable = source_region_filter.estimate_evidence(
            cell_evidence, global_history
        )
        probabilities.extend((1.0 - reliable).reshape(-1).tolist())
        labels.extend(
            np.stack([np.asarray(row["compromised_cells"]) for row in ordered])
            .reshape(-1)
            .tolist()
        )
    probability = np.asarray(probabilities, dtype=float)
    label = np.asarray(labels, dtype=int)
    positive = probability[label == 1]
    negative = np.sort(probability[label == 0])
    lower = np.searchsorted(negative, positive, side="left")
    upper = np.searchsorted(negative, positive, side="right")
    auroc = float(np.mean((lower + upper) / 2) / len(negative))
    prediction = probability >= 0.5
    true_positive = int(((label == 1) & prediction).sum())
    false_positive = int(((label == 0) & prediction).sum())
    false_negative = int(((label == 1) & ~prediction).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "samples": len(label),
        "compromised_prevalence": float(label.mean()),
        "brier": float(np.mean((probability - label) ** 2)),
        "auroc": auroc,
        "f1": 2 * true_positive / denominator if denominator else 0.0,
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = summarize(rows)
    for method in METHODS:
        selected = [row for row in rows if row["controller"] == method]
        output.extend(
            summarize([{**row, "deception_family": "overall"} for row in selected])
        )
    return sorted(output, key=lambda row: (row["deception_family"], row["method"]))


def paired_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families = ("overall", *DEVELOPMENT_RED_FAMILIES)
    differences: dict[tuple[str, str], list[float]] = defaultdict(list)
    for seed in sorted({int(row["development_seed"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["development_seed"]) == seed]
        for family in families:
            selected = seed_rows if family == "overall" else [
                row for row in seed_rows if row["deception_family"] == family
            ]
            accuracy = {
                method: float(
                    np.mean(
                        [
                            row["correct_commitment"]
                            for row in selected
                            if row["controller"] == method
                        ]
                    )
                )
                for method in METHODS
            }
            for method in METHODS:
                if method != "multi_source":
                    differences[(method, family)].append(
                        accuracy[method] - accuracy["multi_source"]
                    )
    output = []
    for (method, family), values_list in sorted(differences.items()):
        values = np.asarray(values_list)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        half_width = 4.303 * std / math.sqrt(3) if len(values) == 3 else float("nan")
        output.append(
            {
                "method": method,
                "deception_family": family,
                "seeds": len(values),
                "accuracy_difference_mean": mean,
                "accuracy_difference_ci95_low": mean - half_width,
                "accuracy_difference_ci95_high": mean + half_width,
            }
        )
    return output


def write_report(
    destination: Path,
    summary: list[dict[str, Any]],
    differences: list[dict[str, Any]],
    filter_metrics: list[dict[str, Any]],
    total_episodes: int,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    lookup = {(row["method"], row["deception_family"]): row for row in summary}
    baseline = lookup[("multi_source", "overall")]["correct_commitment"]
    entropy = lookup[("multi_source_entropy_verify", "overall")]["correct_commitment"]
    learned = lookup[("source_region_reliability_entropy_verify", "overall")][
        "correct_commitment"
    ]
    oracle = lookup[("oracle_cell_reliability", "overall")]["correct_commitment"]
    oracle_headroom = oracle - baseline
    recovered = (learned - baseline) / oracle_headroom if oracle_headroom else float("nan")
    multi_baseline = lookup[("multi_source", "multi_source")]["correct_commitment"]
    multi_learned = lookup[
        ("source_region_reliability_entropy_verify", "multi_source")
    ]["correct_commitment"]
    multi_drop = multi_baseline - multi_learned
    adaptive_gain = (
        lookup[("source_region_reliability_entropy_verify", "adaptive")][
            "correct_commitment"
        ]
        - lookup[("multi_source", "adaptive")]["correct_commitment"]
    )
    mean_prevalence = float(
        np.mean([row["compromised_prevalence"] for row in filter_metrics])
    )
    mean_brier = float(np.mean([row["brier"] for row in filter_metrics]))
    no_skill_brier = mean_prevalence * (1.0 - mean_prevalence)
    brier_skill = 1.0 - mean_brier / no_skill_brier if no_skill_brier else float("nan")
    go = (
        recovered >= 0.5
        and learned > baseline
        and learned > entropy
        and multi_drop <= 0.02
        and adaptive_gain > 0
    )
    table_rows = []
    for method in METHODS:
        row = lookup[(method, "overall")]
        table_rows.append(
            f"| {method} | {100 * row['correct_commitment']:.1f}% | "
            f"{100 * (row['correct_commitment'] - baseline):+.1f} pp | "
            f"{row['information_cost']:.2f} | {row['regret']:.2f} |"
        )
    family_rows = []
    for family in ("multi_source", "switch_dev_step2", "adaptive"):
        family_rows.append(
            f"| {family} | {100 * lookup[('multi_source', family)]['correct_commitment']:.1f}% | "
            f"{100 * lookup[('source_reliability_entropy_verify', family)]['correct_commitment']:.1f}% | "
            f"{100 * lookup[('source_region_reliability', family)]['correct_commitment']:.1f}% | "
            f"{100 * lookup[('source_region_reliability_entropy_verify', family)]['correct_commitment']:.1f}% | "
            f"{100 * lookup[('oracle_cell_reliability', family)]['correct_commitment']:.1f}% |"
        )
    report = f"""# Hierarchical Source–Region Reliability Development

The filter estimates time-varying reliability for each source-region cell while
sharing emission and transition parameters at the source level. Runtime inference
uses only the original 49-dimensional blue observation.

This corrected run keeps every unreported source-region cell at zero during
reliability conditioning. The earlier source-level and source-region learned-filter
results attenuated unreported cells into nonzero values and are retained only as an
audit record; their learned-filter Go / No-Go conclusions are invalidated. Oracle and
baseline results were not affected by that implementation error.

## Protocol

- Development seeds: {', '.join(map(str, seeds))}
- Total evaluation episodes: {total_episodes}
- Filter-fit families: {', '.join(RELIABILITY_TRAIN_FAMILIES)}
- Multi-source, switch, and adaptive are held out from filter fitting.

## Overall results

| Method | Correct commitment | vs Multi-source | Information cost | Regret |
|---|---:|---:|---:|---:|
{chr(10).join(table_rows)}

Source-region reliability plus verification recovered **{100 * recovered:.1f}%**
of Oracle Cell Reliability's overall accuracy headroom.

## Priority families

| Family | Multi-source | Source-level + verify | Source-region | Source-region + verify | Oracle cell |
|---|---:|---:|---:|---:|---:|
{chr(10).join(family_rows)}

## Cell-state estimation

Mean held-out AUROC: {np.mean([row['auroc'] for row in filter_metrics]):.3f};
mean Brier: {mean_brier:.3f}; mean F1: {np.mean([row['f1'] for row in filter_metrics]):.3f}.
Compromised-cell prevalence is {100 * mean_prevalence:.1f}%, so the constant-rate
no-skill Brier is {no_skill_brier:.3f} and Brier skill is {brier_skill:.3f}.
The AUROC therefore reflects ranking ability, while the probabilities are not yet
calibrated well enough for reliability-weighted fusion.

## Frozen Go / No-Go rule

**{'GO' if go else 'NO-GO'}**. Required simultaneously: at least 50% Oracle
headroom recovery, overall accuracy above both Multi-source and Entropy Verify,
Adaptive improvement, and no more than 2 pp degradation on multi-source deception.

- Oracle headroom recovery: {100 * recovered:.1f}%
- Overall vs Multi-source: {100 * (learned - baseline):+.1f} pp
- Overall vs Entropy Verify: {100 * (learned - entropy):+.1f} pp
- Adaptive vs Multi-source: {100 * adaptive_gain:+.1f} pp
- Multi-source deception change: {100 * (multi_learned - multi_baseline):+.1f} pp

This is development evidence and does not update any frozen formal result.
"""
    (destination / "source_region_development_report.md").write_text(
        report, encoding="utf-8"
    )
    decision = {
        "decision": "GO" if go else "NO-GO",
        "oracle_headroom_recovered_fraction": recovered,
        "overall_vs_multi_source": learned - baseline,
        "overall_vs_entropy_verify": learned - entropy,
        "adaptive_gain": adaptive_gain,
        "multi_source_deception_change": multi_learned - multi_baseline,
        "thresholds": {
            "recovered_fraction": 0.5,
            "multi_source_deception_max_drop": 0.02,
        },
    }
    (destination / "go_no_go.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    return decision


def run(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    seeds: tuple[int, ...] = SOURCE_REGION_SEEDS,
    episodes_per_family_seed: int = 30,
    reliability_train_episodes_per_hypothesis: int = 8,
    reliability_validation_episodes_per_hypothesis: int = 3,
    epochs: int = 8,
    train_episodes_per_hypothesis: int = 8,
    validation_episodes_per_hypothesis: int = 3,
    hidden_size: int = 32,
    batch_size: int = 64,
    max_cycles: int = 10,
) -> dict[str, Any]:
    used_seeds = {
        0,
        1,
        2,
        3,
        4,
        101,
        202,
        303,
        404,
        505,
        606,
        707,
        808,
        909,
    }
    if not seeds or set(seeds) & used_seeds:
        raise ValueError("source-region seeds must be new and disjoint")
    destination = Path(output_dir)
    frozen_directories = {
        "formal": FORMAL_OUTPUT,
        "headroom": HEADROOM_OUTPUT,
        "source_reliability": SOURCE_RELIABILITY_OUTPUT,
        "source_region_v1": SOURCE_REGION_V1_OUTPUT,
    }
    if destination.resolve() in {path.resolve() for path in frozen_directories.values()}:
        raise ValueError("source-region development requires an independent output")
    destination.mkdir(parents=True, exist_ok=True)
    frozen_before = {
        name: directory_manifest(path) for name, path in frozen_directories.items()
    }
    all_rows = []
    filter_metric_rows = []
    audit_counts = defaultdict(int)
    for development_seed in seeds:
        torch.manual_seed(development_seed)
        np.random.seed(development_seed)
        random.seed(development_seed)
        train_records = collect_prefix_records(
            TRAIN_FAMILIES,
            train_episodes_per_hypothesis,
            seed=160_000_000 + development_seed * 10_003,
            split=f"source_region_model_train_{development_seed}",
            max_cycles=max_cycles,
        )
        validation_records = collect_prefix_records(
            TRAIN_FAMILIES,
            validation_episodes_per_hypothesis,
            seed=170_000_000 + development_seed * 10_003,
            split=f"source_region_model_validation_{development_seed}",
            max_cycles=max_cycles,
        )
        model, _ = train_model(
            train_records,
            validation_records,
            epochs=epochs,
            hidden_size=hidden_size,
            batch_size=batch_size,
            seed=development_seed,
        )
        calibrator, _ = fit_calibrator(model, validation_records)
        reasoner = HonestIntelligenceReasoner(train_records)
        reliability_train = collect_reliability_samples(
            RELIABILITY_TRAIN_FAMILIES,
            reliability_train_episodes_per_hypothesis,
            seed=180_000_000 + development_seed * 10_003,
            reasoner=reasoner,
            split=f"source_region_filter_train_{development_seed}",
            max_cycles=max_cycles,
        )
        source_filter = DynamicSourceReliabilityFilter.fit(reliability_train)
        source_region_filter = HierarchicalSourceRegionReliabilityFilter.fit(
            reliability_train
        )
        reliability_validation = collect_reliability_samples(
            tuple(DEVELOPMENT_RED_FAMILIES),
            reliability_validation_episodes_per_hypothesis,
            seed=190_000_000 + development_seed * 10_003,
            reasoner=reasoner,
            split=f"source_region_filter_validation_{development_seed}",
            max_cycles=max_cycles,
        )
        filter_metric_rows.append(
            {
                "development_seed": development_seed,
                **source_region_filter_metrics(
                    source_region_filter, reliability_validation
                ),
            }
        )
        (destination / f"source_region_filter_seed{development_seed}.json").write_text(
            json.dumps(source_region_filter.to_dict(), indent=2), encoding="utf-8"
        )
        controllers = build_controllers(
            model,
            reasoner,
            calibrator,
            source_filter,
            source_region_filter,
            max_cycles=max_cycles,
        )
        for family_index, family in enumerate(DEVELOPMENT_RED_FAMILIES):
            for episode in range(episodes_per_family_seed):
                hypothesis = (episode + development_seed + family_index) % NUM_HYPOTHESES
                scenario_seed = 201_000_000 + development_seed * 100_003 + family_index * 1_009 + episode
                for method, controller in controllers.items():
                    row, episode_audit = evaluate_development_episode(
                        controller,
                        family,
                        hypothesis,
                        seed=scenario_seed,
                        max_cycles=max_cycles,
                    )
                    row["development_seed"] = development_seed
                    if row["controller"] != method:
                        raise AssertionError("controller name drifted")
                    all_rows.append(row)
                    for key, value in episode_audit.items():
                        audit_counts[key] += value
    expected = len(seeds) * len(DEVELOPMENT_RED_FAMILIES) * episodes_per_family_seed * len(METHODS)
    if len(all_rows) != expected:
        raise AssertionError(f"expected {expected} episodes, observed {len(all_rows)}")
    summary = aggregate(all_rows)
    differences = paired_differences(all_rows)
    write_csv(destination / "episodes.csv", all_rows)
    write_csv(destination / "summary.csv", summary)
    write_csv(destination / "paired_differences.csv", differences)
    write_csv(destination / "source_region_filter_metrics.csv", filter_metric_rows)
    frozen_after = {
        name: directory_manifest(path) for name, path in frozen_directories.items()
    }
    audit = {
        "ordinary_policy_calls": audit_counts["ordinary_calls"],
        "privileged_oracle_calls": audit_counts["privileged_calls"],
        "blue_observations_audited": audit_counts["boundary_checks"],
        "runtime_input_feature_size": 49,
        "simulator_labels_used_at_runtime": False,
        "simulator_cell_labels_used_for_offline_fit": True,
        "oracle_fields_inserted_into_blue_observation": False,
        **{
            f"{name}_results_unchanged": frozen_before[name] == frozen_after[name]
            for name in frozen_directories
        },
    }
    if not all(value for key, value in audit.items() if key.endswith("_results_unchanged")):
        raise AssertionError("a frozen output directory changed")
    (destination / "information_boundary_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    config = {
        "protocol": "RIDe-v2 Hierarchical Source-Region Reliability Corrected v2",
        "development_seeds": list(seeds),
        "episodes_per_family_seed": episodes_per_family_seed,
        "total_evaluation_episodes": len(all_rows),
        "methods": list(METHODS),
        "filter_training_families": list(RELIABILITY_TRAIN_FAMILIES),
        "formal_results_replaced": False,
        "prior_development_results_replaced": False,
        "large_scale_training": False,
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    decision = write_report(
        destination, summary, differences, filter_metric_rows, len(all_rows), seeds
    )
    return {**config, **decision}


def read_numeric_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            key: value if key in {"method", "deception_family"} else float(value)
            for key, value in row.items()
        }
        for row in rows
    ]


def analyze_existing(destination: Path) -> dict[str, Any]:
    summary = read_numeric_csv(destination / "summary.csv")
    differences = read_numeric_csv(destination / "paired_differences.csv")
    metrics = read_numeric_csv(destination / "source_region_filter_metrics.csv")
    config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    return write_report(
        destination,
        summary,
        differences,
        metrics,
        int(config["total_evaluation_episodes"]),
        tuple(config["development_seeds"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes-per-family-seed", type=int, default=30)
    parser.add_argument("--analyze-existing", action="store_true")
    args = parser.parse_args()
    if args.analyze_existing:
        print(json.dumps(analyze_existing(args.output_dir), indent=2))
    else:
        print(
            json.dumps(
                run(
                    args.output_dir,
                    episodes_per_family_seed=args.episodes_per_family_seed,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
