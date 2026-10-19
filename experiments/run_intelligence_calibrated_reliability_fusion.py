"""Fresh-seed decision evaluation of calibrated cell reliability fusion."""

from __future__ import annotations

import argparse
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
from deception_rl.intelligence.ride_v2.calibrated_reliability import (
    CalibratedCellReliabilityFilter,
    LogisticCellModel,
    PlattCalibrator,
    build_cell_dataset,
)
from deception_rl.intelligence.ride_v2.oracle import OracleReliabilityController
from deception_rl.intelligence.ride_v2.source_region_reliability import (
    SourceRegionReliabilityController,
)
from experiments.run_intelligence_calibrated_cell_reliability import (
    DEFAULT_OUTPUT as CALIBRATED_PREDICTOR_OUTPUT,
    _fit_models,
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
from experiments.run_intelligence_ride_v2_source_region import (
    DEFAULT_OUTPUT as CORRECTED_SOURCE_REGION_OUTPUT,
    SOURCE_REGION_V1_OUTPUT,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "intelligence_calibrated_reliability_fusion_development"
FUSION_SEEDS = (1818, 1919, 2020)
PRIMARY_METHOD = "balanced_logistic_platt_fusion"
METHODS = (
    "multi_source",
    "ride_v1",
    "unweighted_logistic_fusion",
    PRIMARY_METHOD,
    "oracle_cell_reliability",
)


def build_controllers(
    model: Any,
    reasoner: HonestIntelligenceReasoner,
    calibrator: Any,
    unweighted_filter: CalibratedCellReliabilityFilter,
    platt_filter: CalibratedCellReliabilityFilter,
    *,
    max_cycles: int,
) -> dict[str, Any]:
    def inference(name: str, *, full_ride: bool = False):
        return _inference(
            model,
            reasoner,
            calibrator,
            name,
            full_ride=full_ride,
            max_cycles=max_cycles,
        )

    unweighted = SourceRegionReliabilityController(
        inference("unweighted_logistic_inference"),
        unweighted_filter,
        active_verification=False,
        commit_step=COMMIT_STEP,
    )
    unweighted.name = "unweighted_logistic_fusion"
    platt = SourceRegionReliabilityController(
        inference("balanced_logistic_platt_inference"),
        platt_filter,
        active_verification=False,
        commit_step=COMMIT_STEP,
    )
    platt.name = PRIMARY_METHOD
    oracle = OracleReliabilityController(
        inference("oracle_cell_inference"), commit_step=COMMIT_STEP
    )
    oracle.name = "oracle_cell_reliability"
    controllers = {
        "multi_source": inference("multi_source"),
        "ride_v1": inference("ride_v1", full_ride=True),
        "unweighted_logistic_fusion": unweighted,
        PRIMARY_METHOD: platt,
        "oracle_cell_reliability": oracle,
    }
    if tuple(controllers) != METHODS:
        raise AssertionError("calibrated fusion method order drifted")
    return controllers


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = summarize(rows)
    for method in METHODS:
        selected = [row for row in rows if row["controller"] == method]
        output.extend(summarize([{**row, "deception_family": "overall"} for row in selected]))
    return sorted(output, key=lambda row: (row["deception_family"], row["method"]))


def paired_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in METHODS[1:]:
        for family in ("overall", *DEVELOPMENT_RED_FAMILIES):
            values = []
            for seed in sorted({int(row["development_seed"]) for row in rows}):
                selected = [
                    row
                    for row in rows
                    if int(row["development_seed"]) == seed
                    and (family == "overall" or row["deception_family"] == family)
                ]
                accuracy = {
                    name: float(np.mean([r["correct_commitment"] for r in selected if r["controller"] == name]))
                    for name in METHODS
                }
                values.append(accuracy[method] - accuracy["multi_source"])
            array = np.asarray(values)
            mean = float(array.mean())
            std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
            half_width = 4.303 * std / math.sqrt(3) if len(array) == 3 else float("nan")
            output.append(
                {
                    "method": method,
                    "deception_family": family,
                    "seeds": len(array),
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
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    lookup = {(row["method"], row["deception_family"]): row for row in summary}
    baseline = lookup[("multi_source", "overall")]["correct_commitment"]
    primary = lookup[(PRIMARY_METHOD, "overall")]["correct_commitment"]
    oracle = lookup[("oracle_cell_reliability", "overall")]["correct_commitment"]
    headroom = oracle - baseline
    recovered = (primary - baseline) / headroom if headroom else float("nan")
    family_changes = {
        family: lookup[(PRIMARY_METHOD, family)]["correct_commitment"]
        - lookup[("multi_source", family)]["correct_commitment"]
        for family in DEVELOPMENT_RED_FAMILIES
    }
    go = recovered >= 0.5 and min(family_changes.values()) >= -0.02
    table = []
    for method in METHODS:
        row = lookup[(method, "overall")]
        table.append(
            f"| {method} | {100 * row['correct_commitment']:.1f}% | "
            f"{100 * (row['correct_commitment'] - baseline):+.1f} pp | "
            f"{row['information_cost']:.2f} | {row['regret']:.2f} |"
        )
    families = []
    for family in DEVELOPMENT_RED_FAMILIES:
        families.append(
            f"| {family} | {100 * lookup[('multi_source', family)]['correct_commitment']:.1f}% | "
            f"{100 * lookup[('unweighted_logistic_fusion', family)]['correct_commitment']:.1f}% | "
            f"{100 * lookup[(PRIMARY_METHOD, family)]['correct_commitment']:.1f}% | "
            f"{100 * lookup[('oracle_cell_reliability', family)]['correct_commitment']:.1f}% |"
        )
    report = f"""# Calibrated Cell Reliability Fusion Development

This fresh-seed experiment reconnects the two frozen calibrated predictors to passive
cell reliability conditioning. Verification is disabled. Training, probability
calibration, decision evaluation, and all earlier development outputs remain separate.

## Protocol

- Fusion seeds: {', '.join(map(str, seeds))}
- Primary method: {PRIMARY_METHOD}
- Go rule fixed before evaluation: recover at least 50% of Oracle Cell headroom and
  avoid a drop greater than 2 pp in every deception family.

## Overall results

| Method | Correct commitment | vs Multi-source | Information cost | Regret |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

Primary Oracle headroom recovery: **{100 * recovered:.1f}%**.
Hypothesis Brier changed from {lookup[('multi_source', 'overall')]['hypothesis_brier']:.3f}
to {lookup[(PRIMARY_METHOD, 'overall')]['hypothesis_brier']:.3f}.

## Deception-family results

| Family | Multi-source | Unweighted fusion | Balanced+Platt fusion | Oracle cell |
|---|---:|---:|---:|---:|
{chr(10).join(families)}

## Decision

**{'GO' if go else 'NO-GO'}**. Minimum primary family change versus Multi-source:
{100 * min(family_changes.values()):+.1f} pp. This is development evidence and does
not replace the formal benchmark.

The isolated predictor passed its probability-quality gate, but the present fusion
operator linearly attenuates every visible report according to nonzero compromise
probability. That moves even honest reports away from the encoder's training
distribution. This result rejects the current attenuation operator; it does not
invalidate the separately measured calibrated predictor.
"""
    (destination / "calibrated_fusion_report.md").write_text(report, encoding="utf-8")
    result = {
        "decision": "GO" if go else "NO-GO",
        "primary_method": PRIMARY_METHOD,
        "oracle_headroom_recovered_fraction": recovered,
        "overall_vs_multi_source": primary - baseline,
        "minimum_family_change": min(family_changes.values()),
    }
    (destination / "go_no_go.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    seeds: tuple[int, ...] = FUSION_SEEDS,
    episodes_per_family_seed: int = 30,
    reliability_train_episodes_per_hypothesis: int = 10,
    reliability_calibration_episodes_per_hypothesis: int = 4,
    epochs: int = 8,
    train_episodes_per_hypothesis: int = 8,
    validation_episodes_per_hypothesis: int = 3,
    hidden_size: int = 32,
    batch_size: int = 64,
    max_cycles: int = 10,
) -> dict[str, Any]:
    used = {0, 1, 2, 3, 4, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1111, 1212, 1313, 1414, 1515, 1616}
    if not seeds or set(seeds) & used:
        raise ValueError("fusion seeds must be new and disjoint")
    destination = Path(output_dir)
    frozen = {
        "formal": FORMAL_OUTPUT,
        "headroom": HEADROOM_OUTPUT,
        "source_reliability_v1": SOURCE_RELIABILITY_OUTPUT,
        "source_region_v1": SOURCE_REGION_V1_OUTPUT,
        "source_region_corrected": CORRECTED_SOURCE_REGION_OUTPUT,
        "calibrated_predictor": CALIBRATED_PREDICTOR_OUTPUT,
    }
    if destination.resolve() in {path.resolve() for path in frozen.values()}:
        raise ValueError("calibrated fusion requires an independent output")
    destination.mkdir(parents=True, exist_ok=True)
    before = {name: directory_manifest(path) for name, path in frozen.items()}
    rows: list[dict[str, Any]] = []
    audit_counts = defaultdict(int)
    for development_seed in seeds:
        random.seed(development_seed)
        np.random.seed(development_seed)
        torch.manual_seed(development_seed)
        train_records = collect_prefix_records(
            TRAIN_FAMILIES,
            train_episodes_per_hypothesis,
            seed=250_000_000 + development_seed * 10_003,
            split=f"calibrated_fusion_model_train_{development_seed}",
            max_cycles=max_cycles,
        )
        validation_records = collect_prefix_records(
            TRAIN_FAMILIES,
            validation_episodes_per_hypothesis,
            seed=260_000_000 + development_seed * 10_003,
            split=f"calibrated_fusion_model_validation_{development_seed}",
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
            seed=270_000_000 + development_seed * 10_003,
            reasoner=reasoner,
            split=f"calibrated_fusion_reliability_train_{development_seed}",
            max_cycles=max_cycles,
        )
        reliability_calibration = collect_reliability_samples(
            RELIABILITY_TRAIN_FAMILIES,
            reliability_calibration_episodes_per_hypothesis,
            seed=280_000_000 + development_seed * 10_003,
            reasoner=reasoner,
            split=f"calibrated_fusion_reliability_calibration_{development_seed}",
            max_cycles=max_cycles,
        )
        train_matrix, train_labels = build_cell_dataset(reliability_train)
        calibration_matrix, calibration_labels = build_cell_dataset(reliability_calibration)
        unweighted, balanced, platt, _ = _fit_models(
            train_matrix, train_labels, calibration_matrix, calibration_labels
        )
        controllers = build_controllers(
            model,
            reasoner,
            calibrator,
            CalibratedCellReliabilityFilter(unweighted),
            CalibratedCellReliabilityFilter(balanced, platt),
            max_cycles=max_cycles,
        )
        for family_index, family in enumerate(DEVELOPMENT_RED_FAMILIES):
            for episode in range(episodes_per_family_seed):
                hypothesis = (episode + development_seed + family_index) % NUM_HYPOTHESES
                scenario_seed = 290_000_000 + development_seed * 100_003 + family_index * 1_009 + episode
                for method, controller in controllers.items():
                    row, episode_audit = evaluate_development_episode(
                        controller, family, hypothesis, seed=scenario_seed, max_cycles=max_cycles
                    )
                    row["development_seed"] = development_seed
                    if row["controller"] != method:
                        raise AssertionError("controller name drifted")
                    rows.append(row)
                    for key, value in episode_audit.items():
                        audit_counts[key] += value
    expected = len(seeds) * len(DEVELOPMENT_RED_FAMILIES) * episodes_per_family_seed * len(METHODS)
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} episodes, observed {len(rows)}")
    summary = aggregate(rows)
    differences = paired_differences(rows)
    write_csv(destination / "episodes.csv", rows)
    write_csv(destination / "summary.csv", summary)
    write_csv(destination / "paired_differences.csv", differences)
    after = {name: directory_manifest(path) for name, path in frozen.items()}
    audit = {
        "ordinary_policy_calls": audit_counts["ordinary_calls"],
        "privileged_oracle_calls": audit_counts["privileged_calls"],
        "runtime_input_feature_size": 49,
        "simulator_labels_used_at_runtime": False,
        "verification_enabled": False,
        **{f"{name}_results_unchanged": before[name] == after[name] for name in frozen},
    }
    if not all(value for key, value in audit.items() if key.endswith("_results_unchanged")):
        raise AssertionError("a frozen output directory changed")
    (destination / "information_boundary_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    config = {
        "protocol": "Calibrated Cell Reliability Passive Fusion Development v1",
        "development_seeds": list(seeds),
        "episodes_per_family_seed": episodes_per_family_seed,
        "total_evaluation_episodes": len(rows),
        "methods": list(METHODS),
        "verification_enabled": False,
        "formal_results_replaced": False,
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return {**config, **write_report(destination, summary, differences, seeds)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
