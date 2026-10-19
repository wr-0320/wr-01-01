"""Evaluate calibrated cell reliability before reconnecting it to belief fusion."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deception_rl.intelligence.ride import HonestIntelligenceReasoner
from deception_rl.intelligence.ride_v2.calibrated_reliability import (
    IsotonicCalibrator,
    LogisticCellModel,
    PlattCalibrator,
    build_cell_dataset,
    calibration_bins,
    probability_metrics,
)
from experiments.run_intelligence_ride import TRAIN_FAMILIES, collect_prefix_records
from experiments.run_intelligence_ride_v2_headroom import (
    DEVELOPMENT_RED_FAMILIES,
    FORMAL_OUTPUT,
    directory_manifest,
    write_csv,
)
from experiments.run_intelligence_ride_v2_reliability import (
    DEFAULT_OUTPUT as SOURCE_RELIABILITY_OUTPUT,
    HEADROOM_OUTPUT,
    RELIABILITY_TRAIN_FAMILIES,
    collect_reliability_samples,
)
from experiments.run_intelligence_ride_v2_source_region import (
    DEFAULT_OUTPUT as CORRECTED_SOURCE_REGION_OUTPUT,
    SOURCE_REGION_V1_OUTPUT,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "intelligence_calibrated_cell_reliability_development"
CALIBRATED_SEEDS = (1414, 1515, 1616)
PRIMARY_MODEL = "balanced_logistic_platt"
MODELS = (
    "unweighted_logistic",
    "balanced_logistic_raw",
    PRIMARY_MODEL,
    "balanced_logistic_isotonic",
)


def _predictions(
    unweighted: LogisticCellModel,
    balanced: LogisticCellModel,
    platt: PlattCalibrator,
    isotonic: IsotonicCalibrator,
    matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    balanced_score = balanced.decision_function(matrix)
    return {
        "unweighted_logistic": unweighted.predict_proba(matrix),
        "balanced_logistic_raw": balanced.predict_proba(matrix),
        PRIMARY_MODEL: platt.predict(balanced_score),
        "balanced_logistic_isotonic": isotonic.predict(balanced_score),
    }


def _fit_models(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    calibration_matrix: np.ndarray,
    calibration_labels: np.ndarray,
) -> tuple[LogisticCellModel, LogisticCellModel, PlattCalibrator, IsotonicCalibrator]:
    unweighted = LogisticCellModel.fit(
        train_matrix, train_labels, balanced=False, l2=1.0
    )
    balanced = LogisticCellModel.fit(
        train_matrix, train_labels, balanced=True, l2=1.0
    )
    calibration_score = balanced.decision_function(calibration_matrix)
    return (
        unweighted,
        balanced,
        PlattCalibrator.fit(calibration_score, calibration_labels),
        IsotonicCalibrator.fit(calibration_score, calibration_labels),
    )


def _model_payload(
    unweighted: LogisticCellModel,
    balanced: LogisticCellModel,
    platt: PlattCalibrator,
    isotonic: IsotonicCalibrator,
) -> dict[str, Any]:
    return {
        "unweighted_logistic": unweighted.to_dict(),
        "balanced_logistic": balanced.to_dict(),
        "platt": {"intercept": platt.intercept, "slope": platt.slope},
        "isotonic": {
            "upper_scores": isotonic.upper_scores.tolist(),
            "probabilities": isotonic.probabilities.tolist(),
        },
    }


def _aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    numeric = (
        "samples",
        "compromised_prevalence",
        "brier",
        "no_skill_brier",
        "brier_skill",
        "ece",
        "auroc",
        "average_precision",
        "minority_recall_at_0_5",
        "f1_at_0_5",
        "calibration_intercept",
        "calibration_slope",
    )
    for model in MODELS:
        for family in ("overall", *DEVELOPMENT_RED_FAMILIES):
            selected = [
                row
                for row in rows
                if row["model"] == model and row["deception_family"] == family
            ]
            output.append(
                {
                    "model": model,
                    "deception_family": family,
                    "seeds": len(selected),
                    **{
                        name: float(np.mean([float(row[name]) for row in selected]))
                        for name in numeric
                    },
                }
            )
    return output


def write_report(
    destination: Path,
    summary: list[dict[str, Any]],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    lookup = {(row["model"], row["deception_family"]): row for row in summary}
    primary = lookup[(PRIMARY_MODEL, "overall")]
    skill = float(primary["brier_skill"])
    if skill >= 0.1:
        decision = "GO_FOR_FUSION"
    elif skill > 0.0:
        decision = "PROMISING_BUT_BELOW_TARGET"
    else:
        decision = "STOP_RELIABILITY_LINE"
    table = []
    for model in MODELS:
        row = lookup[(model, "overall")]
        table.append(
            f"| {model} | {row['brier']:.4f} | {row['brier_skill']:.3f} | "
            f"{row['ece']:.3f} | {row['auroc']:.3f} | {row['average_precision']:.3f} | "
            f"{row['minority_recall_at_0_5']:.3f} | {row['calibration_intercept']:.3f} | "
            f"{row['calibration_slope']:.3f} |"
        )
    family_table = []
    for family in DEVELOPMENT_RED_FAMILIES:
        row = lookup[(PRIMARY_MODEL, family)]
        family_table.append(
            f"| {family} | {row['compromised_prevalence']:.3f} | {row['brier_skill']:.3f} | "
            f"{row['ece']:.3f} | {row['auroc']:.3f} | {row['average_precision']:.3f} |"
        )
    report = f"""# Calibrated Cell Reliability Development

This experiment evaluates cell-compromise probabilities in isolation. It does not
modify belief fusion, verification, commitment, or any frozen formal result. Training,
calibration, and evaluation episodes are disjoint. Only reported cells are scored.

The primary model was fixed before evaluation as class-balanced logistic regression
followed by Platt calibration. Unweighted, raw balanced, and isotonic variants are
diagnostic comparisons.

## Protocol

- Development seeds: {', '.join(map(str, seeds))}
- Fit families: {', '.join(RELIABILITY_TRAIN_FAMILIES)}
- Held-out families include multi-source, switch, and adaptive deception.
- Input: five visible evidence terms, five recent-change terms, source ID, region ID.
- Hard continuation gate: Brier skill > 0; fusion target: Brier skill >= 0.1.

## Overall held-out probability quality

| Model | Brier | Brier skill | ECE | AUROC | PR-AUC | Recall@0.5 | Cal intercept | Cal slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

## Primary model by deception family

| Family | Prevalence | Brier skill | ECE | AUROC | PR-AUC |
|---|---:|---:|---:|---:|---:|
{chr(10).join(family_table)}

## Decision

**{decision}**. The primary model's mean overall Brier skill is {skill:.3f}.
Only `GO_FOR_FUSION` authorizes a later, separately evaluated fusion experiment.
This run itself never reconnects the predictor to RIDe decisions.
"""
    (destination / "calibrated_reliability_report.md").write_text(report, encoding="utf-8")
    result = {
        "decision": decision,
        "primary_model": PRIMARY_MODEL,
        "mean_overall_brier_skill": skill,
        "hard_continuation_threshold": 0.0,
        "fusion_target": 0.1,
    }
    (destination / "go_no_go.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    seeds: tuple[int, ...] = CALIBRATED_SEEDS,
    train_episodes_per_hypothesis: int = 10,
    calibration_episodes_per_hypothesis: int = 4,
    evaluation_episodes_per_hypothesis: int = 4,
    max_cycles: int = 10,
) -> dict[str, Any]:
    used_seeds = {0, 1, 2, 3, 4, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1111, 1212, 1313}
    if not seeds or set(seeds) & used_seeds:
        raise ValueError("calibrated reliability seeds must be new and disjoint")
    destination = Path(output_dir)
    frozen_directories = {
        "formal": FORMAL_OUTPUT,
        "headroom": HEADROOM_OUTPUT,
        "source_reliability_v1": SOURCE_RELIABILITY_OUTPUT,
        "source_region_v1": SOURCE_REGION_V1_OUTPUT,
        "source_region_corrected": CORRECTED_SOURCE_REGION_OUTPUT,
    }
    if destination.resolve() in {path.resolve() for path in frozen_directories.values()}:
        raise ValueError("calibrated reliability requires an independent output")
    destination.mkdir(parents=True, exist_ok=True)
    frozen_before = {name: directory_manifest(path) for name, path in frozen_directories.items()}
    metric_rows: list[dict[str, Any]] = []
    diagram_rows: list[dict[str, Any]] = []
    for development_seed in seeds:
        random.seed(development_seed)
        np.random.seed(development_seed)
        torch.manual_seed(development_seed)
        reasoner_records = collect_prefix_records(
            TRAIN_FAMILIES,
            8,
            seed=210_000_000 + development_seed * 10_003,
            split=f"calibrated_reasoner_{development_seed}",
            max_cycles=max_cycles,
        )
        reasoner = HonestIntelligenceReasoner(reasoner_records)
        train_samples = collect_reliability_samples(
            RELIABILITY_TRAIN_FAMILIES,
            train_episodes_per_hypothesis,
            seed=220_000_000 + development_seed * 10_003,
            reasoner=reasoner,
            split=f"calibrated_train_{development_seed}",
            max_cycles=max_cycles,
        )
        calibration_samples = collect_reliability_samples(
            RELIABILITY_TRAIN_FAMILIES,
            calibration_episodes_per_hypothesis,
            seed=230_000_000 + development_seed * 10_003,
            reasoner=reasoner,
            split=f"calibrated_calibration_{development_seed}",
            max_cycles=max_cycles,
        )
        evaluation_samples = collect_reliability_samples(
            tuple(DEVELOPMENT_RED_FAMILIES),
            evaluation_episodes_per_hypothesis,
            seed=240_000_000 + development_seed * 10_003,
            reasoner=reasoner,
            split=f"calibrated_evaluation_{development_seed}",
            max_cycles=max_cycles,
        )
        train_matrix, train_labels = build_cell_dataset(train_samples)
        calibration_matrix, calibration_labels = build_cell_dataset(calibration_samples)
        models = _fit_models(
            train_matrix, train_labels, calibration_matrix, calibration_labels
        )
        (destination / f"calibrated_model_seed{development_seed}.json").write_text(
            json.dumps(_model_payload(*models), indent=2), encoding="utf-8"
        )
        for family in ("overall", *DEVELOPMENT_RED_FAMILIES):
            selected = evaluation_samples if family == "overall" else [
                row for row in evaluation_samples if row["deception_family"] == family
            ]
            matrix, labels = build_cell_dataset(selected)
            for model_name, probability in _predictions(*models, matrix).items():
                metric_rows.append(
                    {
                        "development_seed": development_seed,
                        "model": model_name,
                        "deception_family": family,
                        **probability_metrics(probability, labels),
                    }
                )
                for bin_row in calibration_bins(probability, labels):
                    diagram_rows.append(
                        {
                            "development_seed": development_seed,
                            "model": model_name,
                            "deception_family": family,
                            **bin_row,
                        }
                    )
    summary = _aggregate_metrics(metric_rows)
    write_csv(destination / "metrics_by_seed.csv", metric_rows)
    write_csv(destination / "summary.csv", summary)
    write_csv(destination / "reliability_diagram.csv", diagram_rows)
    frozen_after = {name: directory_manifest(path) for name, path in frozen_directories.items()}
    audit = {
        "runtime_predictor_inputs_are_blue_visible": True,
        "reported_cells_only": True,
        "simulator_cell_labels_used_only_for_offline_fit_and_evaluation": True,
        "predictor_connected_to_belief_fusion": False,
        "predictor_connected_to_verify": False,
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
        "protocol": "Calibrated Cell Reliability Isolated Development v1",
        "development_seeds": list(seeds),
        "primary_model": PRIMARY_MODEL,
        "models": list(MODELS),
        "fit_families": list(RELIABILITY_TRAIN_FAMILIES),
        "evaluation_families": list(DEVELOPMENT_RED_FAMILIES),
        "train_episodes_per_hypothesis": train_episodes_per_hypothesis,
        "calibration_episodes_per_hypothesis": calibration_episodes_per_hypothesis,
        "evaluation_episodes_per_hypothesis": evaluation_episodes_per_hypothesis,
        "belief_fusion_evaluated": False,
        "formal_results_replaced": False,
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return {**config, **write_report(destination, summary, seeds)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
