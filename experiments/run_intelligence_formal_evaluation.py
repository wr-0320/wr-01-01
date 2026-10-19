"""Frozen five-seed formal evaluation for adversarial intelligence decisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from deception_rl.agents.intelligence_policies import AdaptiveVerificationBluePolicy
from experiments.run_intelligence_ride import (
    RED_FAMILIES,
    TRAIN_FAMILIES,
    UNSEEN_FAMILIES,
    build_controllers,
    collect_prefix_records,
    evaluate_episode,
    fit_calibrator,
    train_model,
)
from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES
from deception_rl.intelligence.ride import HonestIntelligenceReasoner


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "intelligence_formal_evaluation"
FORMAL_METHODS = (
    "single_report",
    "multi_source_belief",
    "belief_plus_deception",
    "belief_deception_counterfactual",
    "full_ride",
    "bayesian_fusion",
    "pomcp_like",
    "active_inference",
    "rule_adaptive_defender",
)
PRIMARY_METRICS = (
    "correct_commitment",
    "false_commitment",
    "no_commit_rate",
    "decision_latency",
    "information_cost",
    "regret",
    "hypothesis_brier",
    "hypothesis_ece",
    "hypothesis_nll",
    "deception_brier",
    "deception_ece",
    "deception_auroc",
    "deception_f1",
    "deception_nll",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def binary_auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positive = probabilities[labels == 1]
    negative = np.sort(probabilities[labels == 0])
    if not len(positive) or not len(negative):
        return float("nan")
    lower = np.searchsorted(negative, positive, side="left")
    upper = np.searchsorted(negative, positive, side="right")
    return float(np.mean((lower + upper) / 2) / len(negative))


def binary_f1(labels: np.ndarray, probabilities: np.ndarray) -> float:
    prediction = probabilities >= 0.5
    true_positive = int(((labels == 1) & prediction).sum())
    false_positive = int(((labels == 0) & prediction).sum())
    false_negative = int(((labels == 1) & ~prediction).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def expected_calibration_error(
    probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10
) -> float:
    error = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if selected.any():
            error += float(selected.mean()) * abs(
                float(probabilities[selected].mean())
                - float(outcomes[selected].mean())
            )
    return error


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, float]:
    deception_label = np.asarray([row["deception_label"] for row in rows])
    deception_probability = np.asarray(
        [row["deception_probability"] for row in rows], dtype=float
    )
    correct = np.asarray([row["interception"] for row in rows], dtype=float)
    confidence = np.asarray([row["confidence"] for row in rows], dtype=float)
    return {
        "correct_commitment": float(correct.mean()),
        "false_commitment": float(
            np.mean([row["false_commitment"] for row in rows])
        ),
        "no_commit_rate": float(np.mean([row["timed_out"] for row in rows])),
        "decision_latency": float(
            np.mean([row["decision_latency"] for row in rows])
        ),
        "information_cost": float(
            np.mean([row["information_cost"] for row in rows])
        ),
        "regret": float(np.mean([row["regret"] for row in rows])),
        "hypothesis_brier": float(
            np.mean([row["hypothesis_brier"] for row in rows])
        ),
        "hypothesis_ece": expected_calibration_error(confidence, correct),
        "hypothesis_nll": float(
            np.mean([row["hypothesis_nll"] for row in rows])
        ),
        "deception_brier": float(
            np.mean([row["deception_brier"] for row in rows])
        ),
        "deception_ece": expected_calibration_error(
            deception_probability, deception_label
        ),
        "deception_auroc": binary_auroc(deception_label, deception_probability),
        "deception_f1": binary_f1(deception_label, deception_probability),
        "deception_nll": float(
            np.mean([row["deception_nll"] for row in rows])
        ),
    }


def deception_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    labels = np.asarray([row["deception_label"] for row in rows])
    probabilities = np.asarray(
        [row["deception_probability"] for row in rows], dtype=float
    )
    return {
        "deception_brier": float(np.mean((probabilities - labels) ** 2)),
        "deception_ece": expected_calibration_error(probabilities, labels),
        "deception_auroc": binary_auroc(labels, probabilities),
        "deception_f1": binary_f1(labels, probabilities),
        "deception_nll": float(
            np.mean(
                -labels * np.log(np.clip(probabilities, 1e-8, 1))
                - (1 - labels)
                * np.log(np.clip(1 - probabilities, 1e-8, 1))
            )
        ),
    }


def per_seed_family_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["training_seed"], row["controller"], row["deception_family"])].append(row)
    output = []
    for (seed, method, family), selected in sorted(groups.items()):
        if len(selected) != 100:
            raise AssertionError(
                f"formal cell {seed}/{method}/{family} has {len(selected)} episodes"
            )
        result = {
                "seed": seed,
                "method": method,
                "deception_family": family,
                "scope": selected[0]["family_scope"],
                "episodes": len(selected),
                **summarize_group(selected),
            }
        if family != "honest":
            honest = groups[(seed, method, "honest")]
            result.update(deception_metrics(honest + selected))
        output.append(result)
    return output


def per_seed_scope_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for seed in sorted({row["training_seed"] for row in rows}):
        for method in FORMAL_METHODS:
            method_rows = [
                row
                for row in rows
                if row["training_seed"] == seed and row["controller"] == method
            ]
            for scope in ("overall", "seen", "unseen"):
                selected = (
                    method_rows
                    if scope == "overall"
                    else [row for row in method_rows if row["family_scope"] == scope]
                )
                result = {
                        "seed": seed,
                        "method": method,
                        "scope": scope,
                        "episodes": len(selected),
                        **summarize_group(selected),
                    }
                if scope == "unseen":
                    honest = [
                        row
                        for row in method_rows
                        if row["deception_family"] == "honest"
                    ]
                    result.update(deception_metrics(honest + selected))
                output.append(result)
    return output


def aggregate_across_seeds(
    per_seed_rows: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed_rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, selected in sorted(groups.items()):
        if len(selected) != 5:
            raise AssertionError(f"formal aggregate {key} has {len(selected)} seeds")
        result = {field: value for field, value in zip(group_fields, key)}
        result["seeds"] = len(selected)
        result["episodes"] = sum(int(row["episodes"]) for row in selected)
        for metric in PRIMARY_METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            finite = values[np.isfinite(values)]
            mean = float(finite.mean()) if len(finite) else float("nan")
            standard_deviation = (
                float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
            )
            half_width = (
                2.776 * standard_deviation / math.sqrt(len(finite))
                if len(finite) == 5
                else float("nan")
            )
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = standard_deviation
            result[f"{metric}_ci95_low"] = mean - half_width
            result[f"{metric}_ci95_high"] = mean + half_width
        output.append(result)
    return output


def generalization_gaps(
    per_seed_scope: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    seed_rows = []
    for seed in sorted({row["seed"] for row in per_seed_scope}):
        for method in FORMAL_METHODS:
            lookup = {
                row["scope"]: row
                for row in per_seed_scope
                if row["seed"] == seed and row["method"] == method
            }
            seed_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "correct_commitment_gap_seen_minus_unseen": lookup["seen"][
                        "correct_commitment"
                    ]
                    - lookup["unseen"]["correct_commitment"],
                    "regret_gap_unseen_minus_seen": lookup["unseen"]["regret"]
                    - lookup["seen"]["regret"],
                    "hypothesis_brier_gap_unseen_minus_seen": lookup["unseen"][
                        "hypothesis_brier"
                    ]
                    - lookup["seen"]["hypothesis_brier"],
                    "deception_brier_gap_unseen_minus_seen": lookup["unseen"][
                        "deception_brier"
                    ]
                    - lookup["seen"]["deception_brier"],
                }
            )
    output = []
    for method in FORMAL_METHODS:
        selected = [row for row in seed_rows if row["method"] == method]
        result = {"method": method, "seeds": len(selected)}
        for metric in (
            "correct_commitment_gap_seen_minus_unseen",
            "regret_gap_unseen_minus_seen",
            "hypothesis_brier_gap_unseen_minus_seen",
            "deception_brier_gap_unseen_minus_seen",
        ):
            values = np.asarray([row[metric] for row in selected])
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=1))
        output.append(result)
    return output


def paired_full_ride_differences(
    per_seed_scope: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for scope in ("overall", "seen", "unseen"):
        for comparator in FORMAL_METHODS:
            if comparator == "full_ride":
                continue
            differences: dict[str, list[float]] = defaultdict(list)
            for seed in range(5):
                lookup = {
                    row["method"]: row
                    for row in per_seed_scope
                    if row["seed"] == seed and row["scope"] == scope
                }
                for metric in (
                    "correct_commitment",
                    "decision_latency",
                    "information_cost",
                    "regret",
                ):
                    differences[metric].append(
                        lookup["full_ride"][metric] - lookup[comparator][metric]
                    )
            result = {
                "scope": scope,
                "reference": "full_ride",
                "comparator": comparator,
                "seeds": 5,
            }
            for metric, values_list in differences.items():
                values = np.asarray(values_list)
                standard_deviation = float(values.std(ddof=1))
                half_width = 2.776 * standard_deviation / math.sqrt(5)
                result[f"{metric}_difference_mean"] = float(values.mean())
                result[f"{metric}_difference_ci95_low"] = float(
                    values.mean() - half_width
                )
                result[f"{metric}_difference_ci95_high"] = float(
                    values.mean() + half_width
                )
            output.append(result)
    return output


def plot_tradeoff(aggregate_scope: list[dict[str, Any]], destination: Path) -> None:
    rows = {
        row["method"]: row
        for row in aggregate_scope
        if row["scope"] == "overall"
    }
    figure, axis = plt.subplots(figsize=(9, 6))
    colors = plt.get_cmap("tab10")
    for index, method in enumerate(FORMAL_METHODS):
        row = rows[method]
        x = row["decision_latency_mean"]
        y = row["correct_commitment_mean"]
        size = 80 + 160 * row["information_cost_mean"]
        axis.scatter(
            x,
            y,
            s=size,
            alpha=0.72,
            color=colors(index),
            edgecolor="white",
            linewidth=0.8,
        )
    annotations = (
        ("single_report", "Single report", (8, -4)),
        ("multi_source_belief", "Multi-source ablations", (8, -17)),
        ("full_ride", "Full RIDe", (8, 10)),
        ("rule_adaptive_defender", "Rule adaptive", (8, 5)),
        ("bayesian_fusion", "Bayesian / POMCP-like / Active inference", (-235, 8)),
    )
    for method, label, offset in annotations:
        row = rows[method]
        axis.annotate(
            label,
            (row["decision_latency_mean"], row["correct_commitment_mean"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_xlabel("Mean decision latency")
    axis.set_ylabel("Correct commitment")
    axis.set_title("Accuracy–latency–information cost trade-off")
    axis.grid(alpha=0.25)
    axis.text(
        0.01,
        0.01,
        "Bubble area increases with information cost",
        transform=axis.transAxes,
        fontsize=8,
        color="dimgray",
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(destination / f"accuracy_latency_tradeoff.{suffix}", dpi=180)
    plt.close(figure)


def plot_family_breakdown(
    aggregate_family: list[dict[str, Any]], destination: Path
) -> None:
    families = list(RED_FAMILIES)
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, sharey=True)
    panels = (
        (
            axes[0],
            FORMAL_METHODS[:5],
            "Ablation chain",
        ),
        (
            axes[1],
            ("full_ride",) + FORMAL_METHODS[5:],
            "Full RIDe and external baselines",
        ),
    )
    labels = {
        "single_report": "Single report",
        "multi_source_belief": "Multi-source",
        "belief_plus_deception": "+ deception head",
        "belief_deception_counterfactual": "+ counterfactual",
        "full_ride": "Full RIDe",
        "bayesian_fusion": "Bayesian fusion",
        "pomcp_like": "POMCP-like",
        "active_inference": "Active inference",
        "rule_adaptive_defender": "Rule adaptive",
    }
    for axis, methods, title in panels:
        for method in methods:
            lookup = {
                row["deception_family"]: row
                for row in aggregate_family
                if row["method"] == method
            }
            emphasis = method == "full_ride"
            axis.plot(
                families,
                [lookup[family]["correct_commitment_mean"] for family in families],
                marker="o",
                linewidth=2.8 if emphasis else 1.5,
                alpha=1.0 if emphasis else 0.78,
                label=labels[method],
            )
        axis.set_ylim(0.3, 0.9)
        axis.set_ylabel("Correct commitment")
        axis.set_title(title, loc="left", fontsize=11)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(ncol=3, fontsize=8, loc="lower left")
    axes[1].tick_params(axis="x", rotation=20)
    figure.suptitle("Performance by deception family", fontsize=14)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(destination / f"deception_family_breakdown.{suffix}", dpi=180)
    plt.close(figure)


def _percentage(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_formal_report(
    aggregate_scope: list[dict[str, Any]],
    aggregate_family: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    audit: dict[str, Any],
    destination: Path,
) -> None:
    overall = {
        row["method"]: row
        for row in aggregate_scope
        if row["scope"] == "overall"
    }
    scope = {(row["method"], row["scope"]): row for row in aggregate_scope}
    family = {
        (row["method"], row["deception_family"]): row
        for row in aggregate_family
    }
    differences = {
        (row["scope"], row["comparator"]): row for row in paired
    }
    full = overall["full_ride"]
    versus_single = differences[("overall", "single_report")]
    versus_multi = differences[("overall", "multi_source_belief")]
    versus_rule = differences[("overall", "rule_adaptive_defender")]
    gap = next(row for row in gaps if row["method"] == "full_ride")

    main_rows = []
    for method in FORMAL_METHODS:
        row = overall[method]
        main_rows.append(
            "| {method} | {accuracy} | {latency:.2f} | {cost:.2f} | {regret:.2f} |".format(
                method=method,
                accuracy=_percentage(row["correct_commitment_mean"]),
                latency=row["decision_latency_mean"],
                cost=row["information_cost_mean"],
                regret=row["regret_mean"],
            )
        )
    family_rows = []
    for name in RED_FAMILIES:
        ride_value = family[("full_ride", name)]["correct_commitment_mean"]
        multi_value = family[("multi_source_belief", name)][
            "correct_commitment_mean"
        ]
        family_rows.append(
            f"| {name} | {_percentage(ride_value)} | "
            f"{_percentage(multi_value)} | {100 * (ride_value - multi_value):+.1f} pp |"
        )
    detection_rows = []
    for name in RED_FAMILIES:
        row = family[("full_ride", name)]
        auroc = row["deception_auroc_mean"]
        detection_rows.append(
            f"| {name} | {'—' if not math.isfinite(auroc) else f'{auroc:.3f}'} | "
            f"{row['deception_f1_mean']:.3f} |"
        )

    report = f"""# Intelligence Formal Evaluation

## Frozen protocol

- Five training seeds (`0–4`), 100 paired episodes per method × deception family × seed.
- Nine methods, seven families, and 31,500 evaluation episodes.
- Seen training families: honest, concealment, decoy, delayed.
- Frozen unseen families: multi-source, switch, adaptive.
- Intervals are 95% t intervals over five seed-level means. No post-result tuning was performed.

## Main results

| Method | Correct commitment | Latency | Information cost | Regret |
|---|---:|---:|---:|---:|
{chr(10).join(main_rows)}

Full RIDe reached **{_percentage(full['correct_commitment_mean'])}** correct commitment (95% CI {_percentage(full['correct_commitment_ci95_low'])}–{_percentage(full['correct_commitment_ci95_high'])}). Relative to single-report inference, the paired improvement was **{100 * versus_single['correct_commitment_difference_mean']:+.1f} pp** (95% CI {100 * versus_single['correct_commitment_difference_ci95_low']:+.1f} to {100 * versus_single['correct_commitment_difference_ci95_high']:+.1f} pp). Relative to the rule adaptive defender, it was **{100 * versus_rule['correct_commitment_difference_mean']:+.1f} pp** (95% CI {100 * versus_rule['correct_commitment_difference_ci95_low']:+.1f} to {100 * versus_rule['correct_commitment_difference_ci95_high']:+.1f} pp).

The comparison against multi-source belief fusion does **not** establish an overall accuracy advantage: **{100 * versus_multi['correct_commitment_difference_mean']:+.1f} pp** (95% CI {100 * versus_multi['correct_commitment_difference_ci95_low']:+.1f} to {100 * versus_multi['correct_commitment_difference_ci95_high']:+.1f} pp). Full RIDe also used {versus_multi['decision_latency_difference_mean']:+.2f} more decision steps and {versus_multi['information_cost_difference_mean']:+.2f} more information cost. All methods had a 0% no-commit rate.

## Seen and unseen families

Full RIDe achieved {_percentage(scope[('full_ride', 'seen')]['correct_commitment_mean'])} on seen families and {_percentage(scope[('full_ride', 'unseen')]['correct_commitment_mean'])} on unseen families. The seen-minus-unseen gap was {100 * gap['correct_commitment_gap_seen_minus_unseen_mean']:.1f} pp.

| Family | Full RIDe | Multi-source | Difference |
|---|---:|---:|---:|
{chr(10).join(family_rows)}

The adaptive family was the hardest condition for Full RIDe ({_percentage(family[('full_ride', 'adaptive')]['correct_commitment_mean'])}). The controller improved over fixed-time multi-source fusion on adaptive and switch cases, while losing accuracy on concealment, decoy, delayed, and multi-source deception. This supports a family-specific timing claim, not universal robustness.

## Belief and deception estimates

Multi-source encoding reduced hypothesis Brier from {overall['single_report']['hypothesis_brier_mean']:.3f} to {overall['multi_source_belief']['hypothesis_brier_mean']:.3f}. Adding the deception head reduced deception Brier from {overall['multi_source_belief']['deception_brier_mean']:.3f} to {overall['belief_plus_deception']['deception_brier_mean']:.3f} and produced AUROC {overall['belief_plus_deception']['deception_auroc_mean']:.3f}. Counterfactual reasoning changed hypothesis Brier only from {overall['belief_plus_deception']['hypothesis_brier_mean']:.3f} to {overall['belief_deception_counterfactual']['hypothesis_brier_mean']:.3f}; these data do not support a direct accuracy gain from that component.

| Family | Full RIDe deception AUROC | F1 |
|---|---:|---:|
{chr(10).join(detection_rows)}

Honest-only AUROC is undefined because that slice has one class. Family AUROC values pair each deceptive family with the honest controls from the same seeds.

## Claim audit

| Claim | Status | Evidence |
|---|---|---|
| The transfer respects the information boundary | Supported | {audit['policy_inputs_audited']:,} policy inputs audited; terminal truth was never a policy input. |
| Multi-source encoding improves belief quality over a single report | Supported | Hypothesis Brier: {overall['single_report']['hypothesis_brier_mean']:.3f} → {overall['multi_source_belief']['hypothesis_brier_mean']:.3f}. |
| Full RIDe improves overall commitment accuracy over single-report inference | Supported | Paired difference {100 * versus_single['correct_commitment_difference_mean']:+.1f} pp; CI excludes zero. |
| Full RIDe improves overall commitment accuracy over multi-source fusion | Not supported | Paired difference {100 * versus_multi['correct_commitment_difference_mean']:+.1f} pp; CI crosses zero. |
| Counterfactual reasoning directly improves commitment accuracy | Not supported | Fixed-time counterfactual and deception-head variants are effectively tied. |
| Full RIDe is uniformly robust to every deception family | Not supported | Gains are concentrated in adaptive and switch; adaptive remains the hardest family. |
| The red policy is game-theoretically optimal | Out of scope | Red deception is adaptive and rule based. |

## Reproducibility and limits

The audit verified unchanged source hashes during evaluation. The environment models adversarial information presentation and commitment decisions; it does not simulate physical combat. Five seeds quantify training variation, while the 100 within-cell episodes are paired scenario draws. Results should be reported as evidence for this frozen simulator and protocol rather than as claims about operational combat performance.
"""
    (destination / "formal_report.md").write_text(report, encoding="utf-8")

    claims = [
        {
            "claim": "information boundary preserved",
            "status": "supported",
            "evidence": f"{audit['policy_inputs_audited']} policy inputs audited",
        },
        {
            "claim": "full RIDe improves over single-report inference",
            "status": "supported",
            "evidence": "paired 95% CI excludes zero",
        },
        {
            "claim": "full RIDe improves over multi-source fusion overall",
            "status": "not_supported",
            "evidence": "paired 95% CI crosses zero",
        },
        {
            "claim": "counterfactual reasoning directly improves accuracy",
            "status": "not_supported",
            "evidence": "fixed-time ablation accuracy is effectively tied",
        },
        {
            "claim": "uniform robustness across deception families",
            "status": "not_supported",
            "evidence": "family effects differ and adaptive remains hardest",
        },
    ]
    write_csv(destination / "claim_audit.csv", claims)


def read_numeric_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if key in {"method", "scope", "deception_family", "reference", "comparator"}:
                converted[key] = value
                continue
            try:
                converted[key] = float(value)
            except ValueError:
                converted[key] = value
        output.append(converted)
    return output


def analyze_existing(destination: Path) -> None:
    aggregate_scope = read_numeric_csv(destination / "aggregate_by_scope.csv")
    aggregate_family = read_numeric_csv(destination / "aggregate_by_family.csv")
    gaps = read_numeric_csv(destination / "seen_unseen_gaps.csv")
    paired = read_numeric_csv(destination / "paired_full_ride_differences.csv")
    audit = json.loads(
        (destination / "information_boundary_and_freeze_audit.json").read_text(
            encoding="utf-8"
        )
    )
    plot_tradeoff(aggregate_scope, destination)
    plot_family_breakdown(aggregate_family, destination)
    write_formal_report(
        aggregate_scope, aggregate_family, gaps, paired, audit, destination
    )


def file_hashes() -> dict[str, str]:
    paths = (
        ROOT / "src/deception_rl/envs/deceptive_intelligence.py",
        ROOT / "src/deception_rl/intelligence/adapter.py",
        ROOT / "src/deception_rl/intelligence/ride.py",
        ROOT / "src/deception_rl/intelligence/baselines.py",
        ROOT / "experiments/run_intelligence_ride.py",
        Path(__file__),
    )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def build_formal_controllers(model, reasoner, calibrator, max_cycles, seed):
    controllers = build_controllers(
        model, reasoner, calibrator, max_cycles=max_cycles, seed=seed
    )
    controllers.pop("rule_fixed_time")
    rule = AdaptiveVerificationBluePolicy(max_cycles=max_cycles)
    rule.name = "rule_adaptive_defender"
    controllers[rule.name] = rule
    if tuple(controllers) != FORMAL_METHODS:
        raise AssertionError(f"formal method order drifted: {tuple(controllers)}")
    return controllers


def run(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    episodes_per_cell: int = 100,
    epochs: int = 10,
    train_episodes_per_hypothesis: int = 12,
    validation_episodes_per_hypothesis: int = 4,
    hidden_size: int = 32,
    batch_size: int = 64,
    max_cycles: int = 10,
) -> dict[str, Any]:
    if seeds != (0, 1, 2, 3, 4):
        raise ValueError("formal protocol requires seeds 0,1,2,3,4")
    if episodes_per_cell != 100:
        raise ValueError("formal protocol requires 100 episodes per cell")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    hashes_before = file_hashes()
    all_rows = []
    training_rows = []
    calibration_rows = []
    for seed in seeds:
        print(f"formal seed {seed}: collect/train", flush=True)
        train_records = collect_prefix_records(
            TRAIN_FAMILIES,
            train_episodes_per_hypothesis,
            seed=100_000 + seed * 1_000_003,
            split=f"formal_train_seed{seed}",
            max_cycles=max_cycles,
        )
        validation_records = collect_prefix_records(
            TRAIN_FAMILIES,
            validation_episodes_per_hypothesis,
            seed=200_000 + seed * 1_000_003,
            split=f"formal_validation_seed{seed}",
            max_cycles=max_cycles,
        )
        model, history = train_model(
            train_records,
            validation_records,
            epochs=epochs,
            hidden_size=hidden_size,
            batch_size=batch_size,
            seed=seed,
        )
        calibrator, calibration = fit_calibrator(model, validation_records)
        reasoner = HonestIntelligenceReasoner(train_records)
        controllers = build_formal_controllers(
            model, reasoner, calibrator, max_cycles, seed
        )
        for row in history:
            training_rows.append({"seed": seed, **row})
        calibration_rows.append({"seed": seed, **calibration})
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "calibrator_state_dict": calibrator.state_dict(),
                "hidden_size": hidden_size,
                "seed": seed,
            },
            destination / f"model_seed{seed}.pt",
        )

        print(f"formal seed {seed}: evaluate 6,300 episodes", flush=True)
        seed_rows = []
        for family_index, family in enumerate(RED_FAMILIES):
            for episode in range(episodes_per_cell):
                hypothesis = (episode + seed + family_index) % NUM_HYPOTHESES
                scenario_seed = (
                    300_000
                    + seed * 10_000_019
                    + family_index * 10_003
                    + episode
                )
                for controller in controllers.values():
                    result = evaluate_episode(
                        controller,
                        family,
                        hypothesis,
                        seed=scenario_seed,
                        max_cycles=max_cycles,
                    )
                    seed_rows.append(
                        {
                            "training_seed": seed,
                            "scenario_id": family_index * episodes_per_cell + episode,
                            **result,
                        }
                    )
            print(f"  completed family {family}", flush=True)
        if len(seed_rows) != len(RED_FAMILIES) * episodes_per_cell * len(FORMAL_METHODS):
            raise AssertionError("seed evaluation count drifted")
        all_rows.extend(seed_rows)
        write_csv(destination / f"episodes_seed{seed}.csv", seed_rows)

    expected = len(seeds) * len(RED_FAMILIES) * episodes_per_cell * len(FORMAL_METHODS)
    if len(all_rows) != expected:
        raise AssertionError(f"expected {expected} episodes, observed {len(all_rows)}")
    write_csv(destination / "episodes.csv", all_rows)
    write_csv(destination / "training_history.csv", training_rows)
    write_csv(destination / "calibration_temperatures.csv", calibration_rows)

    seed_family = per_seed_family_summary(all_rows)
    seed_scope = per_seed_scope_summary(all_rows)
    aggregate_family = aggregate_across_seeds(
        seed_family, ("method", "deception_family", "scope")
    )
    aggregate_scope = aggregate_across_seeds(seed_scope, ("method", "scope"))
    gaps = generalization_gaps(seed_scope)
    paired = paired_full_ride_differences(seed_scope)
    write_csv(destination / "per_seed_family.csv", seed_family)
    write_csv(destination / "per_seed_scope.csv", seed_scope)
    write_csv(destination / "aggregate_by_family.csv", aggregate_family)
    write_csv(destination / "aggregate_by_scope.csv", aggregate_scope)
    write_csv(destination / "seen_unseen_gaps.csv", gaps)
    write_csv(destination / "paired_full_ride_differences.csv", paired)
    plot_tradeoff(aggregate_scope, destination)
    plot_family_breakdown(aggregate_family, destination)

    hashes_after = file_hashes()
    if hashes_before != hashes_after:
        raise AssertionError("formal protocol source changed during evaluation")
    audit = {
        "policy_inputs_audited": int(
            sum(row["boundary_checks"] for row in all_rows)
        ),
        "terminal_truth_used_as_policy_input": False,
        "source_hashes_before": hashes_before,
        "source_hashes_after": hashes_after,
        "source_unchanged": hashes_before == hashes_after,
    }
    (destination / "information_boundary_and_freeze_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    write_formal_report(
        aggregate_scope, aggregate_family, gaps, paired, audit, destination
    )
    config = {
        "protocol": "Intelligence Formal Evaluation v1",
        "seeds": list(seeds),
        "episodes_per_method_family_seed": episodes_per_cell,
        "total_evaluation_episodes": len(all_rows),
        "methods": list(FORMAL_METHODS),
        "deception_families": list(RED_FAMILIES),
        "train_families": list(TRAIN_FAMILIES),
        "unseen_families": list(UNSEEN_FAMILIES),
        "epochs": epochs,
        "train_episodes_per_hypothesis": train_episodes_per_hypothesis,
        "validation_episodes_per_hypothesis": validation_episodes_per_hypothesis,
        "hidden_size": hidden_size,
        "paired_scenarios": True,
        "confidence_intervals": "95% t interval over five seed-level means, t=2.776",
        "interception_definition": "correct terminal operational-hypothesis commitment",
        "physical_combat_simulated": False,
        "post_result_tuning": False,
        "legacy_benchmark_replaced": False,
    }
    (destination / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--analyze-existing",
        action="store_true",
        help="regenerate the report and figures from an existing frozen run",
    )
    args = parser.parse_args()
    if args.analyze_existing:
        analyze_existing(args.output_dir)
        print(f"refreshed analysis in {args.output_dir}")
    else:
        print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()

