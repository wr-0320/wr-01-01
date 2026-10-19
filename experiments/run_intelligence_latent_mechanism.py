"""Fresh-seed label-efficiency study of semi-supervised latent-mechanism RCL."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES
from deception_rl.intelligence.ride import HonestIntelligenceReasoner
from deception_rl.intelligence.ride_v2.fixed_reliability import FixedCellReliability
from deception_rl.intelligence.ride_v2.latent_mechanism import (
    fit_latent_mechanism_rcl,
    fit_unlabeled_single_mechanism,
    mechanism_responsibilities,
)
from deception_rl.intelligence.ride_v2.reliability_conditioned import (
    ReliabilityConditionedLikelihood,
    ReliabilityLikelihoodController,
)
from experiments.run_intelligence_formal_evaluation import summarize_group
from experiments.run_intelligence_rcl_formal import (
    METRICS,
    SingleReliableLikelihood,
    source_manifest,
)
from experiments.run_intelligence_reliability_conditioned_inference import (
    collect_belief_cases,
    fit_likelihood_temperature,
)
from experiments.run_intelligence_ride import (
    RED_FAMILIES,
    TRAIN_FAMILIES,
    collect_prefix_records,
    evaluate_episode,
    train_model,
)
from experiments.run_intelligence_ride import fit_calibrator
from experiments.run_intelligence_ride_v2_headroom import directory_manifest, write_csv
from experiments.run_intelligence_ride_v2_reliability import (
    RELIABILITY_TRAIN_FAMILIES,
    _inference,
    collect_reliability_samples,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "intelligence_latent_mechanism_development"
SEEDS = (7101, 7102, 7103)
LABEL_RATIOS = (0.0, 0.05, 0.10, 0.25)
METHODS = (
    "multi_source",
    "unlabeled_single",
    "supervised_single",
    "supervised_rcl_100",
    "lm_rcl_0",
    "lm_rcl_5",
    "lm_rcl_10",
    "lm_rcl_25",
    "sparse_supervised_rcl_5",
    "sparse_supervised_rcl_10",
    "sparse_supervised_rcl_25",
)


def _nested_label_masks(
    samples: list[dict[str, Any]], ratios: tuple[float, ...], seed: int
) -> dict[float, list[dict[str, Any]]]:
    locations = []
    for row_index, row in enumerate(samples):
        for source, region in np.argwhere(np.asarray(row["reported_cells"], dtype=bool)):
            locations.append((row_index, int(source), int(region)))
    order = np.random.default_rng(seed).permutation(len(locations))
    output = {}
    for ratio in ratios:
        selected = set(order[: int(round(float(ratio) * len(order)))].tolist())
        masks = [np.zeros((3, 3), dtype=bool) for _ in samples]
        for location_index in selected:
            row_index, source, region = locations[location_index]
            masks[row_index][source, region] = True
        output[ratio] = [dict(row, mechanism_label_mask=mask) for row, mask in zip(samples, masks)]
    return output


def _sparse_supervised_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in samples:
        visible = np.asarray(row["reported_cells"], dtype=bool)
        labeled = np.asarray(row["mechanism_label_mask"], dtype=bool)
        output.append(dict(row, reported_cells=visible & labeled))
    return output


def _diagnostics(probability: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    positives = probability[truth == 1]
    negatives = np.sort(probability[truth == 0])
    if len(positives) and len(negatives):
        lower = np.searchsorted(negatives, positives, side="left")
        upper = np.searchsorted(negatives, positives, side="right")
        auroc = float(np.mean((lower + upper) / 2.0) / len(negatives))
    else:
        auroc = float("nan")
    return {
        "cells": int(len(truth)),
        "compromised_prevalence": float(truth.mean()),
        "mechanism_brier": float(np.mean((probability - truth) ** 2)),
        "mechanism_auroc": auroc,
    }


def _analyze(rows: list[dict[str, Any]]):
    seeds = sorted({int(row["training_seed"]) for row in rows})
    scopes = ("overall", "seen", "unseen", *RED_FAMILIES)
    per_seed = []
    for seed in seeds:
        for method in METHODS:
            base = [
                row
                for row in rows
                if int(row["training_seed"]) == seed and row["controller"] == method
            ]
            for scope in scopes:
                selected = (
                    base
                    if scope == "overall"
                    else [
                        row
                        for row in base
                        if (
                            row["family_scope"] == scope
                            if scope in ("seen", "unseen")
                            else row["deception_family"] == scope
                        )
                    ]
                )
                per_seed.append(
                    {
                        "seed": seed,
                        "method": method,
                        "scope": scope,
                        "episodes": len(selected),
                        **{
                            metric: value
                            for metric, value in summarize_group(selected).items()
                            if metric in METRICS
                        },
                    }
                )
    lookup = {(row["seed"], row["method"], row["scope"]): row for row in per_seed}
    summary = []
    for method in METHODS:
        for scope in scopes:
            item = {
                "method": method,
                "scope": scope,
                "seeds": len(seeds),
                "episodes": sum(lookup[seed, method, scope]["episodes"] for seed in seeds),
            }
            for metric in METRICS:
                values = np.asarray([lookup[seed, method, scope][metric] for seed in seeds])
                half = 4.303 * values.std(ddof=1) / math.sqrt(3) if len(seeds) == 3 else float("nan")
                item[metric] = float(values.mean())
                item[metric + "_ci95_low"] = float(values.mean() - half)
                item[metric + "_ci95_high"] = float(values.mean() + half)
            summary.append(item)
    paired = []
    comparators = (
        "supervised_rcl_100",
        "unlabeled_single",
        "multi_source",
        "sparse_supervised_rcl_10",
    )
    for method in METHODS:
        for comparator in comparators:
            if method == comparator:
                continue
            for scope in scopes:
                for metric in METRICS:
                    values = np.asarray(
                        [
                            lookup[seed, method, scope][metric]
                            - lookup[seed, comparator, scope][metric]
                            for seed in seeds
                        ]
                    )
                    half = (
                        4.303 * values.std(ddof=1) / math.sqrt(3)
                        if len(seeds) == 3
                        else float("nan")
                    )
                    paired.append(
                        {
                            "method": method,
                            "comparator": comparator,
                            "scope": scope,
                            "metric": metric,
                            "difference": float(values.mean()),
                            "ci95_low": float(values.mean() - half),
                            "ci95_high": float(values.mean() + half),
                        }
                    )
    return per_seed, summary, paired


def run(
    output_dir: Path = OUTPUT,
    *,
    seeds: tuple[int, ...] = SEEDS,
    episodes: int = 100,
    train_episodes: int = 10,
    neural_train_episodes: int = 8,
    neural_validation_episodes: int = 3,
    epochs: int = 8,
    smoke: bool = False,
):
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError("Refusing to overwrite outputs")
    if not smoke and (
        tuple(seeds) != SEEDS
        or episodes != 100
        or train_episodes != 10
        or neural_train_episodes != 8
        or neural_validation_episodes != 3
        or epochs != 8
    ):
        raise ValueError("Frozen development protocol")
    frozen = {
        path.name: path
        for path in (ROOT / "outputs").iterdir()
        if path.is_dir() and (path.name.startswith("intelligence_") or path.name.startswith("thesis_"))
    }
    before = {name: directory_manifest(path) for name, path in frozen.items()}
    sources = source_manifest()
    destination.mkdir(parents=True)
    protocol = {
        "protocol": "Latent-Mechanism RCL Label-Efficiency Development v1",
        "seeds": list(seeds),
        "fit_families": list(RELIABILITY_TRAIN_FAMILIES),
        "test_families": list(RED_FAMILIES),
        "label_ratios": list(LABEL_RATIOS),
        "label_unit": "visible source-region report cell",
        "label_selection": "nested random subsets independent of mechanism value",
        "methods": list(METHODS),
        "episodes_per_method_family_seed": episodes,
        "expected_test_episodes": len(seeds) * len(RED_FAMILIES) * episodes * len(METHODS),
        "em_max_iterations": 100,
        "em_tolerance": 1e-6,
        "scale_floor": 0.08,
        "fixed_commit_step": 4,
        "primary_condition": "lm_rcl_10",
        "primary_development_gate": "LM-10 accuracy within 2 pp and Brier within 0.03 of supervised RCL, while outperforming sparse-supervised-10 in accuracy; point-estimate screening, not a noninferiority test",
        "task_gain_required": True,
        "mechanism_fit_alone_sufficient": False,
        "hypothesis_labels_available_for_all_training_reports": True,
        "mechanism_labels_hidden_unless_selected": True,
        "multi_source_uses_episode_deception_family_supervision": True,
        "fresh_development_not_formal": True,
        "smoke_only": smoke,
    }
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (destination / "source_manifest.json").write_text(json.dumps(sources, indent=2))

    all_rows = []
    fit_rows = []
    for seed in seeds:
        neural_train = collect_prefix_records(
            TRAIN_FAMILIES,
            neural_train_episodes,
            seed=5_000_000_000 + seed * 10003,
            split=f"lm_neural_train_{seed}",
            max_cycles=10,
        )
        neural_validation = collect_prefix_records(
            TRAIN_FAMILIES,
            neural_validation_episodes,
            seed=5_050_000_000 + seed * 10003,
            split=f"lm_neural_validation_{seed}",
            max_cycles=10,
        )
        neural_model, _ = train_model(
            neural_train,
            neural_validation,
            epochs=epochs,
            hidden_size=32,
            batch_size=64,
            seed=seed,
        )
        reasoner = HonestIntelligenceReasoner(neural_train)
        samples = collect_reliability_samples(
            RELIABILITY_TRAIN_FAMILIES,
            train_episodes,
            seed=5_100_000_000 + seed * 10003,
            reasoner=reasoner,
            split=f"lm_likelihood_{seed}",
            max_cycles=10,
        )
        masks = _nested_label_masks(samples, LABEL_RATIOS, seed=5_150_000_000 + seed)
        calibration_cases = collect_belief_cases(
            RELIABILITY_TRAIN_FAMILIES,
            12 if not smoke else 1,
            seed=5_200_000_000 + seed * 10003,
            max_cycles=10,
        )

        full_fixed = FixedCellReliability.fit(samples)
        full_raw = ReliabilityConditionedLikelihood.fit(samples)
        full = fit_likelihood_temperature(full_raw, calibration_cases, full_fixed, reasoner)
        supervised_single_fixed = FixedCellReliability(np.ones_like(full_fixed.reliable))
        supervised_single_raw = SingleReliableLikelihood(
            full_raw.reliable_mean,
            full_raw.reliable_scale,
            full_raw.compromised_mean,
            full_raw.compromised_scale,
        )
        supervised_single = fit_likelihood_temperature(
            supervised_single_raw, calibration_cases, supervised_single_fixed, reasoner
        )
        unlabeled_single_raw = fit_unlabeled_single_mechanism(samples)
        unlabeled_single_raw = SingleReliableLikelihood(
            unlabeled_single_raw.reliable_mean,
            unlabeled_single_raw.reliable_scale,
            unlabeled_single_raw.compromised_mean,
            unlabeled_single_raw.compromised_scale,
        )
        unlabeled_single = fit_likelihood_temperature(
            unlabeled_single_raw, calibration_cases, supervised_single_fixed, reasoner
        )

        models: dict[str, tuple[Any, FixedCellReliability]] = {
            "unlabeled_single": (unlabeled_single, supervised_single_fixed),
            "supervised_single": (supervised_single, supervised_single_fixed),
            "supervised_rcl_100": (full, full_fixed),
        }
        parameter_dump: dict[str, Any] = {
            "full_supervised_reliability": full_fixed.reliable.tolist(),
            "full_supervised_temperature": full.temperature,
        }
        for ratio in LABEL_RATIOS:
            result = fit_latent_mechanism_rcl(masks[ratio])
            calibrated = fit_likelihood_temperature(
                result.likelihood, calibration_cases, result.reliability, reasoner
            )
            name = f"lm_rcl_{int(round(100 * ratio))}"
            models[name] = (calibrated, result.reliability)
            probability, truth = mechanism_responsibilities(result, samples)
            diagnostic = _diagnostics(probability, truth)
            fit_rows.append(
                {
                    "seed": seed,
                    "method": name,
                    "requested_label_fraction": ratio,
                    "actual_label_fraction": result.labeled_fraction,
                    "iterations": result.iterations,
                    "converged": int(result.converged),
                    "observed_log_likelihood": result.observed_log_likelihood,
                    "temperature": calibrated.temperature,
                    **diagnostic,
                }
            )
            parameter_dump[name] = {
                "reliability": result.reliability.reliable.tolist(),
                "temperature": calibrated.temperature,
                "iterations": result.iterations,
                "converged": result.converged,
            }
            if ratio > 0:
                sparse_samples = _sparse_supervised_samples(masks[ratio])
                sparse_fixed = FixedCellReliability.fit(sparse_samples)
                sparse_raw = ReliabilityConditionedLikelihood.fit(sparse_samples)
                sparse = fit_likelihood_temperature(
                    sparse_raw, calibration_cases, sparse_fixed, reasoner
                )
                sparse_name = f"sparse_supervised_rcl_{int(round(100 * ratio))}"
                models[sparse_name] = (sparse, sparse_fixed)

        controllers = {
            "multi_source": _inference(
                neural_model,
                reasoner,
                None,
                "multi_source",
                full_ride=False,
                max_cycles=10,
            )
        }
        for name, (model, predictor) in models.items():
            controllers[name] = ReliabilityLikelihoodController(
                model, predictor, reasoner, name=name
            )
        if set(controllers) != set(METHODS):
            raise AssertionError(f"method drift: {set(METHODS) - set(controllers)}")
        (destination / f"parameters_seed{seed}.json").write_text(
            json.dumps(parameter_dump, indent=2)
        )

        seed_rows = []
        for family_index, family in enumerate(RED_FAMILIES):
            for episode in range(episodes):
                hypothesis = (episode + family_index + seed) % NUM_HYPOTHESES
                scenario_seed = 5_300_000_000 + seed * 100003 + family_index * 1009 + episode
                for method in METHODS:
                    row = evaluate_episode(
                        controllers[method], family, hypothesis, seed=scenario_seed, max_cycles=10
                    )
                    row["training_seed"] = seed
                    seed_rows.append(row)
            print(f"seed={seed} family={family} complete", flush=True)
        write_csv(destination / f"episodes_seed{seed}.csv", seed_rows)
        all_rows.extend(seed_rows)

    if len(all_rows) != protocol["expected_test_episodes"]:
        raise AssertionError("episode count drift")
    per_seed, summary, paired = _analyze(all_rows)
    write_csv(destination / "episodes.csv", all_rows)
    write_csv(destination / "per_seed_summary.csv", per_seed)
    write_csv(destination / "summary.csv", summary)
    write_csv(destination / "paired_differences.csv", paired)
    write_csv(destination / "mechanism_fit_diagnostics.csv", fit_rows)

    lookup = {(row["method"], row["scope"]): row for row in summary}
    difference_lookup = {
        (row["method"], row["comparator"], row["scope"], row["metric"]): row
        for row in paired
    }
    primary = lookup["lm_rcl_10", "overall"]
    supervised = lookup["supervised_rcl_100", "overall"]
    sparse = lookup["sparse_supervised_rcl_10", "overall"]
    go = bool(
        primary["correct_commitment"] >= supervised["correct_commitment"] - 0.02
        and primary["hypothesis_brier"] <= supervised["hypothesis_brier"] + 0.03
        and primary["correct_commitment"] > sparse["correct_commitment"]
    )
    claim = {
        "decision": "GO" if go else "NO-GO",
        "screening_only_not_noninferiority_test": True,
        "lm_10_accuracy_gap_vs_supervised": primary["correct_commitment"] - supervised["correct_commitment"],
        "lm_10_brier_gap_vs_supervised": primary["hypothesis_brier"] - supervised["hypothesis_brier"],
        "lm_10_accuracy_gain_vs_sparse_supervised": primary["correct_commitment"] - sparse["correct_commitment"],
        "lm_10_accuracy_difference_ci_vs_supervised": difference_lookup[
            "lm_rcl_10", "supervised_rcl_100", "overall", "correct_commitment"
        ],
        "lm_10_accuracy_difference_ci_vs_sparse_supervised": difference_lookup[
            "lm_rcl_10", "sparse_supervised_rcl_10", "overall", "correct_commitment"
        ],
        "third_innovation_established": False,
        "formal_results_replaced": False,
    }
    (destination / "claim_audit.json").write_text(json.dumps(claim, indent=2))
    audit = {name + "_unchanged": before[name] == directory_manifest(path) for name, path in frozen.items()}
    audit.update(
        {
            "source_unchanged": sources == source_manifest(),
            "nested_label_subsets": True,
            "label_selection_independent_of_mechanism_value": True,
            "unlabeled_mechanism_truth_read_by_fit": False,
            "runtime_mechanism_truth_used": False,
            "runtime_hypothesis_truth_used": False,
            "test_episodes": len(all_rows),
            "policy_inputs_audited": sum(int(row["boundary_checks"]) for row in all_rows),
        }
    )
    if not all(value for key, value in audit.items() if key.endswith("_unchanged")):
        raise AssertionError("frozen artifact changed")
    (destination / "audit.json").write_text(json.dumps(audit, indent=2))

    lines = [
        "# Latent-Mechanism RCL development",
        "",
        "Mechanism-label efficiency is the target. All methods receive every training report and its operational-hypothesis label; LM-RCL sees reliable/compromised labels only on the selected nested cell subset. Sparse-supervised controls discard unlabeled reports during mechanism fitting. Multi-source additionally retains its original episode-level deception-family supervision, so it is contextual rather than an equal-supervision control.",
        "",
        "| Method | Mechanism labels | Accuracy | Brier | ECE | NLL | Regret |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    label_text = {
        "multi_source": "none (episode D label)",
        "unlabeled_single": "0%",
        "supervised_single": "100%",
        "supervised_rcl_100": "100%",
        "lm_rcl_0": "0%",
        "lm_rcl_5": "5%",
        "lm_rcl_10": "10%",
        "lm_rcl_25": "25%",
        "sparse_supervised_rcl_5": "5%",
        "sparse_supervised_rcl_10": "10%",
        "sparse_supervised_rcl_25": "25%",
    }
    for method in METHODS:
        row = lookup[method, "overall"]
        lines.append(
            f"| {method} | {label_text[method]} | {100 * row['correct_commitment']:.1f}% | {row['hypothesis_brier']:.3f} | {row['hypothesis_ece']:.3f} | {row['hypothesis_nll']:.3f} | {row['regret']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Frozen development screen: **{claim['decision']}**.",
            f"LM-RCL 10% minus supervised RCL accuracy: {100 * claim['lm_10_accuracy_gap_vs_supervised']:+.2f} pp.",
            f"LM-RCL 10% minus sparse-supervised 10% accuracy: {100 * claim['lm_10_accuracy_gain_vs_sparse_supervised']:+.2f} pp.",
            f"Three-seed 95% intervals: versus supervised [{100 * claim['lm_10_accuracy_difference_ci_vs_supervised']['ci95_low']:+.2f}, {100 * claim['lm_10_accuracy_difference_ci_vs_supervised']['ci95_high']:+.2f}] pp; versus sparse-supervised [{100 * claim['lm_10_accuracy_difference_ci_vs_sparse_supervised']['ci95_low']:+.2f}, {100 * claim['lm_10_accuracy_difference_ci_vs_sparse_supervised']['ci95_high']:+.2f}] pp.",
            "",
            "This is a three-seed development screen, not formal confirmation or a noninferiority test. A GO result establishes only a candidate label-efficiency contribution; it still requires a frozen held-out replication and literature comparison. Mechanism AUROC/Brier are offline diagnostics and cannot substitute for hidden-state or commitment performance.",
        ]
    )
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    chinese = [
        "# 潜在机制 RCL 标签效率实验",
        "",
        "本实验使用全新开发种子，并冻结环境、RCL结构、承诺时刻和测试协议。LM-RCL始终获得训练样本的真实态势标签，但只在指定比例的可见 source-region 单元上获得可靠/受污染标签；其余机制标签对拟合过程不可见。",
        "",
        "| 方法 | 机制标签 | 正确承诺率 | Hypothesis Brier | Regret |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = lookup[method, "overall"]
        chinese.append(
            f"| {method} | {label_text[method]} | {100 * row['correct_commitment']:.1f}% | {row['hypothesis_brier']:.3f} | {row['regret']:.3f} |"
        )
    chinese.extend(
        [
            "",
            f"预注册开发筛选：**{claim['decision']}**。",
            f"LM-RCL 10% 相对全监督 RCL 为 {100 * claim['lm_10_accuracy_gap_vs_supervised']:+.2f} pp，三种子95%区间为 [{100 * claim['lm_10_accuracy_difference_ci_vs_supervised']['ci95_low']:+.2f}, {100 * claim['lm_10_accuracy_difference_ci_vs_supervised']['ci95_high']:+.2f}] pp。",
            f"LM-RCL 10% 相对同标签量稀疏监督 RCL 为 {100 * claim['lm_10_accuracy_gain_vs_sparse_supervised']:+.2f} pp，95%区间为 [{100 * claim['lm_10_accuracy_difference_ci_vs_sparse_supervised']['ci95_low']:+.2f}, {100 * claim['lm_10_accuracy_difference_ci_vs_sparse_supervised']['ci95_high']:+.2f}] pp。",
            "",
            "GO只表示点估计通过开发门槛，不是统计非劣结论。LM-RCL 10%的下界略低于预设的−2 pp边界；相对稀疏监督的区间也跨零。0% LM-RCL已经接近全监督，说明当前主要信号来自已知态势标签与潜变量生成结构。需要用冻结的新种子正式复现，才能把它提升为第三创新点。",
        ]
    )
    (destination / "中文结果说明.md").write_text("\n".join(chinese) + "\n")

    plot_script = """from pathlib import Path
import csv
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'summary.csv').open()))
lookup={(r['method'],r['scope']):r for r in rows}
x=[0,5,10,25,100]
lm=['lm_rcl_0','lm_rcl_5','lm_rcl_10','lm_rcl_25','supervised_rcl_100']
sparse=[None,'sparse_supervised_rcl_5','sparse_supervised_rcl_10','sparse_supervised_rcl_25',None]
fig,axes=plt.subplots(1,2,figsize=(9,3.8))
axes[0].plot(x,[100*float(lookup[m,'overall']['correct_commitment']) for m in lm],marker='o',label='LM-RCL')
axes[0].plot([5,10,25],[100*float(lookup[m,'overall']['correct_commitment']) for m in sparse if m],marker='s',label='Sparse supervised')
axes[0].axhline(100*float(lookup['multi_source','overall']['correct_commitment']),color='gray',ls='--',label='Multi-source')
axes[0].set(xlabel='Mechanism labels (%)',ylabel='Correct commitment (%)')
axes[1].plot(x,[float(lookup[m,'overall']['hypothesis_brier']) for m in lm],marker='o',label='LM-RCL')
axes[1].plot([5,10,25],[float(lookup[m,'overall']['hypothesis_brier']) for m in sparse if m],marker='s',label='Sparse supervised')
axes[1].set(xlabel='Mechanism labels (%)',ylabel='Hypothesis Brier')
for ax in axes: ax.grid(alpha=.25); ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(root/'label_efficiency.png',dpi=180)
fig.savefig(root/'label_efficiency.pdf')
"""
    (destination / "plot_results.py").write_text(plot_script)
    exec(compile(plot_script, str(destination / "plot_results.py"), "exec"), {"__file__": str(destination / "plot_results.py")})
    (destination / "result_manifest.json").write_text(
        json.dumps(directory_manifest(destination), indent=2)
    )
    print(json.dumps(claim, indent=2), flush=True)
    return claim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    run(parser.parse_args().output_dir)
