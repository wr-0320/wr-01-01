"""Frozen formal confirmation of mechanism-label-efficient LM-RCL."""

from __future__ import annotations

import argparse
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
from experiments.run_intelligence_latent_mechanism import (
    _diagnostics,
    _nested_label_masks,
    _sparse_supervised_samples,
)
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
from experiments.run_intelligence_ride_v2_headroom import directory_manifest, write_csv
from experiments.run_intelligence_ride_v2_reliability import (
    RELIABILITY_TRAIN_FAMILIES,
    _inference,
    collect_reliability_samples,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "intelligence_latent_mechanism_formal"
SEEDS = tuple(range(8101, 8111))
METHODS = (
    "multi_source",
    "unlabeled_single",
    "supervised_rcl_100",
    "lm_rcl_0",
    "lm_rcl_10",
    "lm_rcl_25",
    "sparse_supervised_rcl_10",
)
T95_DF9 = 2.262


def _interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    if len(values) != 10:
        return mean, float("nan"), float("nan")
    half = T95_DF9 * float(values.std(ddof=1)) / math.sqrt(len(values))
    return mean, mean - half, mean + half


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
            result = {
                "method": method,
                "scope": scope,
                "seeds": len(seeds),
                "episodes": sum(lookup[seed, method, scope]["episodes"] for seed in seeds),
            }
            for metric in METRICS:
                values = np.asarray([lookup[seed, method, scope][metric] for seed in seeds])
                mean, low, high = _interval(values)
                result[metric] = mean
                result[metric + "_ci95_low"] = low
                result[metric + "_ci95_high"] = high
            summary.append(result)
    paired = []
    contrasts = (
        ("lm_rcl_0", "supervised_rcl_100"),
        ("lm_rcl_10", "sparse_supervised_rcl_10"),
        ("lm_rcl_0", "unlabeled_single"),
        ("supervised_rcl_100", "multi_source"),
    )
    for method, comparator in contrasts:
        for scope in scopes:
            for metric in METRICS:
                values = np.asarray(
                    [
                        lookup[seed, method, scope][metric]
                        - lookup[seed, comparator, scope][metric]
                        for seed in seeds
                    ]
                )
                mean, low, high = _interval(values)
                paired.append(
                    {
                        "method": method,
                        "comparator": comparator,
                        "scope": scope,
                        "metric": metric,
                        "difference": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return per_seed, summary, paired


def run(
    output_dir: Path = OUTPUT,
    *,
    seeds: tuple[int, ...] = SEEDS,
    episodes: int = 100,
    smoke: bool = False,
):
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError("Refusing to overwrite outputs")
    if not smoke and (tuple(seeds) != SEEDS or episodes != 100):
        raise ValueError("Formal protocol is frozen")
    frozen = {
        path.name: path
        for path in (ROOT / "outputs").iterdir()
        if path.is_dir() and (path.name.startswith("intelligence_") or path.name.startswith("thesis_"))
    }
    before = {name: directory_manifest(path) for name, path in frozen.items()}
    sources = source_manifest()
    destination.mkdir(parents=True)
    protocol = {
        "protocol": "LM-RCL Frozen Formal Confirmation v1",
        "seeds": list(seeds),
        "families": list(RED_FAMILIES),
        "methods": list(METHODS),
        "episodes_per_method_family_seed": episodes,
        "expected_test_episodes": len(seeds) * len(RED_FAMILIES) * len(METHODS) * episodes,
        "fit_episodes_per_hypothesis_family": 10 if not smoke else 1,
        "temperature_validation_cases_per_seen_family": 12 if not smoke else 1,
        "neural_train_episodes_per_hypothesis_family": 8 if not smoke else 1,
        "neural_validation_episodes_per_hypothesis_family": 3 if not smoke else 1,
        "neural_epochs": 8 if not smoke else 1,
        "primary_estimand": "paired seed mean correct-commitment difference LM-RCL 0% minus supervised RCL 100%",
        "primary_noninferiority_margin": -0.02,
        "primary_interval": "two-sided 95% t interval over ten paired training-seed means, df=9",
        "primary_success": "lower 95% bound > -0.02",
        "supportive_guards": "Brier difference upper 95% bound < +0.03; regret difference upper 95% bound < +0.50; equal latency and information cost",
        "key_secondary": "LM-RCL 10% minus sparse-supervised RCL 10%; superiority requires accuracy lower 95% bound > 0",
        "family_stability_guard": "no family point-estimate LM-RCL 0% loss versus supervised RCL worse than -5 pp",
        "multiple_testing": "primary tested first; secondary descriptive unless primary succeeds; family analyses descriptive",
        "label_unit": "visible source-region report cell",
        "label_selection": "nested random selection independent of mechanism value",
        "all_methods_receive_hypothesis_labels": True,
        "unlabeled_mechanism_truth_hidden_from_fit": True,
        "algorithm_frozen_after_development": True,
        "formal_results_replace_development": False,
        "smoke_only": smoke,
    }
    (destination / "preregistered_protocol.json").write_text(json.dumps(protocol, indent=2))
    (destination / "source_manifest.json").write_text(json.dumps(sources, indent=2))

    all_rows = []
    fit_diagnostics = []
    for seed in seeds:
        train = collect_prefix_records(
            TRAIN_FAMILIES,
            8 if not smoke else 1,
            seed=6_000_000_000 + seed * 10003,
            split=f"lm_formal_neural_train_{seed}",
            max_cycles=10,
        )
        validation = collect_prefix_records(
            TRAIN_FAMILIES,
            3 if not smoke else 1,
            seed=6_050_000_000 + seed * 10003,
            split=f"lm_formal_neural_validation_{seed}",
            max_cycles=10,
        )
        neural, _ = train_model(
            train,
            validation,
            epochs=8 if not smoke else 1,
            hidden_size=32,
            batch_size=64,
            seed=seed,
        )
        reasoner = HonestIntelligenceReasoner(train)
        samples = collect_reliability_samples(
            RELIABILITY_TRAIN_FAMILIES,
            10 if not smoke else 1,
            seed=6_100_000_000 + seed * 10003,
            reasoner=reasoner,
            split=f"lm_formal_fit_{seed}",
            max_cycles=10,
        )
        masks = _nested_label_masks(samples, (0.0, 0.10, 0.25), 6_150_000_000 + seed)
        calibration = collect_belief_cases(
            RELIABILITY_TRAIN_FAMILIES,
            12 if not smoke else 1,
            seed=6_200_000_000 + seed * 10003,
            max_cycles=10,
        )
        diagnostic_samples = collect_reliability_samples(
            RELIABILITY_TRAIN_FAMILIES,
            2 if not smoke else 1,
            seed=6_250_000_000 + seed * 10003,
            reasoner=reasoner,
            split=f"lm_formal_mechanism_diagnostic_{seed}",
            max_cycles=10,
        )

        full_fixed = FixedCellReliability.fit(samples)
        full_raw = ReliabilityConditionedLikelihood.fit(samples)
        full = fit_likelihood_temperature(full_raw, calibration, full_fixed, reasoner)
        all_fixed = FixedCellReliability(np.ones_like(full_fixed.reliable))
        unlabeled_raw = fit_unlabeled_single_mechanism(samples)
        unlabeled_raw = SingleReliableLikelihood(
            unlabeled_raw.reliable_mean,
            unlabeled_raw.reliable_scale,
            unlabeled_raw.compromised_mean,
            unlabeled_raw.compromised_scale,
        )
        unlabeled = fit_likelihood_temperature(unlabeled_raw, calibration, all_fixed, reasoner)
        models: dict[str, tuple[Any, FixedCellReliability]] = {
            "unlabeled_single": (unlabeled, all_fixed),
            "supervised_rcl_100": (full, full_fixed),
        }
        parameters: dict[str, Any] = {
            "supervised_rcl_100": {
                "reliability": full_fixed.reliable.tolist(),
                "temperature": full.temperature,
            }
        }
        for ratio in (0.0, 0.10, 0.25):
            result = fit_latent_mechanism_rcl(masks[ratio])
            fitted = fit_likelihood_temperature(
                result.likelihood, calibration, result.reliability, reasoner
            )
            name = f"lm_rcl_{int(100 * ratio)}"
            models[name] = (fitted, result.reliability)
            probability, truth = mechanism_responsibilities(result, diagnostic_samples)
            fit_diagnostics.append(
                {
                    "seed": seed,
                    "method": name,
                    "requested_label_fraction": ratio,
                    "actual_label_fraction": result.labeled_fraction,
                    "iterations": result.iterations,
                    "converged": int(result.converged),
                    "temperature": fitted.temperature,
                    "diagnostic_split": "fresh offline fit-family episodes; never used for fitting or calibration",
                    **_diagnostics(probability, truth),
                }
            )
            parameters[name] = {
                "reliability": result.reliability.reliable.tolist(),
                "temperature": fitted.temperature,
                "iterations": result.iterations,
                "converged": result.converged,
            }
        sparse_samples = _sparse_supervised_samples(masks[0.10])
        sparse_fixed = FixedCellReliability.fit(sparse_samples)
        sparse_raw = ReliabilityConditionedLikelihood.fit(sparse_samples)
        sparse = fit_likelihood_temperature(sparse_raw, calibration, sparse_fixed, reasoner)
        models["sparse_supervised_rcl_10"] = (sparse, sparse_fixed)

        controllers = {
            "multi_source": _inference(
                neural, reasoner, None, "multi_source", full_ride=False, max_cycles=10
            )
        }
        controllers.update(
            {
                name: ReliabilityLikelihoodController(model, predictor, reasoner, name=name)
                for name, (model, predictor) in models.items()
            }
        )
        if set(controllers) != set(METHODS):
            raise AssertionError("formal method drift")
        (destination / f"parameters_seed{seed}.json").write_text(
            json.dumps(parameters, indent=2)
        )
        seed_rows = []
        for family_index, family in enumerate(RED_FAMILIES):
            for episode in range(episodes):
                hypothesis = (episode + family_index + seed) % NUM_HYPOTHESES
                scenario_seed = 6_300_000_000 + seed * 100003 + family_index * 1009 + episode
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
        raise AssertionError("formal episode count drift")
    per_seed, summary, paired = _analyze(all_rows)
    write_csv(destination / "episodes.csv", all_rows)
    write_csv(destination / "per_seed_summary.csv", per_seed)
    write_csv(destination / "summary.csv", summary)
    write_csv(destination / "paired_differences.csv", paired)
    write_csv(destination / "heldout_mechanism_diagnostics.csv", fit_diagnostics)

    summary_lookup = {(row["method"], row["scope"]): row for row in summary}
    diff = {
        (row["method"], row["comparator"], row["scope"], row["metric"]): row
        for row in paired
    }
    primary_accuracy = diff[
        "lm_rcl_0", "supervised_rcl_100", "overall", "correct_commitment"
    ]
    primary_brier = diff[
        "lm_rcl_0", "supervised_rcl_100", "overall", "hypothesis_brier"
    ]
    primary_regret = diff[
        "lm_rcl_0", "supervised_rcl_100", "overall", "regret"
    ]
    secondary = diff[
        "lm_rcl_10", "sparse_supervised_rcl_10", "overall", "correct_commitment"
    ]
    family_differences = {
        family: diff[
            "lm_rcl_0", "supervised_rcl_100", family, "correct_commitment"
        ]["difference"]
        for family in RED_FAMILIES
    }
    equal_effort = all(
        abs(
            summary_lookup["lm_rcl_0", "overall"][metric]
            - summary_lookup["supervised_rcl_100", "overall"][metric]
        )
        < 1e-12
        for metric in ("decision_latency", "information_cost")
    )
    supported = bool(
        primary_accuracy["ci95_low"] > -0.02
        and primary_brier["ci95_high"] < 0.03
        and primary_regret["ci95_high"] < 0.50
        and equal_effort
        and min(family_differences.values()) > -0.05
    )
    claim = {
        "formal_primary_supported": supported,
        "primary_noninferiority_margin": -0.02,
        "lm_0_vs_supervised_accuracy": primary_accuracy,
        "lm_0_vs_supervised_brier": primary_brier,
        "lm_0_vs_supervised_regret": primary_regret,
        "lm_10_vs_sparse_10_accuracy": secondary,
        "key_secondary_superiority_supported": bool(secondary["ci95_low"] > 0),
        "family_accuracy_differences": family_differences,
        "equal_latency_and_information_cost": equal_effort,
        "algorithm_novelty_established_by_experiment": False,
        "older_formal_results_replaced": False,
    }
    (destination / "claim_audit.json").write_text(json.dumps(claim, indent=2))
    audit = {name + "_unchanged": before[name] == directory_manifest(path) for name, path in frozen.items()}
    audit.update(
        {
            "source_unchanged": sources == source_manifest(),
            "algorithm_frozen_after_development": True,
            "unlabeled_mechanism_truth_read_by_fit": False,
            "runtime_mechanism_truth_used": False,
            "runtime_hypothesis_truth_used": False,
            "heldout_mechanism_labels_diagnostic_only": True,
            "paired_scenarios_across_methods": True,
            "test_episodes": len(all_rows),
            "policy_inputs_audited": sum(int(row["boundary_checks"]) for row in all_rows),
        }
    )
    if not all(value for key, value in audit.items() if key.endswith("_unchanged")):
        raise AssertionError("frozen artifact changed")
    (destination / "information_boundary_audit.json").write_text(json.dumps(audit, indent=2))

    label = {
        "multi_source": "no cell labels",
        "unlabeled_single": "0%",
        "supervised_rcl_100": "100%",
        "lm_rcl_0": "0%",
        "lm_rcl_10": "10%",
        "lm_rcl_25": "25%",
        "sparse_supervised_rcl_10": "10%",
    }
    lines = [
        "# LM-RCL frozen formal confirmation",
        "",
        "| Method | Mechanism labels | Accuracy | Brier | ECE | NLL | Regret | Latency | Cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary_lookup[method, "overall"]
        lines.append(
            f"| {method} | {label[method]} | {100 * row['correct_commitment']:.1f}% | {row['hypothesis_brier']:.3f} | {row['hypothesis_ece']:.3f} | {row['hypothesis_nll']:.3f} | {row['regret']:.3f} | {row['decision_latency']:.2f} | {row['information_cost']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Primary formal result: **{'SUPPORTED' if supported else 'NOT SUPPORTED'}**.",
            f"LM-RCL 0% minus supervised RCL accuracy: {100 * primary_accuracy['difference']:+.2f} pp, two-sided 95% paired-seed CI [{100 * primary_accuracy['ci95_low']:+.2f}, {100 * primary_accuracy['ci95_high']:+.2f}] pp; preregistered noninferiority margin −2.00 pp.",
            f"Key secondary LM-RCL 10% minus sparse-supervised 10%: {100 * secondary['difference']:+.2f} pp, 95% CI [{100 * secondary['ci95_low']:+.2f}, {100 * secondary['ci95_high']:+.2f}] pp.",
            "",
            "The primary test concerns loss of task performance after removing cell-level mechanism labels. It does not show that hypothesis labels are unnecessary: all likelihood methods receive the true operational-hypothesis label during offline training. Multi-source retains its original episode-level deception supervision and is contextual rather than an equal-supervision control. Held-out mechanism-label metrics are diagnostic only. Novelty still requires literature comparison.",
        ]
    )
    (destination / "formal_report.md").write_text("\n".join(lines) + "\n")
    chinese = [
        "# LM-RCL 正式确认结果",
        "",
        f"主非劣检验：**{'通过' if supported else '未通过'}**。",
        "",
        *lines[2:11],
        "",
        f"LM-RCL 0% 相对全监督 RCL：{100 * primary_accuracy['difference']:+.2f} pp，双侧95%配对种子区间 [{100 * primary_accuracy['ci95_low']:+.2f}, {100 * primary_accuracy['ci95_high']:+.2f}] pp；预注册非劣界限为 −2.00 pp。",
        f"LM-RCL 10% 相对稀疏监督10%：{100 * secondary['difference']:+.2f} pp，95%区间 [{100 * secondary['ci95_low']:+.2f}, {100 * secondary['ci95_high']:+.2f}] pp。",
        "",
        "该检验证明或否定的是逐条机制标签能否被潜变量学习替代，不代表真实态势标签也可以删除。所有似然模型训练时仍使用真实态势标签。机制识别指标来自独立诊断集，仅作为解释性证据。",
    ]
    (destination / "中文正式结果.md").write_text("\n".join(chinese) + "\n")
    (destination / "result_manifest.json").write_text(
        json.dumps(directory_manifest(destination), indent=2)
    )
    print(json.dumps(claim, indent=2), flush=True)
    return claim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    run(parser.parse_args().output_dir)
