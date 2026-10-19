"""Frozen development scan of controlled source-region heterogeneity."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from deception_rl.envs.controlled_heterogeneity import (
    HETEROGENEITY_PROFILES,
    ControlledHeterogeneityEnv,
)
from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES
from deception_rl.intelligence.adapter import audit_blue_observation
from deception_rl.intelligence.ride_v2.granularity_baselines import (
    GroupedFixedReliability,
)
from deception_rl.intelligence.ride_v2.reliability_conditioned import (
    ReliabilityConditionedLikelihood,
    ReliabilityLikelihoodController,
)
from experiments.run_intelligence_reliability_conditioned_inference import (
    fit_likelihood_temperature,
)
from experiments.run_intelligence_ride import RED_FAMILIES, _collection_action
from experiments.run_intelligence_ride_v2_headroom import DEVELOPMENT_RED_FAMILIES
from experiments.run_intelligence_ride_v2_reliability import RELIABILITY_TRAIN_FAMILIES
from experiments.run_rcl_granularity_development import (
    METRICS,
    _interval,
    summarize_group,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "development" / "rcl_heterogeneity_v1"
PROFILES = tuple(HETEROGENEITY_PROFILES)
METHODS = ("global_mixture", "source_mixture", "source_region_rcl")
SEEDS = (9201, 9202, 9203)


def collect_samples(
    profile: str,
    families: tuple[str, ...],
    episodes_per_hypothesis: int,
    *,
    seed: int,
    max_cycles: int,
) -> list[dict[str, Any]]:
    rows = []
    for family_index, family in enumerate(families):
        for hypothesis in range(NUM_HYPOTHESES):
            for episode in range(episodes_per_hypothesis):
                episode_seed = seed + family_index * 100_003 + hypothesis * 1_009 + episode
                env = ControlledHeterogeneityEnv(
                    heterogeneity=profile, max_cycles=max_cycles
                )
                red = DEVELOPMENT_RED_FAMILIES[family]()
                red.reset()
                observations, _ = env.reset(
                    seed=episode_seed, options={"hypothesis": hypothesis}
                )
                while env.agents:
                    step = int(observations["blue"]["current_step"][0])
                    red_action = red.act(observations["red"])
                    blue_action = _collection_action(observations["blue"], step)
                    observations, *_ = env.step(
                        {"red": red_action, "blue": blue_action}
                    )
                    rows.append(
                        {
                            "profile": profile,
                            "family": family,
                            "hypothesis_label": hypothesis,
                            "reported_cells": np.asarray(
                                observations["blue"]["report_mask"], dtype=bool
                            )[step],
                            "reported_values": np.asarray(
                                observations["blue"]["reports"], dtype=float
                            )[step],
                            "compromised_cells": env.compromised_history[step],
                        }
                    )
    return rows


def collect_calibration_cases(
    profile: str,
    families: tuple[str, ...],
    episodes_per_family: int,
    *,
    seed: int,
    max_cycles: int,
    target_step: int = 4,
) -> list[dict[str, Any]]:
    cases = []
    for family_index, family in enumerate(families):
        for episode in range(episodes_per_family):
            hypothesis = (episode + family_index) % NUM_HYPOTHESES
            episode_seed = seed + family_index * 10_003 + episode
            env = ControlledHeterogeneityEnv(
                heterogeneity=profile, max_cycles=max_cycles
            )
            red = DEVELOPMENT_RED_FAMILIES[family]()
            red.reset()
            observations, _ = env.reset(
                seed=episode_seed, options={"hypothesis": hypothesis}
            )
            while env.agents and int(observations["blue"]["current_step"][0]) < target_step:
                step = int(observations["blue"]["current_step"][0])
                observations, *_ = env.step(
                    {
                        "red": red.act(observations["red"]),
                        "blue": _collection_action(observations["blue"], step),
                    }
                )
            cases.append(
                {
                    "profile": profile,
                    "hypothesis": hypothesis,
                    "seed": episode_seed,
                    "observation": observations["blue"],
                }
            )
    return cases


def evaluate_episode(controller, profile, family, hypothesis, *, seed, max_cycles):
    env = ControlledHeterogeneityEnv(heterogeneity=profile, max_cycles=max_cycles)
    red = RED_FAMILIES[family]()
    red.reset()
    controller.reset()
    observations, _ = env.reset(seed=seed, options={"hypothesis": hypothesis})
    metrics = None
    while env.agents:
        audit_blue_observation(observations["blue"])
        blue_action = controller.act(observations["blue"])
        observations, _, _, _, infos = env.step(
            {"red": red.act(observations["red"]), "blue": blue_action}
        )
        if "episode_metrics" in infos["blue"]:
            metrics = infos["blue"]["episode_metrics"]
    if metrics is None or controller.info is None:
        raise RuntimeError("heterogeneity episode ended without metrics")
    belief = np.asarray(controller.info["hypothesis_belief"], dtype=float)
    target = np.eye(NUM_HYPOTHESES)[hypothesis]
    true_probability = float(np.clip(belief[hypothesis], 1e-8, 1.0))
    row = {
        "controller": controller.name,
        "profile": profile,
        "deception_family": family,
        "hypothesis": hypothesis,
        "seed": seed,
        "interception": metrics["decision_correct"],
        "confidence": float(controller.info["confidence"]),
        "hypothesis_brier": float(np.sum((belief - target) ** 2)),
        "hypothesis_nll": float(-np.log(true_probability)),
        "regret": metrics["decision_regret"],
        "decision_latency": metrics["decision_step"],
        "information_cost": metrics["blue_information_cost"],
    }
    mask = env._report_mask.astype(bool)  # noqa: SLF001 - isolated audit output
    compromised = env.compromised_history.astype(bool)
    return row, mask.sum(axis=0), (mask & compromised).sum(axis=0)


def reliability_audit_rows(profile, seed, split, samples=None, counts=None, compromised=None):
    if samples is not None:
        counts = np.zeros((3, 3), dtype=int)
        compromised = np.zeros((3, 3), dtype=int)
        for row in samples:
            visible = np.asarray(row["reported_cells"], dtype=bool)
            counts += visible
            compromised += visible * np.asarray(row["compromised_cells"], dtype=int)
    counts = np.asarray(counts, dtype=int)
    compromised = np.asarray(compromised, dtype=int)
    rate = np.divide(
        compromised,
        counts,
        out=np.zeros_like(compromised, dtype=float),
        where=counts > 0,
    )
    configured = HETEROGENEITY_PROFILES[profile]
    rows = []
    for source in range(3):
        for region in range(3):
            rows.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "split": split,
                    "source": source,
                    "region": region,
                    "configured_activation": configured[source, region],
                    "configured_activation_mean": float(configured.mean()),
                    "configured_activation_std": float(configured.std()),
                    "configured_activation_range": float(configured.max() - configured.min()),
                    "visible_reports": int(counts[source, region]),
                    "compromised_reports": int(compromised[source, region]),
                    "empirical_contamination_rate": float(rate[source, region]),
                    "empirical_reliability": float(1.0 - rate[source, region]),
                    "overall_contamination_prevalence": float(
                        compromised.sum() / max(1, counts.sum())
                    ),
                    "empirical_cell_rate_std": float(rate.std()),
                    "empirical_cell_rate_range": float(rate.max() - rate.min()),
                }
            )
    return rows


def analyze(rows, seeds):
    per_seed = []
    for seed in seeds:
        for profile in PROFILES:
            for method in METHODS:
                selected = [
                    row
                    for row in rows
                    if row["training_seed"] == seed
                    and row["profile"] == profile
                    and row["controller"] == method
                ]
                per_seed.append(
                    {
                        "seed": seed,
                        "profile": profile,
                        "method": method,
                        "episodes": len(selected),
                        **summarize_group(selected),
                    }
                )
    summary = []
    for profile in PROFILES:
        for method in METHODS:
            selected = [
                row for row in per_seed if row["profile"] == profile and row["method"] == method
            ]
            result = {
                "profile": profile,
                "method": method,
                "seeds": len(selected),
                "episodes": sum(row["episodes"] for row in selected),
            }
            for metric in METRICS:
                values = np.asarray([row[metric] for row in selected], dtype=float)
                low, high = _interval(values)
                result[f"{metric}_mean"] = float(values.mean())
                result[f"{metric}_ci95_low"] = low
                result[f"{metric}_ci95_high"] = high
            summary.append(result)

    lookup = {(row["seed"], row["profile"], row["method"]): row for row in per_seed}
    paired = []
    comparisons = (
        ("source_region_rcl", "global_mixture"),
        ("source_region_rcl", "source_mixture"),
        ("source_mixture", "global_mixture"),
    )
    for profile in PROFILES:
        for reference, comparator in comparisons:
            for metric in ("correct_commitment", "hypothesis_brier", "hypothesis_nll", "hypothesis_ece"):
                differences = np.asarray(
                    [
                        lookup[(seed, profile, reference)][metric]
                        - lookup[(seed, profile, comparator)][metric]
                        for seed in seeds
                    ]
                )
                low, high = _interval(differences)
                paired.append(
                    {
                        "profile": profile,
                        "reference": reference,
                        "comparator": comparator,
                        "metric": metric,
                        "difference": float(differences.mean()),
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return per_seed, summary, paired


def decide(summary, paired):
    differences = {
        (row["profile"], row["reference"], row["comparator"], row["metric"]): row["difference"]
        for row in paired
    }

    def difference(profile, reference, comparator, metric):
        return differences[(profile, reference, comparator, metric)]

    high_levels = ("high", "extreme")
    comparators = ("global_mixture", "source_mixture")
    accuracy_positive = all(
        difference(level, "source_region_rcl", comparator, "correct_commitment") > 0
        for level in high_levels
        for comparator in comparators
    )
    practical_accuracy = all(
        max(
            difference(level, "source_region_rcl", comparator, "correct_commitment")
            for level in high_levels
        )
        >= 0.02
        for comparator in comparators
    )
    probability_guard = all(
        difference(level, "source_region_rcl", comparator, metric) <= 0.005
        for level in high_levels
        for comparator in comparators
        for metric in ("hypothesis_brier", "hypothesis_nll")
    )
    source_trigger = (
        difference("medium", "source_mixture", "global_mixture", "correct_commitment")
        >= 0.02
    )
    probability_focus = all(
        difference(level, "source_region_rcl", comparator, metric) < 0
        for level in high_levels
        for comparator in comparators
        for metric in ("hypothesis_brier", "hypothesis_nll")
    )
    if accuracy_positive and practical_accuracy and probability_guard:
        decision = "retain_source_region_as_core_candidate"
    elif source_trigger:
        decision = "prefer_source_level_candidate"
    elif probability_focus:
        decision = "focus_claim_on_probability_quality"
    else:
        decision = "global_mixture_main_source_region_optional"

    accuracy = {
        (row["profile"], row["method"]): row["correct_commitment_mean"]
        for row in summary
    }
    all_degrade = all(
        accuracy[("extreme", method)] <= accuracy[("low", method)] - 0.05
        for method in METHODS
    )
    return {
        "decision": decision,
        "rule1_accuracy_positive_high_and_extreme": accuracy_positive,
        "rule1_at_least_two_point_gain": practical_accuracy,
        "rule1_probability_guard": probability_guard,
        "rule2_source_medium_gain_at_least_two_points": source_trigger,
        "rule3_probability_quality_only": probability_focus,
        "all_methods_degrade_at_least_five_points": all_degrade,
        "formal_claim_established": False,
    }


def write_report(destination, summary, paired, decision):
    lookup = {(row["profile"], row["method"]): row for row in summary}
    diff = {
        (row["profile"], row["reference"], row["comparator"], row["metric"]): row
        for row in paired
    }
    lines = [
        "# Controlled heterogeneity development v1",
        "",
        "This is three-seed development evidence, not a formal confirmation.",
        "All profiles preserve a configured mean manipulation activation of 0.60.",
        "",
        "| Profile | Method | Accuracy | Brier | NLL | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        for method in METHODS:
            row = lookup[(profile, method)]
            lines.append(
                f"| {profile} | {method} | {100*row['correct_commitment_mean']:.2f}% | "
                f"{row['hypothesis_brier_mean']:.4f} | {row['hypothesis_nll_mean']:.4f} | "
                f"{row['hypothesis_ece_mean']:.4f} |"
            )
    lines.extend(["", "## Source-region paired differences", "", "| Profile | Comparator | Accuracy pp | Brier | NLL |", "|---|---|---:|---:|---:|"])
    for profile in PROFILES:
        for comparator in ("global_mixture", "source_mixture"):
            accuracy = diff[(profile, "source_region_rcl", comparator, "correct_commitment")]
            brier = diff[(profile, "source_region_rcl", comparator, "hypothesis_brier")]
            nll = diff[(profile, "source_region_rcl", comparator, "hypothesis_nll")]
            lines.append(
                f"| {profile} | {comparator} | {100*accuracy['difference']:+.2f} "
                f"[{100*accuracy['ci95_low']:+.2f},{100*accuracy['ci95_high']:+.2f}] | "
                f"{brier['difference']:+.4f} | {nll['difference']:+.4f} |"
            )
    lines.extend(
        [
            "",
            "## Predeclared development decision",
            "",
            f"Decision: **{decision['decision']}**.",
            "",
            "This decision follows `docs/heterogeneity_scan_protocol.md`. An interval crossing zero is not interpreted as equivalence. Existing correlation-aware and intent-coupled NO-GO results remain unchanged.",
        ]
    )
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    output_dir=DEFAULT_OUTPUT,
    *,
    seeds=SEEDS,
    episodes=100,
    likelihood_episodes=10,
    calibration_episodes=3,
    max_cycles=10,
    smoke=False,
):
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError("refusing to overwrite an existing output directory")
    destination.mkdir(parents=True)
    protocol = {
        "protocol": "Controlled Source-Region Heterogeneity Development v1",
        "status": "smoke" if smoke else "development_not_formal",
        "profiles": {name: value.tolist() for name, value in HETEROGENEITY_PROFILES.items()},
        "configured_activation_mean": 0.60,
        "seeds": list(seeds),
        "families": list(RED_FAMILIES),
        "methods": list(METHODS),
        "episodes_per_method_family_profile_seed": episodes,
        "expected_test_episodes": len(seeds) * len(PROFILES) * len(METHODS) * len(RED_FAMILIES) * episodes,
        "paired_scenarios_across_methods_and_profiles": True,
        "likelihood_episodes_per_hypothesis_family": likelihood_episodes,
        "calibration_cases_per_seen_family": calibration_episodes,
        "fixed_commit_step": 4,
        "interval_unit": "training seed",
        "decision_rules": "docs/heterogeneity_scan_protocol.md",
        "formal_claim_allowed": False,
    }
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    all_rows = []
    audit_rows = []
    for seed in seeds:
        for profile in PROFILES:
            samples = collect_samples(
                profile,
                RELIABILITY_TRAIN_FAMILIES,
                likelihood_episodes,
                seed=700_000_000 + seed * 10_003,
                max_cycles=max_cycles,
            )
            audit_rows.extend(reliability_audit_rows(profile, seed, "fit", samples=samples))
            emissions = ReliabilityConditionedLikelihood.fit(samples)
            cases = collect_calibration_cases(
                profile,
                RELIABILITY_TRAIN_FAMILIES,
                calibration_episodes,
                seed=750_000_000 + seed * 10_003,
                max_cycles=max_cycles,
            )
            predictors = {
                granularity: GroupedFixedReliability.fit(samples, granularity=granularity)
                for granularity in ("global", "source", "source_region")
            }
            likelihoods = {
                granularity: fit_likelihood_temperature(emissions, cases, predictor, None)
                for granularity, predictor in predictors.items()
            }
            controllers = {
                "global_mixture": ReliabilityLikelihoodController(
                    likelihoods["global"], predictors["global"], None, name="global_mixture"
                ),
                "source_mixture": ReliabilityLikelihoodController(
                    likelihoods["source"], predictors["source"], None, name="source_mixture"
                ),
                "source_region_rcl": ReliabilityLikelihoodController(
                    likelihoods["source_region"], predictors["source_region"], None, name="source_region_rcl"
                ),
            }
            parameter_payload = {
                method: {
                    "reliability": controller.predictor.reliable.tolist(),
                    "temperature": controller.likelihood.temperature,
                }
                for method, controller in controllers.items()
            }
            (destination / f"parameters_{profile}_seed{seed}.json").write_text(
                json.dumps(parameter_payload, indent=2), encoding="utf-8"
            )

            test_counts = np.zeros((3, 3), dtype=int)
            test_compromised = np.zeros((3, 3), dtype=int)
            seed_profile_rows = []
            for family_index, family in enumerate(RED_FAMILIES):
                for episode in range(episodes):
                    hypothesis = (episode + seed + family_index) % NUM_HYPOTHESES
                    scenario_seed = 800_000_000 + seed * 100_003 + family_index * 1_009 + episode
                    for method, controller in controllers.items():
                        row, counts, compromised = evaluate_episode(
                            controller,
                            profile,
                            family,
                            hypothesis,
                            seed=scenario_seed,
                            max_cycles=max_cycles,
                        )
                        row["training_seed"] = seed
                        seed_profile_rows.append(row)
                        if method == METHODS[0]:
                            test_counts += counts
                            test_compromised += compromised
            write_csv(destination / f"episodes_{profile}_seed{seed}.csv", seed_profile_rows)
            all_rows.extend(seed_profile_rows)
            audit_rows.extend(
                reliability_audit_rows(
                    profile,
                    seed,
                    "test",
                    counts=test_counts,
                    compromised=test_compromised,
                )
            )

    if len(all_rows) != protocol["expected_test_episodes"]:
        raise AssertionError("test episode count drift")
    per_seed, summary, paired = analyze(all_rows, tuple(seeds))
    decision = decide(summary, paired)
    write_csv(destination / "episodes.csv", all_rows)
    write_csv(destination / "per_seed_summary.csv", per_seed)
    write_csv(destination / "summary.csv", summary)
    write_csv(destination / "paired_differences.csv", paired)
    write_csv(destination / "heterogeneity_audit.csv", audit_rows)
    (destination / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_report(destination, summary, paired, decision)
    return {"episodes": len(all_rows), "decision": decision["decision"], "smoke": smoke}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        result = run(
            args.output_dir,
            seeds=(9291,),
            episodes=1,
            likelihood_episodes=1,
            calibration_episodes=1,
            smoke=True,
        )
    else:
        result = run(args.output_dir)
    print(json.dumps(result, indent=2))
