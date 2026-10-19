"""Matched development comparison of the dual mixture and robust baselines."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES
from deception_rl.intelligence.ride import HonestIntelligenceReasoner
from deception_rl.intelligence.ride_v2.granularity_baselines import GroupedFixedReliability
from deception_rl.intelligence.ride_v2.reliability_conditioned import (
    ReliabilityConditionedLikelihood,
    ReliabilityLikelihoodController,
)
from deception_rl.intelligence.ride_v2.robust_likelihood_baselines import (
    HuberReliableLikelihood,
    ReliabilityWeightedGaussianLikelihood,
    StudentTReliableLikelihood,
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
)
from experiments.run_intelligence_ride_v2_reliability import (
    RELIABILITY_TRAIN_FAMILIES,
    collect_reliability_samples,
)
from experiments.run_rcl_granularity_development import (
    METRICS,
    SingleReliableLikelihood,
    _interval,
    summarize_group,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "development" / "rcl_mechanism_baselines_v2"
METHODS = (
    "single_mechanism",
    "global_dual_mixture",
    "student_t",
    "huber",
    "reliability_weighted",
)
SEEDS = (9101, 9102, 9103)
STUDENT_DOF_GRID = (3.0, 5.0, 10.0, 30.0)
HUBER_DELTA_GRID = (1.0, 1.345, 2.0, 3.0)
RELIABILITY_WEIGHT_POWER = 1.0


def validation_nll(likelihood, cases, predictor, reasoner) -> float:
    logits = np.stack(
        [likelihood.logits(row["observation"], predictor, reasoner) for row in cases]
    )
    labels = np.asarray([row["hypothesis"] for row in cases], dtype=int)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probability = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return -float(log_probability[np.arange(len(labels)), labels].mean())


def select_and_calibrate(candidates, cases, predictor, reasoner):
    records = []
    best = None
    for label, candidate in candidates:
        calibrated = fit_likelihood_temperature(candidate, cases, predictor, reasoner)
        score = validation_nll(calibrated, cases, predictor, reasoner)
        records.append(
            {
                "candidate": label,
                "temperature": calibrated.temperature,
                "validation_nll": score,
            }
        )
        if best is None or score < best[0]:
            best = (score, label, calibrated)
    if best is None:
        raise ValueError("candidate grid must be nonempty")
    return best[2], best[1], records


def analyze(rows: list[dict[str, Any]], seeds: tuple[int, ...]):
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (int(row["training_seed"]), row["controller"], row["deception_family"])
        ].append(row)

    per_seed = []
    for seed in seeds:
        for method in METHODS:
            overall = [
                row
                for row in rows
                if int(row["training_seed"]) == seed and row["controller"] == method
            ]
            for scope, selected in (
                ("overall", overall),
                *((family, grouped[(seed, method, family)]) for family in RED_FAMILIES),
            ):
                per_seed.append(
                    {
                        "seed": seed,
                        "method": method,
                        "scope": scope,
                        "episodes": len(selected),
                        **summarize_group(selected),
                    }
                )

    summary = []
    for method in METHODS:
        for scope in ("overall", *RED_FAMILIES):
            selected = [
                row
                for row in per_seed
                if row["method"] == method and row["scope"] == scope
            ]
            result = {
                "method": method,
                "scope": scope,
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

    lookup = {
        (row["seed"], row["method"]): row
        for row in per_seed
        if row["scope"] == "overall"
    }
    paired = []
    for comparator in METHODS:
        if comparator == "global_dual_mixture":
            continue
        for metric in (
            "correct_commitment",
            "hypothesis_brier",
            "hypothesis_nll",
            "hypothesis_ece",
        ):
            differences = np.asarray(
                [
                    lookup[(seed, "global_dual_mixture")][metric]
                    - lookup[(seed, comparator)][metric]
                    for seed in seeds
                ],
                dtype=float,
            )
            low, high = _interval(differences)
            paired.append(
                {
                    "reference": "global_dual_mixture",
                    "comparator": comparator,
                    "metric": metric,
                    "difference": float(differences.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return per_seed, summary, paired


def write_report(destination: Path, summary, paired) -> None:
    overall = {
        row["method"]: row for row in summary if row["scope"] == "overall"
    }
    differences = {
        (row["comparator"], row["metric"]): row for row in paired
    }
    lines = [
        "# Mechanism baseline development v1",
        "",
        "This is three-seed development evidence, not formal confirmation.",
        "",
        "| Method | Accuracy | Brier | NLL | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = overall[method]
        lines.append(
            f"| {method} | {100 * row['correct_commitment_mean']:.2f}% | "
            f"{row['hypothesis_brier_mean']:.4f} | "
            f"{row['hypothesis_nll_mean']:.4f} | "
            f"{row['hypothesis_ece_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Global dual mixture paired differences",
            "",
            "Positive accuracy favors the dual mixture; negative Brier and NLL favor it.",
            "",
            "| Comparator | Accuracy pp (95% CI) | Brier | NLL |",
            "|---|---:|---:|---:|",
        ]
    )
    for comparator in METHODS:
        if comparator == "global_dual_mixture":
            continue
        accuracy = differences[(comparator, "correct_commitment")]
        brier = differences[(comparator, "hypothesis_brier")]
        nll = differences[(comparator, "hypothesis_nll")]
        lines.append(
            f"| {comparator} | {100 * accuracy['difference']:+.2f} "
            f"[{100 * accuracy['ci95_low']:+.2f}, {100 * accuracy['ci95_high']:+.2f}] | "
            f"{brier['difference']:+.4f} | {nll['difference']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "Intervals use paired training-seed means. Crossing zero is not interpreted as equivalence.",
        ]
    )
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def formal_decision(paired):
    lookup = {(row["comparator"], row["metric"]): row for row in paired}
    nll = lookup[("huber", "hypothesis_nll")]
    brier = lookup[("huber", "hypothesis_brier")]
    accuracy = lookup[("huber", "correct_commitment")]
    conditions = {
        "nll_upper_below_zero": nll["ci95_high"] < 0.0,
        "brier_upper_below_zero": brier["ci95_high"] < 0.0,
        "accuracy_lower_above_minus_two_points": accuracy["ci95_low"] > -0.02,
    }
    return {
        "decision": "CONFIRMED" if all(conditions.values()) else "NOT_CONFIRMED",
        "primary_comparator": "huber",
        "conditions": conditions,
        "all_conditions_required": True,
    }


def run(
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    seeds: tuple[int, ...] = SEEDS,
    episodes: int = 100,
    likelihood_episodes: int = 10,
    calibration_episodes: int = 3,
    max_cycles: int = 10,
    smoke: bool = False,
    formal: bool = False,
):
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError("refusing to overwrite an existing output directory")
    destination.mkdir(parents=True)
    protocol = {
        "protocol": "RCL Mechanism Baseline Development v1",
        "status": (
            "smoke" if smoke else "formal_confirmation" if formal else "development_not_formal"
        ),
        "seeds": list(seeds),
        "families": list(RED_FAMILIES),
        "methods": list(METHODS),
        "episodes_per_method_family_seed": episodes,
        "expected_test_episodes": len(seeds)
        * len(RED_FAMILIES)
        * len(METHODS)
        * episodes,
        "mechanism_label_budget": "same full training cell labels for every method",
        "shared_reliable_emissions": True,
        "global_mixture_compromised_emissions": "same labeled training sample",
        "student_dof_validation_grid": list(STUDENT_DOF_GRID),
        "huber_delta_validation_grid": list(HUBER_DELTA_GRID),
        "reliability_weight_power": RELIABILITY_WEIGHT_POWER,
        "temperature_validation_grid": "geomspace(0.1, 10.0, 101)",
        "fixed_commit_step": 4,
        "interval_unit": "training seed",
        "test_tuning_forbidden": True,
        "formal_claim_allowed": formal,
    }
    (destination / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    all_rows = []
    for seed in seeds:
        reasoner_records = collect_prefix_records(
            TRAIN_FAMILIES,
            2 if smoke else 8,
            seed=600_000_000 + seed * 10_003,
            split=f"mechanism_reasoner_{seed}",
            max_cycles=max_cycles,
        )
        reasoner = HonestIntelligenceReasoner(reasoner_records)
        samples = collect_reliability_samples(
            RELIABILITY_TRAIN_FAMILIES,
            likelihood_episodes,
            seed=700_000_000 + seed * 10_003,
            reasoner=reasoner,
            split=f"mechanism_likelihood_{seed}",
            max_cycles=max_cycles,
        )
        emissions = ReliabilityConditionedLikelihood.fit(samples)
        cases = collect_belief_cases(
            RELIABILITY_TRAIN_FAMILIES,
            calibration_episodes,
            seed=750_000_000 + seed * 10_003,
            max_cycles=max_cycles,
        )
        global_predictor = GroupedFixedReliability.fit(samples, granularity="global")
        cell_predictor = GroupedFixedReliability.fit(samples, granularity="source_region")
        certain_predictor = GroupedFixedReliability(
            np.full((3, 3), 1.0 - 1e-8), "source_region"
        )

        single_raw = SingleReliableLikelihood(
            emissions.reliable_mean,
            emissions.reliable_scale,
            emissions.compromised_mean,
            emissions.compromised_scale,
        )
        single = fit_likelihood_temperature(
            single_raw, cases, certain_predictor, reasoner
        )
        mixture = fit_likelihood_temperature(
            emissions, cases, global_predictor, reasoner
        )
        student, selected_dof, student_grid = select_and_calibrate(
            [
                (
                    value,
                    StudentTReliableLikelihood.from_emissions(
                        emissions, degrees_of_freedom=value
                    ),
                )
                for value in STUDENT_DOF_GRID
            ],
            cases,
            certain_predictor,
            reasoner,
        )
        huber, selected_delta, huber_grid = select_and_calibrate(
            [
                (
                    value,
                    HuberReliableLikelihood.from_emissions(emissions, delta=value),
                )
                for value in HUBER_DELTA_GRID
            ],
            cases,
            certain_predictor,
            reasoner,
        )
        weighted = fit_likelihood_temperature(
            ReliabilityWeightedGaussianLikelihood.from_emissions(
                emissions, weight_power=RELIABILITY_WEIGHT_POWER
            ),
            cases,
            cell_predictor,
            reasoner,
        )
        controllers = {
            "single_mechanism": ReliabilityLikelihoodController(
                single, certain_predictor, reasoner, name="single_mechanism"
            ),
            "global_dual_mixture": ReliabilityLikelihoodController(
                mixture, global_predictor, reasoner, name="global_dual_mixture"
            ),
            "student_t": ReliabilityLikelihoodController(
                student, certain_predictor, reasoner, name="student_t"
            ),
            "huber": ReliabilityLikelihoodController(
                huber, certain_predictor, reasoner, name="huber"
            ),
            "reliability_weighted": ReliabilityLikelihoodController(
                weighted, cell_predictor, reasoner, name="reliability_weighted"
            ),
        }
        parameters = {
            "selected_student_dof": selected_dof,
            "selected_huber_delta": selected_delta,
            "reliability_weight_power": RELIABILITY_WEIGHT_POWER,
            "candidate_validation": {
                "student_t": student_grid,
                "huber": huber_grid,
            },
            "temperatures": {
                name: controller.likelihood.temperature
                for name, controller in controllers.items()
            },
            "global_reliability": global_predictor.reliable.tolist(),
            "cell_reliability": cell_predictor.reliable.tolist(),
        }
        (destination / f"parameters_seed{seed}.json").write_text(
            json.dumps(parameters, indent=2), encoding="utf-8"
        )

        seed_rows = []
        for family_index, family in enumerate(RED_FAMILIES):
            for episode in range(episodes):
                hypothesis = (episode + seed + family_index) % NUM_HYPOTHESES
                scenario_seed = (
                    800_000_000 + seed * 100_003 + family_index * 1_009 + episode
                )
                for name, controller in controllers.items():
                    row = evaluate_episode(
                        controller,
                        family,
                        hypothesis,
                        seed=scenario_seed,
                        max_cycles=max_cycles,
                    )
                    if row["controller"] != name:
                        raise AssertionError("method name drift")
                    row["training_seed"] = seed
                    seed_rows.append(row)
        write_csv(destination / f"episodes_seed{seed}.csv", seed_rows)
        all_rows.extend(seed_rows)

    if len(all_rows) != protocol["expected_test_episodes"]:
        raise AssertionError("evaluation episode count drift")
    per_seed, summary, paired = analyze(all_rows, seeds)
    write_csv(destination / "episodes.csv", all_rows)
    write_csv(destination / "per_seed_summary.csv", per_seed)
    write_csv(destination / "summary.csv", summary)
    write_csv(destination / "paired_differences.csv", paired)
    write_report(destination, summary, paired)
    if formal:
        (destination / "decision.json").write_text(
            json.dumps(formal_decision(paired), indent=2), encoding="utf-8"
        )
    return {"episodes": len(all_rows), "smoke": smoke, "formal": formal}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        result = run(
            args.output_dir,
            seeds=(9191,),
            episodes=1,
            likelihood_episodes=1,
            calibration_episodes=1,
            smoke=True,
        )
    else:
        result = run(args.output_dir)
    print(json.dumps(result, indent=2))
