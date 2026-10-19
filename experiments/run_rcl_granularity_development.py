"""Matched development comparison of global, source, and source-region RCL."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES
from deception_rl.intelligence.ride import HonestIntelligenceReasoner
from deception_rl.intelligence.ride_v2.granularity_baselines import (
    GroupedFixedReliability,
)
from deception_rl.intelligence.ride_v2.reliability_conditioned import (
    ReliabilityConditionedLikelihood,
    ReliabilityLikelihoodController,
)
from deception_rl.intelligence.adapter import audit_blue_observation
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "development" / "rcl_granularity_v1"
METHODS = (
    "single_mechanism",
    "global_mixture",
    "source_mixture",
    "source_region_rcl",
)
METRICS = (
    "correct_commitment",
    "hypothesis_brier",
    "hypothesis_nll",
    "hypothesis_ece",
    "regret",
    "decision_latency",
    "information_cost",
)


class SingleReliableLikelihood(ReliabilityConditionedLikelihood):
    """Matched reliable emissions with the compromised component removed."""

    def logits(self, observation, predictor, reasoner):
        audit_blue_observation(observation)
        reports = np.asarray(observation["reports"], dtype=float)
        mask = np.asarray(observation["report_mask"], dtype=bool)
        times = np.flatnonzero(np.asarray(observation["time_mask"], dtype=bool))
        result = np.full(NUM_HYPOTHESES, -np.log(NUM_HYPOTHESES))
        for timestep in times:
            for source, region in np.argwhere(mask[timestep]):
                result += self._log_normal(
                    reports[timestep, source, region, :2],
                    self.reliable_mean[:, source, region],
                    self.reliable_scale[:, source, region],
                )
        return result / self.temperature

    def with_temperature(self, temperature):
        return SingleReliableLikelihood(
            self.reliable_mean,
            self.reliable_scale,
            self.compromised_mean,
            self.compromised_scale,
            float(temperature),
        )


def expected_calibration_error(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    error = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == 9 else probabilities < upper
        )
        if selected.any():
            error += float(selected.mean()) * abs(
                float(probabilities[selected].mean()) - float(outcomes[selected].mean())
            )
    return error


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize an empty group")
    correct = np.asarray([row["interception"] for row in rows], dtype=float)
    confidence = np.asarray([row["confidence"] for row in rows], dtype=float)
    return {
        "correct_commitment": float(correct.mean()),
        "hypothesis_brier": float(np.mean([row["hypothesis_brier"] for row in rows])),
        "hypothesis_nll": float(np.mean([row["hypothesis_nll"] for row in rows])),
        "hypothesis_ece": expected_calibration_error(confidence, correct),
        "regret": float(np.mean([row["regret"] for row in rows])),
        "decision_latency": float(np.mean([row["decision_latency"] for row in rows])),
        "information_cost": float(np.mean([row["information_cost"] for row in rows])),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 10: 2.262}.get(
        len(values), 1.96
    )
    half_width = critical * values.std(ddof=1) / math.sqrt(len(values))
    return float(values.mean() - half_width), float(values.mean() + half_width)


def analyze(rows: list[dict[str, Any]], seeds: tuple[int, ...]):
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["training_seed"]), row["controller"], row["deception_family"])].append(row)

    per_seed = []
    for seed in seeds:
        for method in METHODS:
            selected = [
                row
                for row in rows
                if int(row["training_seed"]) == seed and row["controller"] == method
            ]
            per_seed.append(
                {
                    "seed": seed,
                    "method": method,
                    "scope": "overall",
                    "episodes": len(selected),
                    **{key: value for key, value in summarize_group(selected).items() if key in METRICS},
                }
            )
            for family in RED_FAMILIES:
                family_rows = grouped[(seed, method, family)]
                per_seed.append(
                    {
                        "seed": seed,
                        "method": method,
                        "scope": family,
                        "episodes": len(family_rows),
                        **{
                            key: value
                            for key, value in summarize_group(family_rows).items()
                            if key in METRICS
                        },
                    }
                )

    summary = []
    for method in METHODS:
        for scope in ("overall", *RED_FAMILIES):
            selected = [
                row for row in per_seed if row["method"] == method and row["scope"] == scope
            ]
            result: dict[str, Any] = {
                "method": method,
                "scope": scope,
                "seeds": len(selected),
                "episodes": sum(int(row["episodes"]) for row in selected),
            }
            for metric in METRICS:
                values = np.asarray([row[metric] for row in selected], dtype=float)
                low, high = _interval(values)
                result[f"{metric}_mean"] = float(values.mean())
                result[f"{metric}_ci95_low"] = low
                result[f"{metric}_ci95_high"] = high
            summary.append(result)

    lookup = {(row["seed"], row["method"]): row for row in per_seed if row["scope"] == "overall"}
    paired = []
    for comparator in METHODS[:-1]:
        for metric in ("correct_commitment", "hypothesis_brier", "hypothesis_nll", "hypothesis_ece"):
            differences = np.asarray(
                [lookup[(seed, "source_region_rcl")][metric] - lookup[(seed, comparator)][metric] for seed in seeds]
            )
            low, high = _interval(differences)
            paired.append(
                {
                    "reference": "source_region_rcl",
                    "comparator": comparator,
                    "metric": metric,
                    "difference": float(differences.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return per_seed, summary, paired


def _controller(
    name: str,
    likelihood: ReliabilityConditionedLikelihood,
    predictor: GroupedFixedReliability,
    reasoner: HonestIntelligenceReasoner,
) -> ReliabilityLikelihoodController:
    return ReliabilityLikelihoodController(likelihood, predictor, reasoner, name=name)


def run(
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    seeds: tuple[int, ...] = (9101, 9102, 9103),
    episodes: int = 100,
    likelihood_episodes: int = 10,
    calibration_episodes: int = 3,
    max_cycles: int = 10,
    smoke: bool = False,
):
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError("refusing to overwrite an existing output directory")
    if min(len(seeds), episodes, likelihood_episodes, calibration_episodes) < 1:
        raise ValueError("seeds and episode counts must be nonzero")
    destination.mkdir(parents=True)
    protocol = {
        "protocol": "RCL Granularity Development v1",
        "status": "smoke" if smoke else "development_not_formal",
        "seeds": list(seeds),
        "families": list(RED_FAMILIES),
        "methods": list(METHODS),
        "episodes_per_method_family_seed": episodes,
        "expected_test_episodes": len(seeds) * len(RED_FAMILIES) * len(METHODS) * episodes,
        "likelihood_fit_families": list(RELIABILITY_TRAIN_FAMILIES),
        "likelihood_episodes_per_hypothesis_family": likelihood_episodes,
        "temperature_validation_cases_per_seen_family": calibration_episodes,
        "mechanism_label_budget": "identical full cell labels for all mixture variants",
        "emissions": "shared reliable and compromised Gaussian estimates",
        "fixed_commit_step": 4,
        "interval_unit": "training seed",
        "test_tuning_forbidden": True,
    }
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    all_rows = []
    for seed in seeds:
        reasoner_records = collect_prefix_records(
            TRAIN_FAMILIES,
            2 if smoke else 8,
            seed=600_000_000 + seed * 10_003,
            split=f"granularity_reasoner_{seed}",
            max_cycles=max_cycles,
        )
        reasoner = HonestIntelligenceReasoner(reasoner_records)
        samples = collect_reliability_samples(
            RELIABILITY_TRAIN_FAMILIES,
            likelihood_episodes,
            seed=700_000_000 + seed * 10_003,
            reasoner=reasoner,
            split=f"granularity_likelihood_{seed}",
            max_cycles=max_cycles,
        )
        emissions = ReliabilityConditionedLikelihood.fit(samples)
        cases = collect_belief_cases(
            RELIABILITY_TRAIN_FAMILIES,
            calibration_episodes,
            seed=750_000_000 + seed * 10_003,
            max_cycles=max_cycles,
        )
        predictors = {
            granularity: GroupedFixedReliability.fit(samples, granularity=granularity)
            for granularity in ("global", "source", "source_region")
        }
        calibrated = {
            granularity: fit_likelihood_temperature(emissions, cases, predictor, reasoner)
            for granularity, predictor in predictors.items()
        }
        single_predictor = GroupedFixedReliability(
            np.full((3, 3), 1.0 - 1e-8), "source_region"
        )
        single_raw = SingleReliableLikelihood(
            emissions.reliable_mean,
            emissions.reliable_scale,
            emissions.compromised_mean,
            emissions.compromised_scale,
        )
        single = fit_likelihood_temperature(single_raw, cases, single_predictor, reasoner)
        controllers = {
            "single_mechanism": _controller("single_mechanism", single, single_predictor, reasoner),
            "global_mixture": _controller("global_mixture", calibrated["global"], predictors["global"], reasoner),
            "source_mixture": _controller("source_mixture", calibrated["source"], predictors["source"], reasoner),
            "source_region_rcl": _controller(
                "source_region_rcl", calibrated["source_region"], predictors["source_region"], reasoner
            ),
        }
        parameters = {
            name: {
                "reliability": controller.predictor.reliable.tolist(),
                "temperature": controller.likelihood.temperature,
            }
            for name, controller in controllers.items()
        }
        (destination / f"parameters_seed{seed}.json").write_text(
            json.dumps(parameters, indent=2), encoding="utf-8"
        )
        seed_rows = []
        for family_index, family in enumerate(RED_FAMILIES):
            for episode in range(episodes):
                hypothesis = (episode + seed + family_index) % NUM_HYPOTHESES
                scenario_seed = 800_000_000 + seed * 100_003 + family_index * 1_009 + episode
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
    return {"output_dir": str(destination), "episodes": len(all_rows), "smoke": smoke}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        result = run(
            args.output_dir,
            seeds=(9101,),
            episodes=2,
            likelihood_episodes=1,
            calibration_episodes=1,
            smoke=True,
        )
    else:
        result = run(args.output_dir)
    print(json.dumps(result, indent=2))

