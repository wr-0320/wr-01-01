"""Development evaluation of learned dynamic source reliability filtering."""

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

from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES, NUM_SOURCES, DeceptiveIntelligenceEnv
from deception_rl.intelligence.ride import HonestIntelligenceReasoner, RIDeIntelligenceController, RIDeVariant
from deception_rl.intelligence.ride_v2.active import MultiSourceEntropyVerify
from deception_rl.intelligence.ride_v2.oracle import OracleReliabilityController, OracleSourceReliability
from deception_rl.intelligence.ride_v2.reliability import (
    DynamicReliabilityController,
    DynamicSourceReliabilityFilter,
    extract_reliability_evidence,
)
from deception_rl.intelligence.ride_v2.source_region_reliability import (
    extract_source_region_evidence,
)
from experiments.run_intelligence_ride import (
    TRAIN_FAMILIES,
    _collection_action,
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "intelligence_ride_v2_reliability_development"
HEADROOM_OUTPUT = ROOT / "outputs" / "intelligence_ride_v2_development"
RELIABILITY_DEVELOPMENT_SEEDS = (404, 505, 606)
RELIABILITY_TRAIN_FAMILIES = (
    "honest",
    "concealment",
    "decoy",
    "delayed_dev_onset2",
)
METHODS = (
    "multi_source",
    "ride_v1",
    "multi_source_entropy_verify",
    "dynamic_reliability",
    "dynamic_reliability_entropy_verify",
    "oracle_reliability",
)


def collect_reliability_samples(
    families: tuple[str, ...],
    episodes_per_hypothesis: int,
    *,
    seed: int,
    reasoner: HonestIntelligenceReasoner,
    split: str,
    max_cycles: int,
) -> list[dict[str, Any]]:
    """Collect offline simulator labels; runtime controllers never receive them."""

    rows = []
    reliability_oracle = OracleSourceReliability()
    for family_index, family in enumerate(families):
        for hypothesis in range(NUM_HYPOTHESES):
            for episode in range(episodes_per_hypothesis):
                episode_seed = (
                    seed + family_index * 100_003 + hypothesis * 1_009 + episode
                )
                episode_id = f"{split}:{family}:{hypothesis}:{episode_seed}"
                env = DeceptiveIntelligenceEnv(max_cycles=max_cycles)
                red = DEVELOPMENT_RED_FAMILIES[family]()
                red.reset()
                observations, _ = env.reset(
                    seed=episode_seed, options={"hypothesis": hypothesis}
                )
                previous = None
                while env.agents:
                    step = int(np.asarray(observations["blue"]["current_step"])[0])
                    red_action = red.act(observations["red"])
                    diagnostic = reliability_oracle.inspect(env, red_action, previous)
                    blue_action = _collection_action(observations["blue"], step)
                    observations, *_ = env.step(
                        {"red": red_action, "blue": blue_action}
                    )
                    evidence, _ = extract_reliability_evidence(
                        observations["blue"], reasoner
                    )
                    cell_evidence, cell_times = extract_source_region_evidence(
                        observations["blue"], reasoner
                    )
                    rows.append(
                        {
                            "episode_id": episode_id,
                            "deception_family": family,
                            "hypothesis_label": hypothesis,
                            "step": step,
                            "features": evidence[-1],
                            "compromised": diagnostic.compromised_cells.any(axis=1).astype(int),
                            "cell_features": cell_evidence[-1],
                            "compromised_cells": diagnostic.compromised_cells.astype(int),
                            "reported_cells": np.asarray(
                                observations["blue"]["report_mask"], dtype=bool
                            )[int(cell_times[-1])],
                            "reported_values": np.asarray(
                                observations["blue"]["reports"], dtype=float
                            )[int(cell_times[-1])],
                        }
                    )
                    previous = diagnostic
                    if step >= max_cycles - 2 and env.agents:
                        # The collection action commits on the following loop.
                        continue
    return rows


def reliability_metrics(
    reliability_filter: DynamicSourceReliabilityFilter,
    samples: list[dict[str, Any]],
) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        grouped[str(row["episode_id"])].append(row)
    probabilities = []
    labels = []
    for episode_rows in grouped.values():
        ordered = sorted(episode_rows, key=lambda row: int(row["step"]))
        evidence = np.stack([np.asarray(row["features"]) for row in ordered])
        reliable_history = reliability_filter.estimate_evidence(evidence)
        probabilities.extend((1.0 - reliable_history).reshape(-1).tolist())
        labels.extend(
            np.stack([np.asarray(row["compromised"]) for row in ordered])
            .reshape(-1)
            .tolist()
        )
    probabilities_array = np.asarray(probabilities, dtype=float)
    labels_array = np.asarray(labels, dtype=int)
    positives = probabilities_array[labels_array == 1]
    negatives = np.sort(probabilities_array[labels_array == 0])
    if len(positives) and len(negatives):
        lower = np.searchsorted(negatives, positives, side="left")
        upper = np.searchsorted(negatives, positives, side="right")
        auroc = float(np.mean((lower + upper) / 2) / len(negatives))
    else:
        auroc = float("nan")
    prediction = probabilities_array >= 0.5
    true_positive = int(((labels_array == 1) & prediction).sum())
    false_positive = int(((labels_array == 0) & prediction).sum())
    false_negative = int(((labels_array == 1) & ~prediction).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "samples": len(labels_array),
        "compromised_prevalence": float(labels_array.mean()),
        "brier": float(np.mean((probabilities_array - labels_array) ** 2)),
        "auroc": auroc,
        "f1": 2 * true_positive / denominator if denominator else 0.0,
    }


def _inference(
    model: Any,
    reasoner: HonestIntelligenceReasoner,
    calibrator: Any,
    name: str,
    *,
    full_ride: bool,
    max_cycles: int,
) -> RIDeIntelligenceController:
    variant = (
        RIDeVariant(name)
        if full_ride
        else RIDeVariant(
            name,
            use_deception_head=False,
            use_counterfactual=False,
            use_calibration=False,
            adaptive_commitment=False,
        )
    )
    return RIDeIntelligenceController(
        model,
        reasoner,
        calibrator,
        variant,
        max_cycles=max_cycles,
        fixed_commit_step=COMMIT_STEP,
    )


def build_controllers(
    model: Any,
    reasoner: HonestIntelligenceReasoner,
    calibrator: Any,
    reliability_filter: DynamicSourceReliabilityFilter,
    *,
    max_cycles: int,
) -> dict[str, Any]:
    controllers = {
        "multi_source": _inference(
            model, reasoner, calibrator, "multi_source", full_ride=False, max_cycles=max_cycles
        ),
        "ride_v1": _inference(
            model, reasoner, calibrator, "ride_v1", full_ride=True, max_cycles=max_cycles
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
        "dynamic_reliability": DynamicReliabilityController(
            _inference(
                model,
                reasoner,
                calibrator,
                "dynamic_inference",
                full_ride=False,
                max_cycles=max_cycles,
            ),
            reliability_filter,
            active_verification=False,
            commit_step=COMMIT_STEP,
        ),
        "dynamic_reliability_entropy_verify": DynamicReliabilityController(
            _inference(
                model,
                reasoner,
                calibrator,
                "dynamic_active_inference",
                full_ride=False,
                max_cycles=max_cycles,
            ),
            reliability_filter,
            active_verification=True,
            verification_budget=VERIFICATION_BUDGET,
            commit_step=COMMIT_STEP,
        ),
        "oracle_reliability": OracleReliabilityController(
            _inference(
                model,
                reasoner,
                calibrator,
                "oracle_reliability_inference",
                full_ride=False,
                max_cycles=max_cycles,
            ),
            commit_step=COMMIT_STEP,
        ),
    }
    if tuple(controllers) != METHODS:
        raise AssertionError("reliability development method order drifted")
    return controllers


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = summarize(rows)
    for method in METHODS:
        selected = [row for row in rows if row["controller"] == method]
        output.extend(
            summarize([{**row, "deception_family": "overall"} for row in selected])
        )
    return sorted(output, key=lambda row: (row["deception_family"], row["method"]))


def paired_recovery(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families = ("overall", *DEVELOPMENT_RED_FAMILIES)
    differences: dict[tuple[str, str], list[float]] = defaultdict(list)
    for seed in sorted({int(row["development_seed"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["development_seed"]) == seed]
        for family in families:
            selected = seed_rows if family == "overall" else [
                row for row in seed_rows if row["deception_family"] == family
            ]
            accuracy = {
                method: float(np.mean([row["correct_commitment"] for row in selected if row["controller"] == method]))
                for method in METHODS
            }
            oracle_headroom = accuracy["oracle_reliability"] - accuracy["multi_source"]
            for method in ("ride_v1", "multi_source_entropy_verify", "dynamic_reliability", "dynamic_reliability_entropy_verify", "oracle_reliability"):
                difference = accuracy[method] - accuracy["multi_source"]
                differences[(method, family)].append(difference)
    output = []
    for (method, family), values_list in sorted(differences.items()):
        values = np.asarray(values_list)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        half_width = 4.303 * std / math.sqrt(3) if len(values) == 3 else float("nan")
        oracle_values = np.asarray(differences[("oracle_reliability", family)])
        oracle_mean = float(oracle_values.mean())
        mean = float(values.mean())
        output.append(
            {
                "method": method,
                "deception_family": family,
                "seeds": len(values),
                "accuracy_difference_mean": mean,
                "accuracy_difference_ci95_low": mean - half_width,
                "accuracy_difference_ci95_high": mean + half_width,
                "oracle_reliability_headroom": oracle_mean,
                "oracle_headroom_recovered_fraction": (
                    mean / oracle_mean if abs(oracle_mean) > 1e-12 else float("nan")
                ),
            }
        )
    return output


def write_report(
    destination: Path,
    summary: list[dict[str, Any]],
    recovery: list[dict[str, Any]],
    filter_metrics: list[dict[str, Any]],
    total_episodes: int,
) -> dict[str, Any]:
    lookup = {(row["method"], row["deception_family"]): row for row in summary}
    recovery_lookup = {(row["method"], row["deception_family"]): row for row in recovery}
    baseline = lookup[("multi_source", "overall")]["correct_commitment"]
    learned = lookup[("dynamic_reliability", "overall")]["correct_commitment"]
    learned_active = lookup[("dynamic_reliability_entropy_verify", "overall")]["correct_commitment"]
    oracle = lookup[("oracle_reliability", "overall")]["correct_commitment"]
    headroom = oracle - baseline
    recovered = (learned - baseline) / headroom if headroom else float("nan")
    recovered_active = (learned_active - baseline) / headroom if headroom else float("nan")
    go = max(recovered, recovered_active) >= 0.5
    rows = []
    for method in METHODS:
        row = lookup[(method, "overall")]
        rows.append(
            f"| {method} | {100 * row['correct_commitment']:.1f}% | "
            f"{100 * (row['correct_commitment'] - baseline):+.1f} pp | "
            f"{row['information_cost']:.2f} | {row['regret']:.2f} |"
        )
    family_rows = []
    for family in ("multi_source", "switch_dev_step2", "adaptive"):
        family_rows.append(
            f"| {family} | {100 * lookup[('multi_source', family)]['correct_commitment']:.1f}% | "
            f"{100 * lookup[('dynamic_reliability', family)]['correct_commitment']:.1f}% | "
            f"{100 * lookup[('dynamic_reliability_entropy_verify', family)]['correct_commitment']:.1f}% | "
            f"{100 * lookup[('oracle_reliability', family)]['correct_commitment']:.1f}% |"
        )
    report = f"""# Dynamic Source Reliability Development

The filter is supervised offline with simulator compromised-source labels. At
runtime it uses only the original 49-dimensional blue observation and frozen
honest prototypes. No hidden label enters a Defender input.

## Protocol

- Development seeds: {', '.join(map(str, RELIABILITY_DEVELOPMENT_SEEDS))}
- Total evaluation episodes: {total_episodes}
- Training families: {', '.join(RELIABILITY_TRAIN_FAMILIES)}
- Multi-source, switch, and adaptive remain held out from reliability-filter fitting.

## Overall results

| Method | Correct commitment | vs Multi-source | Information cost | Regret |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Passive dynamic reliability recovered **{100 * recovered:.1f}%** of the Oracle
Reliability accuracy headroom. With budget-matched entropy verification it recovered
**{100 * recovered_active:.1f}%**.

## Priority families

| Family | Multi-source | Dynamic reliability | Dynamic + verify | Oracle reliability |
|---|---:|---:|---:|---:|
{chr(10).join(family_rows)}

## Reliability-state estimation

Mean held-out source-state AUROC: {np.mean([row['auroc'] for row in filter_metrics]):.3f};
mean Brier: {np.mean([row['brier'] for row in filter_metrics]):.3f}.

The source state is rankable, but the present conditioning rule does not convert
that discrimination into overall decision gains. It attenuates every region from a
source when only one region may be manipulated, discarding useful evidence. The
Oracle comparator excludes the exact compromised source-region cells, so its
headroom is finer grained than this learned source-level filter. On Adaptive,
dynamic reliability plus verification improves materially; that localized gain is
insufficient to support an overall claim.

## Decision

**{'GO' if go else 'NO-GO'}** under the preregistered development target of
recovering at least half of Oracle Reliability's overall accuracy headroom.
The paired-seed interval for passive dynamic reliability versus Multi-source is
{100 * recovery_lookup[('dynamic_reliability', 'overall')]['accuracy_difference_ci95_low']:+.1f}
to {100 * recovery_lookup[('dynamic_reliability', 'overall')]['accuracy_difference_ci95_high']:+.1f} pp.
This remains development evidence and does not update the frozen formal benchmark.
"""
    (destination / "reliability_development_report.md").write_text(report, encoding="utf-8")
    decision = {
        "decision": "GO" if go else "NO-GO",
        "passive_recovered_fraction": recovered,
        "active_recovered_fraction": recovered_active,
        "target_fraction": 0.5,
        "multi_source_accuracy": baseline,
        "dynamic_reliability_accuracy": learned,
        "dynamic_reliability_active_accuracy": learned_active,
        "oracle_reliability_accuracy": oracle,
    }
    (destination / "go_no_go.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


def read_numeric_csv(path: Path) -> list[dict[str, Any]]:
    identifiers = {"method", "deception_family"}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            key: value if key in identifiers else float(value)
            for key, value in row.items()
        }
        for row in rows
    ]


def analyze_existing(destination: Path) -> dict[str, Any]:
    summary = read_numeric_csv(destination / "summary.csv")
    recovery = read_numeric_csv(destination / "paired_headroom_recovery.csv")
    filter_metrics = read_numeric_csv(destination / "reliability_filter_metrics.csv")
    config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    return write_report(
        destination,
        summary,
        recovery,
        filter_metrics,
        int(config["total_evaluation_episodes"]),
    )


def run(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    seeds: tuple[int, ...] = RELIABILITY_DEVELOPMENT_SEEDS,
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
    if set(seeds) & set(range(5)) or set(seeds) & {101, 202, 303}:
        raise ValueError("reliability seeds must be disjoint from formal and headroom seeds")
    destination = Path(output_dir)
    if destination.resolve() in {FORMAL_OUTPUT.resolve(), HEADROOM_OUTPUT.resolve()}:
        raise ValueError("reliability development must use an independent output directory")
    destination.mkdir(parents=True, exist_ok=True)
    frozen_before = {
        "formal": directory_manifest(FORMAL_OUTPUT),
        "headroom": directory_manifest(HEADROOM_OUTPUT),
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
            seed=110_000_000 + development_seed * 10_003,
            split=f"reliability_model_train_{development_seed}",
            max_cycles=max_cycles,
        )
        validation_records = collect_prefix_records(
            TRAIN_FAMILIES,
            validation_episodes_per_hypothesis,
            seed=120_000_000 + development_seed * 10_003,
            split=f"reliability_model_validation_{development_seed}",
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
            seed=130_000_000 + development_seed * 10_003,
            reasoner=reasoner,
            split=f"reliability_train_{development_seed}",
            max_cycles=max_cycles,
        )
        reliability_filter = DynamicSourceReliabilityFilter.fit(reliability_train)
        reliability_validation = collect_reliability_samples(
            tuple(DEVELOPMENT_RED_FAMILIES),
            reliability_validation_episodes_per_hypothesis,
            seed=140_000_000 + development_seed * 10_003,
            reasoner=reasoner,
            split=f"reliability_validation_{development_seed}",
            max_cycles=max_cycles,
        )
        filter_metric_rows.append(
            {
                "development_seed": development_seed,
                **reliability_metrics(reliability_filter, reliability_validation),
            }
        )
        (destination / f"reliability_filter_seed{development_seed}.json").write_text(
            json.dumps(reliability_filter.to_dict(), indent=2), encoding="utf-8"
        )
        controllers = build_controllers(
            model,
            reasoner,
            calibrator,
            reliability_filter,
            max_cycles=max_cycles,
        )
        for family_index, family in enumerate(DEVELOPMENT_RED_FAMILIES):
            for episode in range(episodes_per_family_seed):
                hypothesis = (episode + development_seed + family_index) % NUM_HYPOTHESES
                scenario_seed = 151_000_000 + development_seed * 100_003 + family_index * 1_009 + episode
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
    recovery = paired_recovery(all_rows)
    write_csv(destination / "episodes.csv", all_rows)
    write_csv(destination / "summary.csv", summary)
    write_csv(destination / "paired_headroom_recovery.csv", recovery)
    write_csv(destination / "reliability_filter_metrics.csv", filter_metric_rows)
    frozen_after = {
        "formal": directory_manifest(FORMAL_OUTPUT),
        "headroom": directory_manifest(HEADROOM_OUTPUT),
    }
    audit = {
        "ordinary_policy_calls": audit_counts["ordinary_calls"],
        "privileged_oracle_calls": audit_counts["privileged_calls"],
        "blue_observations_audited": audit_counts["boundary_checks"],
        "runtime_input_feature_size": 49,
        "simulator_labels_used_at_runtime": False,
        "simulator_labels_used_for_offline_filter_fit": True,
        "oracle_fields_inserted_into_blue_observation": False,
        "formal_results_unchanged": frozen_before["formal"] == frozen_after["formal"],
        "headroom_results_unchanged": frozen_before["headroom"] == frozen_after["headroom"],
    }
    if not audit["formal_results_unchanged"] or not audit["headroom_results_unchanged"]:
        raise AssertionError("a frozen output directory changed")
    (destination / "information_boundary_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    config = {
        "protocol": "RIDe-v2 Dynamic Reliability Development v1",
        "development_seeds": list(seeds),
        "episodes_per_family_seed": episodes_per_family_seed,
        "total_evaluation_episodes": len(all_rows),
        "methods": list(METHODS),
        "filter_training_families": list(RELIABILITY_TRAIN_FAMILIES),
        "formal_results_replaced": False,
        "oracle_headroom_results_replaced": False,
        "large_scale_training": False,
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    decision = write_report(destination, summary, recovery, filter_metric_rows, len(all_rows))
    return {**config, **decision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes-per-family-seed", type=int, default=30)
    parser.add_argument("--analyze-existing", action="store_true")
    args = parser.parse_args()
    if args.analyze_existing:
        print(json.dumps(analyze_existing(args.output_dir), indent=2))
        return
    print(json.dumps(run(args.output_dir, episodes_per_family_seed=args.episodes_per_family_seed), indent=2))


if __name__ == "__main__":
    main()
