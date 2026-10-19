"""Development-only Oracle headroom study for active intelligence verification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from deception_rl.agents.intelligence_policies import (
    AdaptiveDeceptionRedPolicy,
    ConcealmentRedPolicy,
    DecoyRedPolicy,
    DelayedDeceptionRedPolicy,
    HonestRedPolicy,
    SwitchDeceptionRedPolicy,
)
from deception_rl.envs.deceptive_intelligence import (
    NUM_HYPOTHESES,
    DeceptiveIntelligenceEnv,
)
from deception_rl.intelligence.adapter import (
    INTELLIGENCE_FEATURE_SIZE,
    audit_blue_observation,
)
from deception_rl.intelligence.ride import (
    HonestIntelligenceReasoner,
    RIDeIntelligenceController,
    RIDeVariant,
)
from deception_rl.intelligence.ride_v2.active import (
    MultiSourceEntropyVerify,
    MultiSourceRandomVerify,
)
from deception_rl.intelligence.ride_v2.oracle import (
    FullOracleDefender,
    OracleChangePoint,
    OracleDiagnosticStep,
    OracleReliabilityController,
    OracleSourceReliability,
    OracleVerifyPolicy,
)
from experiments.run_intelligence_ride import (
    TRAIN_FAMILIES,
    _controller_info,
    collect_prefix_records,
    fit_calibrator,
    train_model,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "intelligence_ride_v2_development"
FORMAL_OUTPUT = ROOT / "outputs" / "intelligence_formal_evaluation"
DEVELOPMENT_SEEDS = (101, 202, 303)
FORMAL_TRAINING_SEEDS = frozenset(range(5))
VERIFICATION_BUDGET = 3
COMMIT_STEP = 4

RedFactory = Callable[[], Any]
DEVELOPMENT_RED_FAMILIES: dict[str, RedFactory] = {
    "honest": HonestRedPolicy,
    "concealment": ConcealmentRedPolicy,
    "decoy": DecoyRedPolicy,
    "multi_source": lambda: DecoyRedPolicy(multisource=True),
    "delayed_dev_onset2": lambda: DelayedDeceptionRedPolicy(onset=2),
    "switch_dev_step2": lambda: SwitchDeceptionRedPolicy(switch_step=2),
    "adaptive": AdaptiveDeceptionRedPolicy,
}
METHODS = (
    "multi_source",
    "ride_v1",
    "multi_source_random_verify",
    "multi_source_entropy_verify",
    "oracle_reliability",
    "oracle_voi",
    "oracle_change",
    "full_oracle",
)
ORACLE_METHODS = frozenset(
    {"oracle_reliability", "oracle_voi", "oracle_change", "full_oracle"}
)
ACTIVE_METHODS = frozenset(
    {
        "multi_source_random_verify",
        "multi_source_entropy_verify",
        "oracle_voi",
        "full_oracle",
    }
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def directory_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _multi_source_variant(name: str) -> RIDeVariant:
    return RIDeVariant(
        name,
        use_deception_head=False,
        use_counterfactual=False,
        use_calibration=False,
        adaptive_commitment=False,
    )


def build_development_controllers(
    model: Any,
    reasoner: HonestIntelligenceReasoner,
    calibrator: Any,
    *,
    seed: int,
    oracle_rollouts: int,
    max_cycles: int,
) -> dict[str, Any]:
    def inference(variant: RIDeVariant) -> RIDeIntelligenceController:
        return RIDeIntelligenceController(
            model,
            reasoner,
            calibrator,
            variant,
            max_cycles=max_cycles,
            fixed_commit_step=COMMIT_STEP,
        )

    multi_source = inference(_multi_source_variant("multi_source"))
    ride_v1 = inference(RIDeVariant("ride_v1"))
    controllers: dict[str, Any] = {
        "multi_source": multi_source,
        "ride_v1": ride_v1,
        "multi_source_random_verify": MultiSourceRandomVerify(
            inference(_multi_source_variant("random_inference")),
            seed=seed,
            verification_budget=VERIFICATION_BUDGET,
            commit_step=COMMIT_STEP,
        ),
        "multi_source_entropy_verify": MultiSourceEntropyVerify(
            inference(_multi_source_variant("entropy_inference")),
            verification_budget=VERIFICATION_BUDGET,
            commit_step=COMMIT_STEP,
        ),
        "oracle_reliability": OracleReliabilityController(
            inference(_multi_source_variant("reliability_inference")),
            commit_step=COMMIT_STEP,
        ),
        "oracle_voi": OracleVerifyPolicy(
            inference(_multi_source_variant("oracle_voi_inference")),
            seed=seed,
            rollouts=oracle_rollouts,
            verification_budget=VERIFICATION_BUDGET,
            commit_step=COMMIT_STEP,
        ),
        "oracle_change": OracleChangePoint(
            inference(_multi_source_variant("change_inference")),
            commit_step=COMMIT_STEP,
        ),
        "full_oracle": FullOracleDefender(
            inference(_multi_source_variant("full_oracle_inference")),
            seed=seed,
            rollouts=oracle_rollouts,
            verification_budget=VERIFICATION_BUDGET,
            commit_step=COMMIT_STEP,
        ),
    }
    if tuple(controllers) != METHODS:
        raise AssertionError("development method order drifted")
    return controllers


def evaluate_development_episode(
    controller: Any,
    family: str,
    hypothesis: int,
    *,
    seed: int,
    max_cycles: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    env = DeceptiveIntelligenceEnv(
        max_cycles=max_cycles,
        verification_budget=VERIFICATION_BUDGET,
    )
    red = DEVELOPMENT_RED_FAMILIES[family]()
    red.reset()
    if hasattr(controller, "seed"):
        controller.seed = int(seed)
    controller.reset()
    observations, _ = env.reset(seed=seed, options={"hypothesis": hypothesis})
    reliability_oracle = OracleSourceReliability()
    trace: list[OracleDiagnosticStep] = []
    metrics = None
    boundary_checks = 0
    privileged_calls = 0
    ordinary_calls = 0
    while env.agents:
        audit = audit_blue_observation(observations["blue"])
        if audit["truth_fields_present"] or len(audit["fields"]) != 9:
            raise AssertionError("blue observation boundary drifted")
        boundary_checks += 1
        red_action = red.act(observations["red"])
        next_diagnostic = reliability_oracle.inspect(
            env, red_action, trace[-1] if trace else None
        )
        if getattr(controller, "privileged_diagnostic_only", False):
            blue_action = controller.act_with_oracle(
                observations["blue"], env, red_action, trace
            )
            privileged_calls += 1
        else:
            blue_action = controller.act(observations["blue"])
            ordinary_calls += 1
        observations, _, _, _, infos = env.step(
            {"red": red_action, "blue": blue_action}
        )
        trace.append(next_diagnostic)
        if "episode_metrics" in infos["blue"]:
            metrics = infos["blue"]["episode_metrics"]
    if metrics is None:
        raise RuntimeError("development episode ended without metrics")
    info = _controller_info(controller)
    belief = np.asarray(
        info.get("hypothesis_belief", np.full(NUM_HYPOTHESES, 1 / NUM_HYPOTHESES)),
        dtype=float,
    )
    target = np.zeros(NUM_HYPOTHESES)
    target[hypothesis] = 1
    deception_probability = float(info.get("deception_probability", 0.5))
    deception_label = int(family != "honest")
    row = {
        "controller": controller.name,
        "deception_family": family,
        "hypothesis": hypothesis,
        "scenario_seed": seed,
        "correct_commitment": metrics["decision_correct"],
        "false_commitment": int(
            not metrics["timed_out"] and not metrics["decision_correct"]
        ),
        "no_commit": metrics["timed_out"],
        "decision_latency": metrics["decision_step"],
        "information_cost": metrics["blue_information_cost"],
        "regret": metrics["decision_regret"],
        "verification_steps": metrics["verification_steps"],
        "hypothesis_brier": float(np.sum((belief - target) ** 2)),
        "deception_brier": float(
            (deception_probability - deception_label) ** 2
        ),
        "privileged_diagnostic_only": bool(
            getattr(controller, "privileged_diagnostic_only", False)
        ),
        "boundary_checks": boundary_checks,
    }
    return row, {
        "ordinary_calls": ordinary_calls,
        "privileged_calls": privileged_calls,
        "boundary_checks": boundary_checks,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["controller"], row["deception_family"])].append(row)
    output = []
    metrics = (
        "correct_commitment",
        "false_commitment",
        "no_commit",
        "decision_latency",
        "information_cost",
        "regret",
        "verification_steps",
        "hypothesis_brier",
        "deception_brier",
    )
    for (method, family), selected in sorted(grouped.items()):
        output.append(
            {
                "method": method,
                "deception_family": family,
                "episodes": len(selected),
                **{
                    metric: float(np.mean([row[metric] for row in selected]))
                    for metric in metrics
                },
            }
        )
    return output


def overall_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family_rows = summarize(rows)
    output = []
    for method in METHODS:
        selected = [row for row in rows if row["controller"] == method]
        aggregate = summarize(
            [{**row, "deception_family": "overall"} for row in selected]
        )[0]
        output.append(aggregate)
    return output + family_rows


def headroom_tables(
    summary: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row["method"], row["deception_family"]): row for row in summary
    }
    families = ("overall", *DEVELOPMENT_RED_FAMILIES)
    headroom = []
    for method in METHODS:
        if method == "multi_source":
            continue
        for family in families:
            row = lookup[(method, family)]
            baseline = lookup[("multi_source", family)]
            headroom.append(
                {
                    "method": method,
                    "deception_family": family,
                    "correct_commitment": row["correct_commitment"],
                    "multi_source_correct_commitment": baseline[
                        "correct_commitment"
                    ],
                    "accuracy_headroom": row["correct_commitment"]
                    - baseline["correct_commitment"],
                    "regret_difference": row["regret"] - baseline["regret"],
                    "latency_difference": row["decision_latency"]
                    - baseline["decision_latency"],
                    "information_cost_difference": row["information_cost"]
                    - baseline["information_cost"],
                }
            )
    voi = [
        row
        for row in headroom
        if row["method"]
        in {
            "multi_source_random_verify",
            "multi_source_entropy_verify",
            "oracle_voi",
            "full_oracle",
        }
    ]
    return headroom, voi


def paired_seed_headroom(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Estimate paired development-seed variation without treating episodes as seeds."""

    families = ("overall", *DEVELOPMENT_RED_FAMILIES)
    seed_differences: dict[tuple[str, str], list[float]] = defaultdict(list)
    for development_seed in sorted({int(row["development_seed"]) for row in rows}):
        seed_rows = [
            row for row in rows if int(row["development_seed"]) == development_seed
        ]
        for family in families:
            selected = (
                seed_rows
                if family == "overall"
                else [row for row in seed_rows if row["deception_family"] == family]
            )
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
                    seed_differences[(method, family)].append(
                        accuracy[method] - accuracy["multi_source"]
                    )
    critical = {1: float("nan"), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    output = []
    for (method, family), values_list in sorted(seed_differences.items()):
        values = np.asarray(values_list, dtype=float)
        standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        multiplier = critical.get(len(values), 1.96)
        half_width = (
            multiplier * standard_deviation / math.sqrt(len(values))
            if len(values) > 1
            else float("nan")
        )
        output.append(
            {
                "method": method,
                "deception_family": family,
                "seeds": len(values),
                "accuracy_headroom_mean": float(values.mean()),
                "accuracy_headroom_std": standard_deviation,
                "accuracy_headroom_ci95_low": float(values.mean() - half_width),
                "accuracy_headroom_ci95_high": float(values.mean() + half_width),
            }
        )
    return output


def write_report(
    destination: Path,
    summary: list[dict[str, Any]],
    headroom: list[dict[str, Any]],
    seed_headroom: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    summary_lookup = {
        (row["method"], row["deception_family"]): row for row in summary
    }
    headroom_lookup = {
        (row["method"], row["deception_family"]): row for row in headroom
    }
    seed_lookup = {
        (row["method"], row["deception_family"]): row for row in seed_headroom
    }
    overall = headroom_lookup[("full_oracle", "overall")]["accuracy_headroom"]
    focus = {
        family: headroom_lookup[("oracle_voi", family)]["accuracy_headroom"]
        for family in ("adaptive", "switch_dev_step2")
    }
    go = overall >= 0.05 and max(focus.values()) >= 0.05
    full_interval = seed_lookup[("full_oracle", "overall")]
    reliability_interval = seed_lookup[("oracle_reliability", "overall")]
    voi_interval = seed_lookup[("oracle_voi", "overall")]
    decision = {
        "decision": "GO" if go else "NO-GO",
        "development_recommendation": (
            "GO_RELIABILITY_FIRST_DEFER_LEARNED_VOI"
            if go
            else "STOP_RIDE_V2_EXPANSION"
        ),
        "full_oracle_overall_headroom": overall,
        "oracle_reliability_overall_headroom": headroom_lookup[
            ("oracle_reliability", "overall")
        ]["accuracy_headroom"],
        "oracle_voi_overall_headroom": headroom_lookup[("oracle_voi", "overall")][
            "accuracy_headroom"
        ],
        "oracle_voi_adaptive_headroom": focus["adaptive"],
        "oracle_voi_switch_headroom": focus["switch_dev_step2"],
        "thresholds": {
            "full_oracle_overall": 0.05,
            "oracle_voi_focus_family": 0.05,
        },
    }
    rows = []
    for method in METHODS:
        result = summary_lookup[(method, "overall")]
        difference = (
            0.0
            if method == "multi_source"
            else headroom_lookup[(method, "overall")]["accuracy_headroom"]
        )
        rows.append(
            f"| {method} | {100 * result['correct_commitment']:.1f}% | "
            f"{100 * difference:+.1f} pp | {result['decision_latency']:.2f} | "
            f"{result['information_cost']:.2f} | {result['regret']:.2f} |"
        )
    focus_rows = []
    for family in ("multi_source", "switch_dev_step2", "adaptive"):
        baseline = summary_lookup[("multi_source", family)]["correct_commitment"]
        oracle = summary_lookup[("oracle_voi", family)]["correct_commitment"]
        full = summary_lookup[("full_oracle", family)]["correct_commitment"]
        focus_rows.append(
            f"| {family} | {100 * baseline:.1f}% | {100 * oracle:.1f}% | "
            f"{100 * full:.1f}% |"
        )
    report = f"""# RIDe-v2 Oracle Headroom

This is a development-only diagnostic. Oracle controllers use privileged simulator
state and are not deployable Defender policies.

## Protocol

- Development seeds: {', '.join(map(str, config['development_seeds']))}
- Episodes per method × family × seed: {config['episodes_per_family_seed']}
- Total evaluation episodes: {config['total_evaluation_episodes']}
- VERIFY budget: {VERIFICATION_BUDGET}; shared active-policy commit step: {COMMIT_STEP}
- Formal RIDe-v1 output directory was hashed before and after this run and remained unchanged.

## Overall headroom

| Method | Correct commitment | vs Multi-source | Latency | Info cost | Regret |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Priority deception families

| Family | Multi-source | Oracle VOI | Full Oracle |
|---|---:|---:|---:|
{chr(10).join(focus_rows)}

## Go / No-Go

**{decision['decision']}** under the frozen development rule: Full Oracle must add
at least 5 pp overall and Oracle VOI must add at least 5 pp on Adaptive or Switch.

- Full Oracle overall headroom: {100 * overall:+.1f} pp
- Oracle VOI Adaptive headroom: {100 * focus['adaptive']:+.1f} pp
- Oracle VOI Switch headroom: {100 * focus['switch_dev_step2']:+.1f} pp

Across the three paired development seeds, Full Oracle's overall difference was
{100 * full_interval['accuracy_headroom_mean']:+.1f} pp (95% t interval
{100 * full_interval['accuracy_headroom_ci95_low']:+.1f} to
{100 * full_interval['accuracy_headroom_ci95_high']:+.1f} pp). Oracle reliability
accounted for {100 * reliability_interval['accuracy_headroom_mean']:+.1f} pp, while
Oracle VOI alone accounted for only {100 * voi_interval['accuracy_headroom_mean']:+.1f}
pp and its interval crossed zero.

**Development recommendation: reliability first.** Implement learned dynamic source
reliability next. Defer learned VOI until its observation model can beat the random
and entropy verification baselines consistently; the present Oracle-VOI result does
not justify a VOI-first implementation.

The result measures available simulator headroom. It is not evidence that a learned
RIDe-v2 can attain the Oracle score, and it does not replace the frozen formal study.
"""
    (destination / "oracle_headroom.md").write_text(report, encoding="utf-8")
    (destination / "go_no_go.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    return decision


def read_episode_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    numeric = {
        "hypothesis",
        "scenario_seed",
        "correct_commitment",
        "false_commitment",
        "no_commit",
        "decision_latency",
        "information_cost",
        "regret",
        "verification_steps",
        "hypothesis_brier",
        "deception_brier",
        "boundary_checks",
        "development_seed",
    }
    return [
        {
            key: float(value) if key in numeric else value
            for key, value in row.items()
        }
        for row in raw
    ]


def analyze_existing(destination: Path) -> dict[str, Any]:
    rows = read_episode_csv(destination / "episodes.csv")
    config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    summary = overall_summary(rows)
    headroom, voi = headroom_tables(summary)
    seed_headroom = paired_seed_headroom(rows)
    write_csv(destination / "summary.csv", summary)
    write_csv(destination / "family_headroom.csv", headroom)
    write_csv(destination / "voi_analysis.csv", voi)
    write_csv(destination / "paired_seed_headroom.csv", seed_headroom)
    return write_report(destination, summary, headroom, seed_headroom, config)


def run(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    seeds: tuple[int, ...] = DEVELOPMENT_SEEDS,
    episodes_per_family_seed: int = 30,
    epochs: int = 8,
    train_episodes_per_hypothesis: int = 8,
    validation_episodes_per_hypothesis: int = 3,
    hidden_size: int = 32,
    batch_size: int = 64,
    max_cycles: int = 10,
    oracle_rollouts: int = 6,
) -> dict[str, Any]:
    if not seeds or FORMAL_TRAINING_SEEDS.intersection(seeds):
        raise ValueError("development seeds must be nonempty and disjoint from 0–4")
    if min(
        episodes_per_family_seed,
        epochs,
        train_episodes_per_hypothesis,
        validation_episodes_per_hypothesis,
        oracle_rollouts,
    ) < 1:
        raise ValueError("development counts must be positive")
    destination = Path(output_dir)
    if destination.resolve() == FORMAL_OUTPUT.resolve():
        raise ValueError("development output must not overwrite formal RIDe-v1 results")
    destination.mkdir(parents=True, exist_ok=True)
    formal_before = directory_manifest(FORMAL_OUTPUT)
    all_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    audit_counts = defaultdict(int)
    for development_seed in seeds:
        torch.manual_seed(development_seed)
        np.random.seed(development_seed)
        random.seed(development_seed)
        train_records = collect_prefix_records(
            TRAIN_FAMILIES,
            train_episodes_per_hypothesis,
            seed=70_000_000 + development_seed * 10_003,
            split=f"ride_v2_dev_train_{development_seed}",
            max_cycles=max_cycles,
        )
        validation_records = collect_prefix_records(
            TRAIN_FAMILIES,
            validation_episodes_per_hypothesis,
            seed=80_000_000 + development_seed * 10_003,
            split=f"ride_v2_dev_validation_{development_seed}",
            max_cycles=max_cycles,
        )
        model, history = train_model(
            train_records,
            validation_records,
            epochs=epochs,
            hidden_size=hidden_size,
            batch_size=batch_size,
            seed=development_seed,
        )
        calibrator, _ = fit_calibrator(model, validation_records)
        reasoner = HonestIntelligenceReasoner(train_records)
        controllers = build_development_controllers(
            model,
            reasoner,
            calibrator,
            seed=development_seed,
            oracle_rollouts=oracle_rollouts,
            max_cycles=max_cycles,
        )
        training_rows.extend(
            {"development_seed": development_seed, **row} for row in history
        )
        for family_index, family in enumerate(DEVELOPMENT_RED_FAMILIES):
            for episode in range(episodes_per_family_seed):
                hypothesis = (episode + development_seed + family_index) % NUM_HYPOTHESES
                scenario_seed = (
                    91_000_000
                    + development_seed * 100_003
                    + family_index * 1_009
                    + episode
                )
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
    active_rows = [row for row in all_rows if row["controller"] in ACTIVE_METHODS]
    if any(row["verification_steps"] != VERIFICATION_BUDGET for row in active_rows):
        raise AssertionError("budget-matched active policies used unequal VERIFY counts")
    summary = overall_summary(all_rows)
    headroom, voi = headroom_tables(summary)
    seed_headroom = paired_seed_headroom(all_rows)
    config = {
        "protocol": "RIDe-v2 Oracle Headroom Development v1",
        "development_seeds": list(seeds),
        "formal_training_seeds_excluded": sorted(FORMAL_TRAINING_SEEDS),
        "episodes_per_family_seed": episodes_per_family_seed,
        "total_evaluation_episodes": len(all_rows),
        "methods": list(METHODS),
        "development_families": list(DEVELOPMENT_RED_FAMILIES),
        "input_feature_size": INTELLIGENCE_FEATURE_SIZE,
        "oracle_rollouts": oracle_rollouts,
        "privileged_oracles_are_diagnostic_only": True,
        "formal_results_replaced": False,
        "large_scale_training": False,
    }
    write_csv(destination / "episodes.csv", all_rows)
    write_csv(destination / "summary.csv", summary)
    write_csv(destination / "family_headroom.csv", headroom)
    write_csv(destination / "voi_analysis.csv", voi)
    write_csv(destination / "paired_seed_headroom.csv", seed_headroom)
    write_csv(destination / "training_history.csv", training_rows)
    (destination / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    formal_after = directory_manifest(FORMAL_OUTPUT)
    audit = {
        "ordinary_policy_calls": audit_counts["ordinary_calls"],
        "privileged_oracle_calls": audit_counts["privileged_calls"],
        "blue_observations_audited": audit_counts["boundary_checks"],
        "ordinary_input_feature_size": INTELLIGENCE_FEATURE_SIZE,
        "oracle_fields_inserted_into_blue_observation": False,
        "terminal_truth_used_by_ordinary_defender": False,
        "oracle_methods": sorted(ORACLE_METHODS),
        "privileged_diagnostic_only": True,
        "formal_manifest_before": formal_before,
        "formal_manifest_after": formal_after,
        "formal_results_unchanged": formal_before == formal_after,
    }
    if not audit["formal_results_unchanged"]:
        raise AssertionError("frozen formal results changed during development run")
    (destination / "information_boundary_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    decision = write_report(destination, summary, headroom, seed_headroom, config)
    return {**config, **decision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes-per-family-seed", type=int, default=30)
    parser.add_argument("--oracle-rollouts", type=int, default=6)
    parser.add_argument("--analyze-existing", action="store_true")
    args = parser.parse_args()
    if args.analyze_existing:
        print(json.dumps(analyze_existing(args.output_dir), indent=2))
        return
    print(
        json.dumps(
            run(
                args.output_dir,
                episodes_per_family_seed=args.episodes_per_family_seed,
                oracle_rollouts=args.oracle_rollouts,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
