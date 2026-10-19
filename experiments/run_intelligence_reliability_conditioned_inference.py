"""Belief-first evaluation of reliability-conditioned intelligence inference."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from deception_rl.envs.deceptive_intelligence import (
    NUM_HYPOTHESES,
    DeceptiveIntelligenceEnv,
    encode_commit_action,
)
from deception_rl.intelligence.ride import (
    HonestIntelligenceReasoner,
    TemperatureCalibrator,
)
from deception_rl.intelligence.ride_v2.calibrated_reliability import (
    CalibratedCellReliabilityFilter,
    LogisticCellModel,
    PlattCalibrator,
)
from deception_rl.intelligence.ride_v2.reliability_conditioned import (
    ReliabilityConditionedController,
    ReliabilityConditionedLikelihood,
    ReliabilityConditionedModel,
    ReliabilityLikelihoodController,
    collate_conditioned_records,
    encode_reliability_conditioned_sequence,
)
from experiments.run_intelligence_calibrated_cell_reliability import (
    DEFAULT_OUTPUT as CALIBRATED_PREDICTOR_OUTPUT,
    CALIBRATED_SEEDS,
)
from experiments.run_intelligence_calibrated_reliability_fusion import (
    DEFAULT_OUTPUT as ATTENUATION_FUSION_OUTPUT,
)
from experiments.run_intelligence_ride import (
    RED_FAMILIES,
    TRAIN_FAMILIES,
    _batches,
    _collection_action,
    collect_prefix_records,
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
    SOURCE_REGION_V1_OUTPUT,
    DEFAULT_OUTPUT as CORRECTED_SOURCE_REGION_OUTPUT,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "intelligence_reliability_conditioned_development"
CONDITIONED_SEEDS = (2222, 2323, 2424)
BELIEF_METHODS = ("multi_source", "ride_rc_encoder", "ride_rcl")


def collect_conditioned_records(
    families: tuple[str, ...],
    episodes_per_hypothesis: int,
    *,
    seed: int,
    split: str,
    max_cycles: int,
    predictor: CalibratedCellReliabilityFilter,
    reasoner: HonestIntelligenceReasoner,
) -> list[dict[str, Any]]:
    records = []
    for family_index, family in enumerate(families):
        for hypothesis in range(NUM_HYPOTHESES):
            for episode in range(episodes_per_hypothesis):
                episode_seed = seed + family_index * 100_003 + hypothesis * 1_003 + episode
                env = DeceptiveIntelligenceEnv(max_cycles=max_cycles)
                red = RED_FAMILIES[family]()
                red.reset()
                observations, _ = env.reset(seed=episode_seed, options={"hypothesis": hypothesis})
                while env.agents:
                    step = int(observations["blue"]["current_step"][0])
                    records.append(
                        {
                            "episode_id": f"{split}:{family}:{hypothesis}:{episode_seed}",
                            "family": family,
                            "sequence": encode_reliability_conditioned_sequence(
                                observations["blue"], predictor, reasoner
                            ),
                            "hypothesis_label": hypothesis,
                            "deception_label": int(family != "honest"),
                        }
                    )
                    blue_action = (
                        encode_commit_action(0)
                        if step >= max_cycles - 1
                        else _collection_action(observations["blue"], step)
                    )
                    observations, *_ = env.step(
                        {"red": red.act(observations["red"]), "blue": blue_action}
                    )
    return records


def _conditioned_loss(
    model: ReliabilityConditionedModel,
    records: list[dict[str, Any]],
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            x, lengths, hypothesis, deception = collate_conditioned_records(batch)
            hypothesis_logits, deception_logits = model(x, lengths)
            loss = nn.functional.cross_entropy(hypothesis_logits, hypothesis)
            loss += nn.functional.cross_entropy(deception_logits, deception)
            total += float(loss) * len(batch)
    return total / len(records)


def train_conditioned_model(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    *,
    epochs: int,
    hidden_size: int,
    batch_size: int,
    seed: int,
) -> ReliabilityConditionedModel:
    torch.manual_seed(seed)
    model = ReliabilityConditionedModel(hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    labels = np.asarray([row["deception_label"] for row in train_records], dtype=int)
    counts = np.bincount(labels, minlength=2)
    weights = torch.tensor(
        len(labels) / (2 * np.maximum(counts, 1)), dtype=torch.float32
    )
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    rng = random.Random(seed)
    for _ in range(epochs):
        model.train()
        for batch in _batches(train_records, batch_size, rng):
            x, lengths, hypothesis, deception = collate_conditioned_records(batch)
            optimizer.zero_grad()
            hypothesis_logits, deception_logits = model(x, lengths)
            loss = nn.functional.cross_entropy(hypothesis_logits, hypothesis)
            loss += nn.functional.cross_entropy(
                deception_logits, deception, weight=weights
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation_loss = _conditioned_loss(model, validation_records, batch_size)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model


def fit_conditioned_calibrator(
    model: ReliabilityConditionedModel,
    records: list[dict[str, Any]],
) -> TemperatureCalibrator:
    x, lengths, hypothesis, deception = collate_conditioned_records(records)
    with torch.no_grad():
        hypothesis_logits, deception_logits = model(x, lengths)
    calibrator = TemperatureCalibrator()
    calibrator.fit(
        hypothesis_logits.detach(),
        deception_logits.detach(),
        hypothesis,
        deception,
    )
    calibrator.eval()
    return calibrator


def collect_belief_cases(
    families: tuple[str, ...],
    episodes_per_family: int,
    *,
    seed: int,
    max_cycles: int,
    target_step: int = COMMIT_STEP,
) -> list[dict[str, Any]]:
    cases = []
    for family_index, family in enumerate(families):
        for episode in range(episodes_per_family):
            hypothesis = (episode + family_index) % NUM_HYPOTHESES
            episode_seed = seed + family_index * 10_003 + episode
            env = DeceptiveIntelligenceEnv(max_cycles=max_cycles)
            red = DEVELOPMENT_RED_FAMILIES[family]()
            red.reset()
            observations, _ = env.reset(seed=episode_seed, options={"hypothesis": hypothesis})
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
                    "deception_family": family,
                    "hypothesis": hypothesis,
                    "seed": episode_seed,
                    "observation": deepcopy(observations["blue"]),
                }
            )
    return cases


def fit_likelihood_temperature(
    likelihood: ReliabilityConditionedLikelihood,
    cases: list[dict[str, Any]],
    predictor: CalibratedCellReliabilityFilter,
    reasoner: HonestIntelligenceReasoner,
) -> ReliabilityConditionedLikelihood:
    logits = np.stack(
        [likelihood.logits(row["observation"], predictor, reasoner) for row in cases]
    )
    labels = np.asarray([row["hypothesis"] for row in cases], dtype=int)
    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in np.geomspace(0.1, 10.0, 101):
        scaled = logits / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        log_probability = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        nll = -float(log_probability[np.arange(len(labels)), labels].mean())
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
    return likelihood.with_temperature(best_temperature)


def _belief_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    target = np.eye(NUM_HYPOTHESES)[labels]
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == labels
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        selected = (confidence >= lower) & (confidence < lower + 0.1 + 1e-12)
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(confidence[selected].mean()) - float(correct[selected].mean())
            )
    return {
        "episodes": len(labels),
        "hypothesis_accuracy": float(correct.mean()),
        "hypothesis_brier": float(np.mean(np.sum((probabilities - target) ** 2, axis=1))),
        "hypothesis_nll": float(
            -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-8, 1)).mean()
        ),
        "hypothesis_ece": ece,
    }


def evaluate_beliefs(
    controllers: dict[str, Any], cases: list[dict[str, Any]], development_seed: int
) -> list[dict[str, Any]]:
    rows = []
    for family in ("overall", *DEVELOPMENT_RED_FAMILIES):
        selected = cases if family == "overall" else [
            row for row in cases if row["deception_family"] == family
        ]
        labels = np.asarray([row["hypothesis"] for row in selected], dtype=int)
        for method, controller in controllers.items():
            probabilities = np.stack(
                [
                    np.asarray(controller.infer(row["observation"])["hypothesis_belief"])
                    for row in selected
                ]
            )
            rows.append(
                {
                    "development_seed": development_seed,
                    "method": method,
                    "deception_family": family,
                    **_belief_metrics(probabilities, labels),
                }
            )
    return rows


def aggregate_beliefs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in BELIEF_METHODS:
        for family in ("overall", *DEVELOPMENT_RED_FAMILIES):
            selected = [
                row for row in rows if row["method"] == method and row["deception_family"] == family
            ]
            output.append(
                {
                    "method": method,
                    "deception_family": family,
                    "seeds": len(selected),
                    **{
                        metric: float(np.mean([row[metric] for row in selected]))
                        for metric in (
                            "episodes",
                            "hypothesis_accuracy",
                            "hypothesis_brier",
                            "hypothesis_nll",
                            "hypothesis_ece",
                        )
                    },
                }
            )
    return output


def write_report(
    destination: Path,
    belief_summary: list[dict[str, Any]],
    commitment_summary: list[dict[str, Any]],
    passed_methods: list[str],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    belief = {(row["method"], row["deception_family"]): row for row in belief_summary}
    baseline_brier = belief[("multi_source", "overall")]["hypothesis_brier"]
    belief_table = []
    for method in BELIEF_METHODS:
        row = belief[(method, "overall")]
        belief_table.append(
            f"| {method} | {row['hypothesis_brier']:.3f} | {row['hypothesis_ece']:.3f} | "
            f"{row['hypothesis_nll']:.3f} | {100 * row['hypothesis_accuracy']:.1f}% | "
            f"{row['hypothesis_brier'] - baseline_brier:+.3f} |"
        )
    commitment_table = []
    final_go = []
    if commitment_summary:
        commitment = {(row["method"], row["deception_family"]): row for row in commitment_summary}
        baseline = commitment[("multi_source", "overall")]
        for method in ("multi_source", *passed_methods):
            row = commitment[(method, "overall")]
            commitment_table.append(
                f"| {method} | {100 * row['correct_commitment']:.1f}% | "
                f"{row['regret']:.2f} | {row['decision_latency']:.2f} | {row['information_cost']:.2f} |"
            )
            if method != "multi_source":
                final_go.append(
                    method
                    if row["correct_commitment"] > baseline["correct_commitment"]
                    and row["regret"] < baseline["regret"]
                    else ""
                )
    successful = [method for method in final_go if method]
    report = f"""# Reliability-Conditioned Inference Development

The exact saved calibrated cell predictor coefficients and original prototypes are frozen. Original
49-dimensional observations are preserved byte-for-byte as the first encoder features;
reliability is appended as a separate condition. Linear attenuation is not used.

## Protocol

- New development seeds: {', '.join(map(str, seeds))}
- Belief gate: hypothesis Brier must be lower than Multi-source.
- Commitment gate: correct commitment must increase and regret must decrease.
- All decision controllers use the same fixed commit step; Verify is disabled in commitment evaluation.
- Belief cases use a shared fixed collection policy with verification/tracking, identical across methods.
- Raw and RC neural outputs are both evaluated without temperature scaling. RCL temperature is fitted only on separate seen-family validation cases.
- Predictor seed mapping is fixed by run order: 2222→1414, 2323→1515, 2424→1616. No predictor coefficients are refitted.

## Belief-first results

| Method | Brier | ECE | NLL | Accuracy | Brier vs Multi-source |
|---|---:|---:|---:|---:|---:|
{chr(10).join(belief_table)}

Methods passing the belief gate: {', '.join(passed_methods) if passed_methods else 'none'}.

## Matched commitment results

{('| Method | Correct commitment | Regret | Latency | Information cost |' + chr(10) + '|---|---:|---:|---:|---:|' + chr(10) + chr(10).join(commitment_table)) if commitment_table else 'Skipped because no reliability-conditioned method passed the belief gate.'}

## Decision

**{'GO' if successful else 'NO-GO'}**. Methods satisfying both belief and commitment
criteria: {', '.join(successful) if successful else 'none'}.
This is development evidence and does not replace any frozen formal result.
The RCL baseline assumes conditionally independent Gaussian report features and uses
a hypothesis-independent compromised mechanism. A constant-reliability likelihood
ablation is still needed to attribute gains specifically to learned reliability.
Calibration and gate selection are development procedures, not a formal held-out claim.
"""
    (destination / "reliability_conditioned_report.md").write_text(report, encoding="utf-8")
    result = {
        "decision": "GO" if successful else "NO-GO",
        "belief_gate_passed": passed_methods,
        "full_gate_passed": successful,
        "baseline_hypothesis_brier": baseline_brier,
    }
    (destination / "go_no_go.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    seeds: tuple[int, ...] = CONDITIONED_SEEDS,
    belief_episodes_per_family: int = 30,
    commitment_episodes_per_family: int = 30,
    epochs: int = 8,
    train_episodes_per_hypothesis: int = 8,
    validation_episodes_per_hypothesis: int = 3,
    reliability_train_episodes_per_hypothesis: int = 10,
    hidden_size: int = 32,
    batch_size: int = 64,
    max_cycles: int = 10,
) -> dict[str, Any]:
    prior_seeds = {0, 1, 2, 3, 4, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1111, 1212, 1313, 1414, 1515, 1616, 1818, 1919, 2020, 2121}
    if not seeds or set(seeds) & prior_seeds:
        raise ValueError("conditioned inference seeds must be new and disjoint")
    destination = Path(output_dir)
    frozen = {
        "formal": FORMAL_OUTPUT,
        "headroom": HEADROOM_OUTPUT,
        "source_reliability": SOURCE_RELIABILITY_OUTPUT,
        "source_region_v1": SOURCE_REGION_V1_OUTPUT,
        "source_region_corrected": CORRECTED_SOURCE_REGION_OUTPUT,
        "calibrated_predictor": CALIBRATED_PREDICTOR_OUTPUT,
        "attenuation_fusion": ATTENUATION_FUSION_OUTPUT,
    }
    if destination.resolve() in {path.resolve() for path in frozen.values()}:
        raise ValueError("conditioned inference requires an independent output")
    destination.mkdir(parents=True, exist_ok=True)
    before = {name: directory_manifest(path) for name, path in frozen.items()}
    belief_rows = []
    seed_controllers: list[tuple[int, dict[str, Any]]] = []
    for development_seed in seeds:
        random.seed(development_seed)
        np.random.seed(development_seed)
        torch.manual_seed(development_seed)
        raw_train = collect_prefix_records(
            TRAIN_FAMILIES,
            train_episodes_per_hypothesis,
            seed=300_000_000 + development_seed * 10_003,
            split=f"rc_raw_train_{development_seed}",
            max_cycles=max_cycles,
        )
        raw_validation = collect_prefix_records(
            TRAIN_FAMILIES,
            validation_episodes_per_hypothesis,
            seed=310_000_000 + development_seed * 10_003,
            split=f"rc_raw_validation_{development_seed}",
            max_cycles=max_cycles,
        )
        raw_model, _ = train_model(
            raw_train,
            raw_validation,
            epochs=epochs,
            hidden_size=hidden_size,
            batch_size=batch_size,
            seed=development_seed,
        )
        # Freeze the exact saved predictor coefficients and its original feature prototypes.
        predictor_seed = CALIBRATED_SEEDS[len(seed_controllers) % len(CALIBRATED_SEEDS)]
        payload = json.loads((CALIBRATED_PREDICTOR_OUTPUT / f"calibrated_model_seed{predictor_seed}.json").read_text())
        parameters = payload["balanced_logistic"]
        balanced = LogisticCellModel(
            np.asarray(parameters["mean"]), np.asarray(parameters["scale"]),
            np.asarray(parameters["coefficient"]), bool(parameters["balanced"]),
        )
        platt = PlattCalibrator(**payload["platt"])
        predictor = CalibratedCellReliabilityFilter(balanced, platt)
        prototype_records = collect_prefix_records(
            TRAIN_FAMILIES, 8,
            seed=210_000_000 + predictor_seed * 10_003,
            split=f"calibrated_reasoner_{predictor_seed}", max_cycles=10,
        )
        reasoner = HonestIntelligenceReasoner(prototype_records)
        reliability_train = collect_reliability_samples(
            RELIABILITY_TRAIN_FAMILIES, reliability_train_episodes_per_hypothesis,
            seed=320_000_000 + development_seed * 10_003, reasoner=reasoner,
            split=f"rc_likelihood_train_{development_seed}", max_cycles=max_cycles,
        )
        conditioned_train = collect_conditioned_records(
            TRAIN_FAMILIES,
            train_episodes_per_hypothesis,
            seed=300_000_000 + development_seed * 10_003,
            split=f"rc_conditioned_train_{development_seed}",
            max_cycles=max_cycles,
            predictor=predictor,
            reasoner=reasoner,
        )
        conditioned_validation = collect_conditioned_records(
            TRAIN_FAMILIES,
            validation_episodes_per_hypothesis,
            seed=310_000_000 + development_seed * 10_003,
            split=f"rc_conditioned_validation_{development_seed}",
            max_cycles=max_cycles,
            predictor=predictor,
            reasoner=reasoner,
        )
        conditioned_model = train_conditioned_model(
            conditioned_train,
            conditioned_validation,
            epochs=epochs,
            hidden_size=hidden_size,
            batch_size=batch_size,
            seed=development_seed,
        )
        likelihood = ReliabilityConditionedLikelihood.fit(reliability_train)
        likelihood_validation_cases = collect_belief_cases(
            RELIABILITY_TRAIN_FAMILIES,
            validation_episodes_per_hypothesis,
            seed=340_000_000 + development_seed * 10_003,
            max_cycles=max_cycles,
        )
        likelihood = fit_likelihood_temperature(
            likelihood, likelihood_validation_cases, predictor, reasoner
        )
        controllers = {
            "multi_source": _inference(
                raw_model,
                reasoner,
                None,
                "multi_source",
                full_ride=False,
                max_cycles=max_cycles,
            ),
            "ride_rc_encoder": ReliabilityConditionedController(
                conditioned_model, predictor, reasoner, None
            ),
            "ride_rcl": ReliabilityLikelihoodController(
                likelihood, predictor, reasoner
            ),
        }
        cases = collect_belief_cases(
            tuple(DEVELOPMENT_RED_FAMILIES),
            belief_episodes_per_family,
            seed=350_000_000 + development_seed * 10_003,
            max_cycles=max_cycles,
        )
        belief_rows.extend(evaluate_beliefs(controllers, cases, development_seed))
        seed_controllers.append((development_seed, controllers))
    belief_summary = aggregate_beliefs(belief_rows)
    belief_lookup = {(row["method"], row["deception_family"]): row for row in belief_summary}
    baseline_brier = belief_lookup[("multi_source", "overall")]["hypothesis_brier"]
    passed_methods = [
        method
        for method in BELIEF_METHODS[1:]
        if belief_lookup[(method, "overall")]["hypothesis_brier"] < baseline_brier
    ]
    commitment_rows = []
    if passed_methods:
        for development_seed, controllers in seed_controllers:
            selected_controllers = {
                method: controllers[method] for method in ("multi_source", *passed_methods)
            }
            for family_index, family in enumerate(DEVELOPMENT_RED_FAMILIES):
                for episode in range(commitment_episodes_per_family):
                    hypothesis = (episode + development_seed + family_index) % NUM_HYPOTHESES
                    scenario_seed = 360_000_000 + development_seed * 100_003 + family_index * 1_009 + episode
                    for method, controller in selected_controllers.items():
                        row, _ = evaluate_development_episode(
                            controller,
                            family,
                            hypothesis,
                            seed=scenario_seed,
                            max_cycles=max_cycles,
                        )
                        row["development_seed"] = development_seed
                        if row["controller"] != method:
                            raise AssertionError("controller name drifted")
                        commitment_rows.append(row)
    commitment_summary = []
    if commitment_rows:
        commitment_summary = summarize(commitment_rows)
        for method in ("multi_source", *passed_methods):
            selected = [row for row in commitment_rows if row["controller"] == method]
            commitment_summary.extend(
                summarize([{**row, "deception_family": "overall"} for row in selected])
            )
        commitment_summary = sorted(
            commitment_summary, key=lambda row: (row["deception_family"], row["method"])
        )
    write_csv(destination / "belief_metrics_by_seed.csv", belief_rows)
    write_csv(destination / "belief_summary.csv", belief_summary)
    if commitment_rows:
        write_csv(destination / "commitment_episodes.csv", commitment_rows)
        write_csv(destination / "commitment_summary.csv", commitment_summary)
    after = {name: directory_manifest(path) for name, path in frozen.items()}
    audit = {
        "original_49_features_preserved": True,
        "reliability_appended_as_condition": True,
        "linear_attenuation_used": False,
        "saved_predictor_coefficients_loaded_without_refit": True,
        "simulator_labels_used_at_runtime": False,
        "verification_enabled": False,
        **{f"{name}_results_unchanged": before[name] == after[name] for name in frozen},
    }
    if not all(value for key, value in audit.items() if key.endswith("_results_unchanged")):
        raise AssertionError("a frozen result directory changed")
    (destination / "information_boundary_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    config = {
        "protocol": "Reliability-Conditioned Inference Belief-First Development v1",
        "development_seeds": list(seeds),
        "belief_episodes_per_family": belief_episodes_per_family,
        "commitment_episodes_per_family": commitment_episodes_per_family,
        "belief_methods": list(BELIEF_METHODS),
        "epochs": epochs,
        "hidden_size": hidden_size,
        "train_episodes_per_hypothesis": train_episodes_per_hypothesis,
        "validation_episodes_per_hypothesis": validation_episodes_per_hypothesis,
        "neural_temperature_scaling": False,
        "saved_predictor_seed_mapping": {str(seed): CALIBRATED_SEEDS[index % len(CALIBRATED_SEEDS)] for index, seed in enumerate(seeds)},
        "passed_belief_gate": passed_methods,
        "verification_enabled": False,
        "formal_results_replaced": False,
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return {
        **config,
        **write_report(destination, belief_summary, commitment_summary, passed_methods, seeds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
