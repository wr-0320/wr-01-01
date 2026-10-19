"""Small RIDe transfer, calibration, and ablation on intelligence reports."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from deception_rl.agents.intelligence_policies import (
    AdaptiveDeceptionRedPolicy,
    ConcealmentRedPolicy,
    DecoyRedPolicy,
    DelayedDeceptionRedPolicy,
    FixedTimeBluePolicy,
    HonestRedPolicy,
    SwitchDeceptionRedPolicy,
)
from deception_rl.envs.deceptive_intelligence import (
    BLUE_WAIT,
    NUM_HYPOTHESES,
    NUM_REGIONS,
    NUM_SOURCES,
    DeceptiveIntelligenceEnv,
    encode_commit_action,
    encode_track_action,
    encode_verify_action,
)
from deception_rl.intelligence.adapter import (
    audit_blue_observation,
    encode_intelligence_sequence,
)
from deception_rl.intelligence.baselines import (
    ActiveInferenceController,
    BayesianFusionController,
    POMCPLikeController,
)
from deception_rl.intelligence.ride import (
    ABLATION_VARIANTS,
    HonestIntelligenceReasoner,
    IntelligenceRIDeModel,
    RIDeIntelligenceController,
    TemperatureCalibrator,
    collate_intelligence_records,
)


RedFactory = Callable[[], Any]
RED_FAMILIES: dict[str, RedFactory] = {
    "honest": HonestRedPolicy,
    "concealment": ConcealmentRedPolicy,
    "decoy": DecoyRedPolicy,
    "multi_source": lambda: DecoyRedPolicy(multisource=True),
    "delayed": lambda: DelayedDeceptionRedPolicy(onset=3),
    "switch": lambda: SwitchDeceptionRedPolicy(switch_step=3),
    "adaptive": AdaptiveDeceptionRedPolicy,
}
TRAIN_FAMILIES = ("honest", "concealment", "decoy", "delayed")
UNSEEN_FAMILIES = ("multi_source", "switch", "adaptive")


def _collection_action(observation: dict[str, Any], step: int) -> int:
    if step % 4 == 1 and int(observation["remaining_verification"][0]) > 0:
        return encode_verify_action(step % NUM_SOURCES, (step // 2) % NUM_REGIONS)
    if step % 4 == 2 and int(observation["remaining_tracking"][0]) > 0:
        return encode_track_action(step % NUM_REGIONS)
    return BLUE_WAIT


def collect_prefix_records(
    families: tuple[str, ...],
    episodes_per_hypothesis: int,
    *,
    seed: int,
    split: str,
    max_cycles: int,
) -> list[dict[str, Any]]:
    """Collect only blue-visible prefixes; labels stay in the offline harness."""

    records = []
    for family_index, family in enumerate(families):
        for hypothesis in range(NUM_HYPOTHESES):
            for episode in range(episodes_per_hypothesis):
                episode_seed = (
                    seed
                    + family_index * 100_003
                    + hypothesis * 1_003
                    + episode
                )
                episode_id = f"{split}:{family}:{hypothesis}:{episode_seed}"
                env = DeceptiveIntelligenceEnv(max_cycles=max_cycles)
                red = RED_FAMILIES[family]()
                red.reset()
                observations, _ = env.reset(
                    seed=episode_seed, options={"hypothesis": hypothesis}
                )
                while env.agents:
                    step = int(observations["blue"]["current_step"][0])
                    audit_blue_observation(observations["blue"])
                    records.append(
                        {
                            "episode_id": episode_id,
                            "family": family,
                            "sequence": encode_intelligence_sequence(
                                observations["blue"]
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
                        {
                            "red": red.act(observations["red"]),
                            "blue": blue_action,
                        }
                    )
    return records


def _batches(
    records: list[dict[str, Any]], batch_size: int, rng: random.Random
) -> list[list[dict[str, Any]]]:
    order = list(range(len(records)))
    rng.shuffle(order)
    return [
        [records[index] for index in order[start : start + batch_size]]
        for start in range(0, len(order), batch_size)
    ]


def _loss_and_accuracy(
    model: IntelligenceRIDeModel, records: list[dict[str, Any]], batch_size: int
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    hypothesis_correct = 0
    deception_correct = 0
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            x, lengths, hypothesis, deception = collate_intelligence_records(batch)
            hypothesis_logits, deception_logits = model(x, lengths)
            loss = nn.functional.cross_entropy(
                hypothesis_logits, hypothesis
            ) + nn.functional.cross_entropy(deception_logits, deception)
            total_loss += float(loss) * len(batch)
            hypothesis_correct += int(
                (hypothesis_logits.argmax(1) == hypothesis).sum()
            )
            deception_correct += int((deception_logits.argmax(1) == deception).sum())
    return {
        "loss": total_loss / len(records),
        "hypothesis_accuracy": hypothesis_correct / len(records),
        "deception_accuracy": deception_correct / len(records),
    }


def train_model(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    *,
    epochs: int,
    hidden_size: int,
    batch_size: int,
    seed: int,
) -> tuple[IntelligenceRIDeModel, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    model = IntelligenceRIDeModel(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    deception_labels = np.array(
        [row["deception_label"] for row in train_records], dtype=int
    )
    counts = np.bincount(deception_labels, minlength=2)
    deception_weights = torch.tensor(
        len(deception_labels) / (2 * np.maximum(counts, 1)), dtype=torch.float32
    )
    history = []
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = 0.0
        for batch in _batches(train_records, batch_size, rng):
            x, lengths, hypothesis, deception = collate_intelligence_records(batch)
            optimizer.zero_grad()
            hypothesis_logits, deception_logits = model(x, lengths)
            loss = nn.functional.cross_entropy(
                hypothesis_logits, hypothesis
            ) + nn.functional.cross_entropy(
                deception_logits, deception, weight=deception_weights
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_loss += float(loss.detach()) * len(batch)
        validation = _loss_and_accuracy(model, validation_records, batch_size)
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss / len(train_records),
                "validation_loss": validation["loss"],
                "validation_hypothesis_accuracy": validation[
                    "hypothesis_accuracy"
                ],
                "validation_deception_accuracy": validation["deception_accuracy"],
            }
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def fit_calibrator(
    model: IntelligenceRIDeModel, validation_records: list[dict[str, Any]]
) -> tuple[TemperatureCalibrator, dict[str, float]]:
    x, lengths, hypothesis, deception = collate_intelligence_records(
        validation_records
    )
    with torch.no_grad():
        hypothesis_logits, deception_logits = model(x, lengths)
    calibrator = TemperatureCalibrator()
    metrics = calibrator.fit(
        hypothesis_logits.detach(),
        deception_logits.detach(),
        hypothesis,
        deception,
    )
    calibrator.eval()
    return calibrator, metrics


def build_controllers(
    model: IntelligenceRIDeModel,
    reasoner: HonestIntelligenceReasoner,
    calibrator: TemperatureCalibrator,
    max_cycles: int,
    seed: int,
) -> dict[str, Any]:
    controllers: dict[str, Any] = {}
    for variant in ABLATION_VARIANTS:
        controllers[variant.name] = RIDeIntelligenceController(
            model, reasoner, calibrator, variant, max_cycles=max_cycles
        )
    rule = FixedTimeBluePolicy(commit_step=4)
    rule.name = "rule_fixed_time"
    controllers[rule.name] = rule
    for controller in (
        BayesianFusionController(max_cycles=max_cycles),
        POMCPLikeController(max_cycles=max_cycles, seed=seed),
        ActiveInferenceController(max_cycles=max_cycles),
    ):
        controllers[controller.name] = controller
    return controllers


def _controller_info(controller: Any) -> dict[str, Any]:
    if getattr(controller, "info", None) is not None:
        return dict(controller.info)
    belief = getattr(controller, "last_belief", None)
    if belief is None:
        return {}
    return {
        "hypothesis_belief": belief.hypothesis_probabilities.tolist(),
        "predicted_hypothesis": int(
            np.argmax(belief.hypothesis_probabilities)
        ),
        "confidence": belief.confidence,
        "entropy": belief.entropy,
        "deception_probability": belief.deception_probability,
    }


def evaluate_episode(
    controller: Any,
    family: str,
    hypothesis: int,
    *,
    seed: int,
    max_cycles: int,
) -> dict[str, Any]:
    env = DeceptiveIntelligenceEnv(max_cycles=max_cycles)
    red = RED_FAMILIES[family]()
    red.reset()
    controller.reset()
    observations, _ = env.reset(seed=seed, options={"hypothesis": hypothesis})
    boundary_checks = 0
    metrics = None
    while env.agents:
        audit_blue_observation(observations["blue"])
        boundary_checks += 1
        blue_action = controller.act(observations["blue"])
        observations, _, _, _, infos = env.step(
            {"red": red.act(observations["red"]), "blue": blue_action}
        )
        if "episode_metrics" in infos["blue"]:
            metrics = infos["blue"]["episode_metrics"]
    if metrics is None:
        raise RuntimeError("evaluation episode ended without metrics")
    info = _controller_info(controller)
    belief = np.asarray(
        info.get("hypothesis_belief", np.full(NUM_HYPOTHESES, 1 / NUM_HYPOTHESES)),
        dtype=float,
    )
    target = np.zeros(NUM_HYPOTHESES)
    target[hypothesis] = 1
    deception_probability = float(info.get("deception_probability", 0.5))
    deception_label = int(family != "honest")
    true_hypothesis_probability = float(np.clip(belief[hypothesis], 1e-8, 1))
    clipped_deception_probability = float(
        np.clip(deception_probability, 1e-8, 1 - 1e-8)
    )
    return {
        "controller": controller.name,
        "deception_family": family,
        "family_scope": "seen" if family in TRAIN_FAMILIES else "unseen",
        "hypothesis": hypothesis,
        "seed": seed,
        "interception": metrics["decision_correct"],
        "false_commitment": int(
            not metrics["timed_out"] and not metrics["decision_correct"]
        ),
        "decision_latency": metrics["decision_step"],
        "information_cost": metrics["blue_information_cost"],
        "regret": metrics["decision_regret"],
        "timed_out": metrics["timed_out"],
        "confidence": float(info.get("confidence", belief.max())),
        "deception_probability": deception_probability,
        "deception_label": deception_label,
        "hypothesis_probability_true": true_hypothesis_probability,
        "hypothesis_nll": float(-np.log(true_hypothesis_probability)),
        "deception_nll": float(
            -np.log(
                clipped_deception_probability
                if deception_label
                else 1 - clipped_deception_probability
            )
        ),
        "hypothesis_brier": float(np.sum((belief - target) ** 2)),
        "deception_brier": float(
            (deception_probability - deception_label) ** 2
        ),
        "counterfactual_score": info.get("counterfactual_deception_score", ""),
        "boundary_checks": boundary_checks,
    }


def _calibration_error(rows: list[dict[str, Any]], bins: int = 5) -> float:
    confidence = np.asarray([row["confidence"] for row in rows], dtype=float)
    correct = np.asarray([row["interception"] for row in rows], dtype=float)
    error = 0.0
    for lower in np.linspace(0, 1, bins + 1)[:-1]:
        upper = lower + 1 / bins
        selected = (confidence >= lower) & (
            confidence <= upper if upper >= 1 else confidence < upper
        )
        if selected.any():
            error += float(selected.mean()) * abs(
                float(confidence[selected].mean()) - float(correct[selected].mean())
            )
    return error


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["controller"], row["deception_family"])].append(row)
    output = []
    for (controller, family), selected in sorted(groups.items()):
        output.append(
            {
                "controller": controller,
                "deception_family": family,
                "family_scope": selected[0]["family_scope"],
                "episodes": len(selected),
                "interception": float(
                    np.mean([row["interception"] for row in selected])
                ),
                "false_commitment": float(
                    np.mean([row["false_commitment"] for row in selected])
                ),
                "decision_latency": float(
                    np.mean([row["decision_latency"] for row in selected])
                ),
                "information_cost": float(
                    np.mean([row["information_cost"] for row in selected])
                ),
                "regret": float(np.mean([row["regret"] for row in selected])),
                "calibration_error": _calibration_error(selected),
                "hypothesis_brier": float(
                    np.mean([row["hypothesis_brier"] for row in selected])
                ),
                "deception_brier": float(
                    np.mean([row["deception_brier"] for row in selected])
                ),
            }
        )
    return output


def summarize_ablation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate the ordered RIDe component ablation over seen/unseen scopes."""

    variant_names = [variant.name for variant in ABLATION_VARIANTS]
    output = []
    for order, controller in enumerate(variant_names, start=1):
        controller_rows = [row for row in rows if row["controller"] == controller]
        for scope in ("overall", "seen", "unseen"):
            selected = (
                controller_rows
                if scope == "overall"
                else [row for row in controller_rows if row["family_scope"] == scope]
            )
            output.append(
                {
                    "component_order": order,
                    "controller": controller,
                    "scope": scope,
                    "episodes": len(selected),
                    "interception": float(
                        np.mean([row["interception"] for row in selected])
                    ),
                    "false_commitment": float(
                        np.mean([row["false_commitment"] for row in selected])
                    ),
                    "decision_latency": float(
                        np.mean([row["decision_latency"] for row in selected])
                    ),
                    "information_cost": float(
                        np.mean([row["information_cost"] for row in selected])
                    ),
                    "regret": float(
                        np.mean([row["regret"] for row in selected])
                    ),
                    "calibration_error": _calibration_error(selected),
                    "hypothesis_brier": float(
                        np.mean([row["hypothesis_brier"] for row in selected])
                    ),
                    "deception_brier": float(
                        np.mean([row["deception_brier"] for row in selected])
                    ),
                }
            )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    output_dir: str | Path,
    *,
    seed: int = 0,
    epochs: int = 6,
    train_episodes_per_hypothesis: int = 4,
    validation_episodes_per_hypothesis: int = 2,
    evaluation_episodes_per_hypothesis: int = 2,
    hidden_size: int = 32,
    batch_size: int = 64,
    max_cycles: int = 10,
) -> list[dict[str, Any]]:
    if min(
        epochs,
        train_episodes_per_hypothesis,
        validation_episodes_per_hypothesis,
        evaluation_episodes_per_hypothesis,
    ) < 1:
        raise ValueError("training and episode counts must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    train_records = collect_prefix_records(
        TRAIN_FAMILIES,
        train_episodes_per_hypothesis,
        seed=seed + 10_000,
        split="train",
        max_cycles=max_cycles,
    )
    validation_records = collect_prefix_records(
        TRAIN_FAMILIES,
        validation_episodes_per_hypothesis,
        seed=seed + 20_000,
        split="validation",
        max_cycles=max_cycles,
    )
    assert {row["episode_id"] for row in train_records}.isdisjoint(
        {row["episode_id"] for row in validation_records}
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
    controllers = build_controllers(
        model, reasoner, calibrator, max_cycles=max_cycles, seed=seed
    )

    rows = []
    scenario = 0
    for family_index, family in enumerate(RED_FAMILIES):
        for hypothesis in range(NUM_HYPOTHESES):
            for episode in range(evaluation_episodes_per_hypothesis):
                episode_seed = (
                    seed
                    + 30_000
                    + family_index * 10_003
                    + hypothesis * 103
                    + episode
                )
                for controller in controllers.values():
                    rows.append(
                        evaluate_episode(
                            controller,
                            family,
                            hypothesis,
                            seed=episode_seed,
                            max_cycles=max_cycles,
                        )
                    )
                scenario += 1
    summary = summarize(rows)
    _write_csv(destination / "episodes.csv", rows)
    _write_csv(destination / "summary_by_deception_family.csv", summary)
    _write_csv(destination / "ablation_overall.csv", summarize_ablation(rows))
    _write_csv(destination / "training_history.csv", history)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "calibrator_state_dict": calibrator.state_dict(),
            "hidden_size": hidden_size,
            "hypotheses": NUM_HYPOTHESES,
        },
        destination / "intelligence_ride_smoke.pt",
    )
    (destination / "calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    boundary = {
        "adapter_audit": audit_blue_observation(
            DeceptiveIntelligenceEnv(max_cycles=max_cycles).reset(seed=seed)[0][
                "blue"
            ]
        ),
        "evaluated_policy_inputs": int(
            sum(row["boundary_checks"] for row in rows)
        ),
        "terminal_truth_used_as_policy_input": False,
        "training_labels": "offline harness only; never concatenated to sequence features",
    }
    (destination / "information_boundary_audit.json").write_text(
        json.dumps(boundary, indent=2), encoding="utf-8"
    )
    config = {
        "seed": seed,
        "scope": "small transfer smoke and ablation",
        "epochs": epochs,
        "hidden_size": hidden_size,
        "train_families": list(TRAIN_FAMILIES),
        "unseen_evaluation_families": list(UNSEEN_FAMILIES),
        "train_episode_ids": len({row["episode_id"] for row in train_records}),
        "validation_episode_ids": len(
            {row["episode_id"] for row in validation_records}
        ),
        "evaluation_scenarios": scenario,
        "evaluation_episodes": len(rows),
        "controllers": list(controllers),
        "large_scale_training": False,
        "legacy_results_replaced": False,
        "pomcp_like_limit": "one-step categorical Monte Carlo belief rollout",
        "interception_definition": "correct terminal operational hypothesis commitment",
    }
    (destination / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/intelligence_ride_smoke")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--train-episodes-per-hypothesis", type=int, default=4)
    parser.add_argument("--validation-episodes-per-hypothesis", type=int, default=2)
    parser.add_argument("--evaluation-episodes-per-hypothesis", type=int, default=2)
    args = parser.parse_args()
    summary = run(
        args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        train_episodes_per_hypothesis=args.train_episodes_per_hypothesis,
        validation_episodes_per_hypothesis=args.validation_episodes_per_hypothesis,
        evaluation_episodes_per_hypothesis=args.evaluation_episodes_per_hypothesis,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

