"""Train a decision-only imitation policy and run a 512-episode smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from deception_rl.agents import ScriptedDeception
from deception_rl.agents.decoupled_defender import DecoupledAntiDeception
from deception_rl.beliefs.decoupled import (
    CalibratedGoalInference,
    EvidenceCalibrator,
    LikelihoodRatioReasoner,
)
from deception_rl.beliefs.two_level import DeceptionAwareBelief
from experiments.run_stage11_adaptive_attacker import run_episode, summarize
from experiments.stage11_adaptive_deceiver import AdaptiveDeceiver
from experiments.stage13_learned_controller import (
    ACTION_INDEX,
    LearnedDecisionController,
    decision_features,
)
from experiments.train_learned_defender import write_csv
from models.anti_deception_gru import AntiDeceptionGRU
from models.stage13_decision import ACTION_NAMES, INPUT_NAMES, Stage13DecisionMLP


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "stage13_learned_controller"
TEACHER_EPISODES = Path("/workspace/stage10-1-decoupled/evaluation_episodes.csv")


def load_inference():
    stage10 = Path("/workspace/stage10-two-level")
    stage101 = Path("/workspace/stage10-1-decoupled")
    checkpoint = torch.load(stage10 / "belief_model.pt", map_location="cpu", weights_only=True)
    model = AntiDeceptionGRU(checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    calibrator = EvidenceCalibrator()
    calibrator.load_state_dict(
        torch.load(stage101 / "calibrator.pt", map_location="cpu", weights_only=True)[
            "model_state_dict"
        ]
    )
    calibrator.eval()
    for module in (model, calibrator):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    records = [json.loads(line) for line in (stage10 / "train.jsonl").read_text().splitlines()]
    return CalibratedGoalInference(
        DeceptionAwareBelief(model), LikelihoodRatioReasoner(records), calibrator
    )


def teacher_dataset(path=TEACHER_EPISODES):
    episodes = []
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            if row["controller"] != "stage10_1":
                continue
            trace = json.loads(row["trace"])
            episode_id = f"{row['group']}:{row['episode']}:{row['family']}:{row['true_goal']}"
            cumulative_cost = 0.0
            committed = False
            samples = []
            for step in trace:
                info = step["info"]
                action_name = step["action"]
                features = decision_features(info, cumulative_cost, committed)
                utilities = info.get("action_utilities")
                if utilities:
                    ordered = np.asarray(
                        [utilities[name] for name in ACTION_NAMES], dtype=np.float32
                    )
                    regret = float(ordered.max()) - ordered
                else:
                    # One post-commit state teaches the status bit and irreversible
                    # action. It is a terminal constraint, not a new reward target.
                    regret = np.ones(4, dtype=np.float32)
                    regret[ACTION_INDEX[action_name]] = 0.0
                samples.append(
                    {
                        "episode_id": episode_id,
                        "family": row["family"],
                        "goal": row["true_goal"],
                        "timestep": step["timestep"],
                        "features": features,
                        "action": ACTION_INDEX[action_name],
                        "regret": regret,
                    }
                )
                if action_name == "observe":
                    cumulative_cost += 0.01
                elif action_name == "track":
                    cumulative_cost += 0.03
                elif action_name.startswith("commit_"):
                    if committed:
                        break
                    committed = True
            episodes.append({"episode_id": episode_id, "samples": samples})
    if not episodes:
        raise ValueError("No Stage10.1 teacher episodes found")
    return episodes


def split_episodes(episodes, seed=0):
    rng = random.Random(seed)
    shuffled = list(episodes)
    rng.shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * 0.2))
    validation = shuffled[:validation_count]
    training = shuffled[validation_count:]
    assert not ({row["episode_id"] for row in training} & {row["episode_id"] for row in validation})
    return training, validation


def tensors(episodes):
    samples = [sample for episode in episodes for sample in episode["samples"]]
    features = torch.tensor(np.stack([sample["features"] for sample in samples]))
    actions = torch.tensor([sample["action"] for sample in samples], dtype=torch.long)
    regrets = torch.tensor(np.stack([sample["regret"] for sample in samples]))
    return samples, features, actions, regrets


def policy_metrics(model, features, actions, regrets):
    model.eval()
    with torch.inference_mode():
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1)
        cross_entropy = nn.functional.cross_entropy(logits, actions)
        expected_regret = (probabilities * regrets).sum(dim=1).mean()
        predictions = logits.argmax(dim=1)
    return {
        "cross_entropy": float(cross_entropy),
        "expected_regret": float(expected_regret),
        "imitation_accuracy": float((predictions == actions).float().mean()),
    }


def train_controller(destination, seed=0, epochs=200, regret_weight=1.0):
    episodes = teacher_dataset()
    training_episodes, validation_episodes = split_episodes(episodes, seed)
    train_samples, train_x, train_y, train_regret = tensors(training_episodes)
    val_samples, val_x, val_y, val_regret = tensors(validation_episodes)
    torch.manual_seed(seed)
    model = Stage13DecisionMLP(32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history = []
    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(train_x)
        probabilities = torch.softmax(logits, dim=1)
        imitation_loss = nn.functional.cross_entropy(logits, train_y)
        regret_loss = (probabilities * train_regret).sum(dim=1).mean()
        loss = imitation_loss + regret_weight * regret_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        validation = policy_metrics(model, val_x, val_y, val_regret)
        history.append(
            {
                "epoch": epoch,
                "training_imitation_loss": float(imitation_loss.detach()),
                "training_regret_loss": float(regret_loss.detach()),
                "training_total_loss": float(loss.detach()),
                **{f"validation_{key}": value for key, value in validation.items()},
            }
        )
        score = validation["cross_entropy"] + regret_weight * validation["expected_regret"]
        if score < best:
            best = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_size": 32,
                    "epoch": epoch,
                    "input_names": INPUT_NAMES,
                    "action_names": ACTION_NAMES,
                    "regret_weight": regret_weight,
                    "seed": seed,
                },
                destination / "stage13_controller.pt",
            )
    saved = torch.load(destination / "stage13_controller.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    write_csv(destination / "training_metrics.csv", history)
    train_result = policy_metrics(model, train_x, train_y, train_regret)
    validation_result = policy_metrics(model, val_x, val_y, val_regret)
    metadata = {
        "source": str(TEACHER_EPISODES),
        "source_sha256": hashlib.sha256(TEACHER_EPISODES.read_bytes()).hexdigest(),
        "teacher_episodes": len(episodes),
        "training_episodes": len(training_episodes),
        "validation_episodes": len(validation_episodes),
        "training_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "training_action_counts": dict(Counter(ACTION_NAMES[s["action"]] for s in train_samples)),
        "validation_action_counts": dict(Counter(ACTION_NAMES[s["action"]] for s in val_samples)),
        "selected_epoch": saved["epoch"],
        "epochs": epochs,
        "loss": "cross_entropy + 1.0 * expected_teacher_utility_regret",
        "post_commit": "at most one terminal-status sample per episode",
        "train": train_result,
        "validation": validation_result,
        "limitations": "episode-ID split; deterministic route aliases can overlap across splits",
    }
    (destination / "imitation_metrics.json").write_text(json.dumps(metadata, indent=2))
    return model, metadata


def schedule(seed=0):
    scenarios = []
    for attack in ("scripted", "adaptive_attacker"):
        for index in range(128):
            scenarios.append(
                {
                    "attack": attack,
                    "goal": "A" if index % 2 == 0 else "B",
                    "seed": seed + 2_100_000 + len(scenarios),
                    "scenario": len(scenarios),
                    "scope": "main",
                }
            )
    assert len(scenarios) * 2 == 512
    return scenarios


def render_report(destination, summary, metadata, failures):
    result = {(row["controller"], row["attack"]): row for row in summary}
    lines = [
        "# Stage 13 Learned Decision Controller",
        "",
        "## Method",
        "",
        "The Stage10.1 dual-head GRU, calibrator and counterfactual likelihood module are frozen. A 1,412-parameter MLP maps [P(A), P(B), entropy, tanh(log likelihood-ratio), cumulative observation cost, commitment status] to observe/track/commit_A/commit_B.",
        "",
        "Offline teacher data comes from the existing Stage10.1 evaluation artifact. Training uses cross-entropy imitation plus expected teacher-utility regret under the learned action distribution. Splitting is by episode ID; deterministic trajectory aliases still cross splits, so validation is an imitation diagnostic rather than independent generalization evidence.",
        "",
        f"Selected epoch {metadata['selected_epoch']}/{metadata['epochs']}; validation imitation accuracy {metadata['validation']['imitation_accuracy']:.2%}, expected utility regret {metadata['validation']['expected_regret']:.4f}. No simulator episode or Deceiver training is used for imitation.",
        "",
        "## 512-episode smoke",
        "",
        "256 paired scenarios × Stage10.1/Stage13 =512 environment episodes: ScriptedDeception(steps=3) 128 and AdaptiveDeceiver128, each A/B balanced. The online attacker keeps a separate profile for each Defender.",
        "",
        "| Controller | Attack | Interception | False commitment | Observation cost | Regret | Latency |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"|{row['controller']}|{row['attack']}|{row['interception_success']:.2%}|"
            f"{row['false_commitment_rate']:.2%}|{row['mean_observation_cost']:.4f}|"
            f"{row['decision_regret']:.4f}|{row['decision_latency']:.2f}|"
        )
    teacher_adaptive = result[("stage10_1", "adaptive_attacker")]
    learned_adaptive = result[("stage13_learned", "adaptive_attacker")]
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"On AdaptiveDeceiver, learned-minus-teacher interception is {(learned_adaptive['interception_success'] - teacher_adaptive['interception_success']) * 100:+.2f} percentage points, false commitment is {(learned_adaptive['false_commitment_rate'] - teacher_adaptive['false_commitment_rate']) * 100:+.2f} points, and regret is {learned_adaptive['decision_regret'] - teacher_adaptive['decision_regret']:+.4f}.",
            "",
            "This tests whether a small network reproduces Stage10.1 decisions under an unseen online attacker. It does not retrain the belief representation and is not a policy-gradient or PPO experiment.",
            "",
            "AdaptiveDeceiver receives each Defender's completed action history, so paired seed/goal does not imply identical attacker trajectories. The lower learned-controller cost is a closed-loop outcome, not an offline comparison on a fixed trajectory.",
            "",
            f"Failure examples saved: {len(failures)}. Training curves, checkpoint, evaluation episodes and summaries are in this directory.",
            "",
            "Stopped after the fixed smoke; no additional training was started.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (destination / "stage13_report.md").write_text(report)
    (destination / "comparison_table.md").write_text("\n".join(lines[10:]) + "\n")


def run(destination=OUTPUT, seed=0):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    protected = [
        ROOT / "src/deception_rl/envs/gridworld.py",
        ROOT / "experiments/stage11_adaptive_deceiver.py",
        ROOT / "src/deception_rl/beliefs/decoupled.py",
        ROOT / "src/deception_rl/beliefs/two_level.py",
        ROOT / "src/deception_rl/agents/decoupled_defender.py",
        ROOT / "models/anti_deception_gru.py",
        Path("/workspace/stage10-two-level/belief_model.pt"),
        Path("/workspace/stage10-1-decoupled/calibrator.pt"),
        TEACHER_EPISODES,
    ]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    model, metadata = train_controller(destination, seed)
    baseline_inference = load_inference()
    learned_inference = load_inference()
    controllers = {
        "stage10_1": DecoupledAntiDeception(baseline_inference),
        "stage13_learned": LearnedDecisionController(learned_inference, model),
    }
    attackers = {name: AdaptiveDeceiver(seed) for name in controllers}
    rows = []
    for scenario in schedule(seed):
        for name, controller in controllers.items():
            attacker = (
                ScriptedDeception(3)
                if scenario["attack"] == "scripted"
                else attackers[name]
            )
            rows.append(run_episode(controller, attacker, name, **scenario))
    assert len(rows) == 512
    summary = []
    for name in controllers:
        for attack in ("scripted", "adaptive_attacker"):
            selected = [
                row for row in rows if row["controller"] == name and row["attack"] == attack
            ]
            summary.append({"controller": name, "attack": attack, **summarize(selected)})
    failures = []
    for name in controllers:
        for attack in ("scripted", "adaptive_attacker"):
            selected = [
                row
                for row in rows
                if row["controller"] == name
                and row["attack"] == attack
                and not row["interception_success"]
            ]
            failures.extend(selected[:2])
    write_csv(destination / "episodes.csv", rows)
    write_csv(destination / "summary.csv", summary)
    (destination / "failure_cases.json").write_text(json.dumps(failures, indent=2))
    config = {
        "seed": seed,
        "offline_teacher_episodes": metadata["teacher_episodes"],
        "offline_training_simulator_episodes": 0,
        "evaluation_episodes": 512,
        "paired_scenarios": 256,
        "scripted_scenarios": 128,
        "adaptive_attacker_scenarios": 128,
        "goal_balance": "64 A / 64 B within each attack",
        "input_names": INPUT_NAMES,
        "action_names": ACTION_NAMES,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "belief_encoder_frozen": True,
        "deceiver_training": False,
        "adaptive_comparison": "paired seed/goal, but attacker trajectories may differ because the attacker responds to each Defender",
        "selection": "best offline validation composite loss; no evaluation tuning",
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    checks = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() == before[str(path)]
        for path in protected
    }
    assert all(checks.values())
    (destination / "preserved_inputs.json").write_text(
        json.dumps({"sha256_before": before, "unchanged": checks}, indent=2)
    )
    render_report(destination, summary, metadata, failures)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.seed), indent=2))

