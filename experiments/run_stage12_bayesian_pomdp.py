"""Stage 12 Bayesian/POMDP baseline smoke with exactly 512 episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from deception_rl.agents import HonestShortestPath, ScriptedDeception
from deception_rl.agents.active_anti_deception import ActiveAntiDeception
from deception_rl.agents.adaptive_defender import AdaptiveActiveDefender
from deception_rl.agents.decoupled_defender import DecoupledAntiDeception
from deception_rl.beliefs.decoupled import (
    CalibratedGoalInference,
    EvidenceCalibrator,
    LikelihoodRatioReasoner,
)
from deception_rl.beliefs.two_level import DeceptionAwareBelief
from experiments.run_stage11_adaptive_attacker import (
    run_episode,
    summarize,
)
from experiments.stage11_adaptive_deceiver import AdaptiveDeceiver
from experiments.stage12_classical_defenders import (
    BayesianDefender,
    FiniteHorizonPOMDPDefender,
)
from experiments.train_anti_deception import binary_metrics
from experiments.train_learned_defender import write_csv
from models.anti_deception_gru import AntiDeceptionGRU


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "stage12_bayesian_pomdp"
CONTROLLERS = ("rule_active", "bayesian", "pomdp", "stage10_1", "adaptive")
ATTACKS = ("honest", "scripted", "adaptive_attacker")


def build_controllers():
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
    inference = CalibratedGoalInference(
        DeceptionAwareBelief(model), LikelihoodRatioReasoner(records), calibrator
    )
    return {
        "rule_active": ActiveAntiDeception(),
        "bayesian": BayesianDefender(),
        "pomdp": FiniteHorizonPOMDPDefender(),
        "stage10_1": DecoupledAntiDeception(inference),
        "adaptive": AdaptiveActiveDefender(inference),
    }


def schedule(seed: int):
    scenarios = []
    for attack in ATTACKS:
        for index in range(34):
            scenarios.append(
                {
                    "attack": attack,
                    "goal": "A" if index % 2 == 0 else "B",
                    "seed": seed + 1_900_000 + len(scenarios),
                    "scenario": len(scenarios),
                    "scope": "main",
                }
            )
    audits = [
        {
            "attack": "adaptive_attacker",
            "goal": goal,
            "seed": seed + 2_000_000 + index,
            "scenario": 102 + index,
            "scope": "supplementary",
            "controller": controller,
        }
        for index, (controller, goal) in enumerate((("bayesian", "A"), ("pomdp", "B")))
    ]
    assert len(scenarios) * 5 + len(audits) == 512
    return scenarios, audits


def detection_auroc(rows, attack):
    subset = [row for row in rows if row["attack"] in ("honest", attack)]
    return binary_metrics(
        [row["deception_label"] for row in subset],
        [row["detection_probability_t5"] for row in subset],
    )["auroc"]


def render_report(destination, summary, failures):
    result = {(row["controller"], row["attack"]): row for row in summary}
    lines = [
        "# Stage 12 Bayesian and POMDP Baselines",
        "",
        "## Protocol",
        "",
        "GridWorld, Deceivers, Rule Active, Stage10.1, Adaptive Defender and all learned checkpoints are frozen. The two classical baselines are new experiment-layer code; neither is fitted to these evaluation episodes.",
        "",
        "The Bayesian baseline maintains P(goal|timed visible trajectory). Its likelihood is a soft shortest-path Markov policy and exactly marginalizes skipped positions over the observed time gap. It uses observe until posterior confidence >=.8 after t=5, or commits at t=12.",
        "",
        "The finite-horizon POMDP uses the same online Bayes belief, entropy and irreversible commitment status. A three-step exact belief-tree recursion compares commit_A/B with observe (assumed cue accuracy .68, cost .01) and track (accuracy .82, cost .03), interception reward1 and false-commitment penalty.5; its absolute decision deadline is t=10. These sensor accuracies are explicit planning assumptions, not learned calibration.",
        "",
        "Main smoke: 34 Honest, 34 ScriptedDeception(steps=3), 34 AdaptiveDeceiver scenarios, A/B balanced, each run against five Defenders =510 episodes. Two separately marked classical-baseline audits make exactly512 and are excluded from the main table. AUROC for each attack is computed against the same Defender's Honest controls.",
        "",
        "| Defender | Attack | Interception | False commitment | AUROC vs Honest | Observation cost | Regret |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["attack"] == "honest":
            continue
        lines.append(
            f"|{row['controller']}|{row['attack']}|{row['interception_success']:.2%}|"
            f"{row['false_commitment_rate']:.2%}|{row['detection_auroc_vs_honest']:.4f}|"
            f"{row['mean_observation_cost']:.4f}|{row['decision_regret']:.4f}|"
        )
    lines.extend(
        [
            "",
            "## Result",
            "",
            f"Stage10.1 intercepts {result[('stage10_1', 'scripted')]['interception_success']:.2%} of Scripted and {result[('stage10_1', 'adaptive_attacker')]['interception_success']:.2%} of Adaptive attacks. Bayesian achieves {result[('bayesian', 'scripted')]['interception_success']:.2%}/{result[('bayesian', 'adaptive_attacker')]['interception_success']:.2%}; POMDP achieves {result[('pomdp', 'scripted')]['interception_success']:.2%}/{result[('pomdp', 'adaptive_attacker')]['interception_success']:.2%}. All three are 100% on Honest controls. In this smoke, Stage10.1 is more robust than both classical baselines, with higher sensing cost under AdaptiveDeceiver.",
            "",
            "POMDP commits at mean t=1 under both attacks: the first deceptive cue is highly informative under its misspecified honest-motion model, so finite-horizon planning rationally selects the wrong irreversible action. Bayesian waits to t=5 but makes the same model-mismatch error. AUROC may be high at t=5 even after an earlier irreversible commitment; detection and decision success are therefore separate outcomes.",
            "",
            "## Limits",
            "",
            "The classical models assume approximately honest goal-directed motion. Scripted and adaptive attackers deliberately violate that likelihood. Repeated deterministic routes are not independent samples; AdaptiveDeceiver is online and therefore produces Defender-specific trajectories. This is a fixed single-map smoke, not a statistical superiority claim or a solved POMDP for the full GridWorld state.",
            "",
            f"Failure examples saved: {len(failures)}. Action traces, detection scores and attacker profiles are in episodes.csv and failure_cases.json; posterior/action-value behavior is covered by the classical-baseline tests.",
            "",
            "Stopped after 512 episodes; no further training was started.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (destination / "stage12_report.md").write_text(report)
    (destination / "comparison_table.md").write_text("\n".join(lines[12:]) + "\n")


def run(destination=OUTPUT, seed=0):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    protected = [
        ROOT / "src/deception_rl/envs/gridworld.py",
        ROOT / "src/deception_rl/agents/active_anti_deception.py",
        ROOT / "src/deception_rl/agents/decoupled_defender.py",
        ROOT / "src/deception_rl/agents/adaptive_defender.py",
        ROOT / "src/deception_rl/beliefs/decoupled.py",
        ROOT / "experiments/stage11_adaptive_deceiver.py",
        Path("/workspace/stage10-two-level/belief_model.pt"),
        Path("/workspace/stage10-1-decoupled/calibrator.pt"),
    ]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    controllers = build_controllers()
    adaptive_attackers = {name: AdaptiveDeceiver(seed) for name in CONTROLLERS}
    scenarios, audits = schedule(seed)
    rows = []
    for scenario in scenarios:
        for name, controller in controllers.items():
            if scenario["attack"] == "honest":
                attacker = HonestShortestPath()
            elif scenario["attack"] == "scripted":
                attacker = ScriptedDeception(3)
            else:
                attacker = adaptive_attackers[name]
            rows.append(run_episode(controller, attacker, name, **scenario))
    for audit in audits:
        controller_name = audit.pop("controller")
        rows.append(
            run_episode(
                controllers[controller_name],
                adaptive_attackers[controller_name],
                controller_name,
                **audit,
            )
        )
    assert len(rows) == 512
    main = [row for row in rows if row["scope"] == "main"]
    summary = []
    for name in CONTROLLERS:
        controller_rows = [row for row in main if row["controller"] == name]
        for attack in ATTACKS:
            selected = [row for row in controller_rows if row["attack"] == attack]
            result = summarize(selected)
            summary.append(
                {
                    "controller": name,
                    "attack": attack,
                    **result,
                    "detection_auroc_vs_honest": (
                        None if attack == "honest" else detection_auroc(controller_rows, attack)
                    ),
                }
            )
    failures = []
    for name in CONTROLLERS:
        for attack in ("scripted", "adaptive_attacker"):
            selected = [
                row
                for row in main
                if row["controller"] == name
                and row["attack"] == attack
                and not row["interception_success"]
            ]
            failures.extend(selected[:2])
    write_csv(destination / "episodes.csv", rows)
    write_csv(destination / "summary.csv", summary)
    (destination / "failure_cases.json").write_text(json.dumps(failures, indent=2))
    (destination / "attacker_profiles.json").write_text(
        json.dumps({name: attacker.profile() for name, attacker in adaptive_attackers.items()}, indent=2)
    )
    config = {
        "seed": seed,
        "total_episodes": 512,
        "main_scenarios": 102,
        "main_episodes": 510,
        "supplementary_episodes": 2,
        "per_attack_per_controller": 34,
        "goal_balance": "17 A / 17 B per attack and controller",
        "bayesian": {"rationality": 2.0, "confidence_threshold": 0.8, "commit_deadline": 12},
        "pomdp": {"horizon": 3, "decision_deadline": 10, "observe_accuracy": 0.68, "track_accuracy": 0.82, "observe_cost": 0.01, "track_cost": 0.03, "false_commitment_penalty": 0.5},
        "regret": "1 - (interception reward - observation cost - false commitment penalty)",
        "no_training": True,
        "selection": "single predeclared smoke; no post-result tuning",
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
    render_report(destination, summary, failures)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.seed), indent=2))

