"""Evaluate Stage 14 formal baselines on the frozen Stage 12 scenario schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from deception_rl.agents import HonestShortestPath, ScriptedDeception
from deception_rl.envs import (
    ForkedRoutesAdapter,
    GridWorldAdapter,
    validate_environment_interface,
)
from experiments.run_stage11_adaptive_attacker import run_episode, summarize
from experiments.run_stage12_bayesian_pomdp import detection_auroc, schedule
from experiments.run_stage13_learned_controller import load_inference
from experiments.stage11_adaptive_deceiver import AdaptiveDeceiver
from experiments.stage13_learned_controller import LearnedDecisionController
from experiments.stage14_classical_baselines import (
    FormalBayesianDefender,
    FormalFiniteHorizonPOMDPDefender,
)
from experiments.train_learned_defender import write_csv
from models.stage13_decision import Stage13DecisionMLP


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "stage14_formal_baselines"
CONTROLLERS = ("formal_bayesian", "formal_pomdp", "stage13_learned")
ATTACKS = ("honest", "scripted", "adaptive_attacker")


def build_controllers():
    model_spec = GridWorldAdapter().model
    checkpoint = torch.load(
        ROOT / "outputs" / "stage13_learned_controller" / "stage13_controller.pt",
        map_location="cpu",
        weights_only=True,
    )
    decision_model = Stage13DecisionMLP(checkpoint["hidden_size"])
    decision_model.load_state_dict(checkpoint["model_state_dict"])
    decision_model.eval()
    for parameter in decision_model.parameters():
        parameter.requires_grad_(False)
    return {
        "formal_bayesian": FormalBayesianDefender(model_spec),
        "formal_pomdp": FormalFiniteHorizonPOMDPDefender(model_spec),
        "stage13_learned": LearnedDecisionController(load_inference(), decision_model),
    }


def run(destination=OUTPUT, seed=0):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    protected = [
        ROOT / "src" / "deception_rl" / "envs" / "gridworld.py",
        ROOT / "experiments" / "stage11_adaptive_deceiver.py",
        ROOT / "src" / "deception_rl" / "beliefs" / "decoupled.py",
        ROOT / "outputs" / "stage12_bayesian_pomdp" / "episodes.csv",
        ROOT / "outputs" / "stage13_learned_controller" / "stage13_controller.pt",
        Path("/workspace/stage10-two-level/belief_model.pt"),
        Path("/workspace/stage10-1-decoupled/calibrator.pt"),
    ]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    controllers = build_controllers()
    adaptive_attackers = {name: AdaptiveDeceiver(seed) for name in controllers}
    scenarios, _ = schedule(seed)
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
    assert len(rows) == 306
    summary = []
    for name in CONTROLLERS:
        controller_rows = [row for row in rows if row["controller"] == name]
        for attack in ATTACKS:
            selected = [row for row in controller_rows if row["attack"] == attack]
            summary.append(
                {
                    "controller": name,
                    "attack": attack,
                    **summarize(selected),
                    "detection_auroc_vs_honest": (
                        None if attack == "honest" else detection_auroc(controller_rows, attack)
                    ),
                }
            )
    failures = [
        row
        for name in CONTROLLERS
        for attack in ("scripted", "adaptive_attacker")
        for row in [
            item
            for item in rows
            if item["controller"] == name
            and item["attack"] == attack
            and not item["interception_success"]
        ][:4]
    ]
    write_csv(destination / "episodes.csv", rows)
    write_csv(destination / "summary.csv", summary)
    (destination / "failure_cases.json").write_text(json.dumps(failures, indent=2))
    interface_rows = [
        validate_environment_interface(GridWorldAdapter()),
        validate_environment_interface(ForkedRoutesAdapter()),
    ]
    write_csv(destination / "environment_interfaces.csv", interface_rows)
    config = {
        "seed": seed,
        "evaluation_only": True,
        "training_episodes": 0,
        "episodes": len(rows),
        "common_scenarios": len(scenarios),
        "per_attack_per_controller": 34,
        "goal_balance": "17 A / 17 B per attack and controller",
        "scenario_schedule": "identical seed/goal/attack schedule to Stage 12 main evaluation",
        "bayesian": {
            "rationality": 2.0,
            "likelihood": "exact finite-state Boltzmann transition model in log space",
            "confidence_threshold": 0.8,
            "earliest_commit": 5,
            "decision_deadline": 12,
        },
        "pomdp": {
            "horizon": 5,
            "belief_state": "P(goal=A), P(goal=B), irreversible commitment",
            "observe_accuracy": 0.68,
            "track_accuracy": 0.82,
            "observe_cost": 0.01,
            "track_cost": 0.03,
            "interception_reward": 1.0,
            "false_commitment_penalty": 0.5,
            "solver": "exact finite-horizon belief-tree dynamic program",
        },
        "stage13_checkpoint_frozen": True,
        "gridworld_modified": False,
        "deceiver_training": False,
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
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir, arguments.seed), indent=2))

