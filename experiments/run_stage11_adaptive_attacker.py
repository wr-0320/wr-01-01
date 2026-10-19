"""Stage 11 fixed-budget evaluation of an online rule-adaptive attacker."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from deception_rl.agents import HonestShortestPath, ScriptedDeception
from deception_rl.agents.active_anti_deception import (
    ActiveAntiDeception,
    DefenderReward,
    trajectory_evidence,
)
from deception_rl.agents.adaptive_defender import AdaptiveActiveDefender
from deception_rl.agents.decoupled_defender import DecoupledAntiDeception
from deception_rl.beliefs.decoupled import (
    CalibratedGoalInference,
    EvidenceCalibrator,
    LikelihoodRatioReasoner,
)
from deception_rl.beliefs.two_level import DeceptionAwareBelief
from deception_rl.envs.gridworld import GridWorldParallelEnv
from experiments.run_active_anti_deception import summarize as base_summary
from experiments.stage11_adaptive_deceiver import AdaptiveDeceiver
from experiments.train_anti_deception import binary_metrics
from experiments.train_learned_defender import write_csv
from models.anti_deception_gru import AntiDeceptionGRU


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "stage11_adaptive_attacker"
ATTACKS = ("honest", "scripted", "adaptive")
CONTROLLERS = ("rule_active", "stage10_1", "adaptive_defender")
ACTION_NAMES = ("commit_A", "commit_B", "observe", "track")


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
        "stage10_1": DecoupledAntiDeception(inference),
        "adaptive_defender": AdaptiveActiveDefender(inference),
    }


def scenario_schedule(seed: int):
    scenarios = []
    for attack in ATTACKS:
        for index in range(56):
            scenarios.append(
                {
                    "attack": attack,
                    "goal": "A" if index % 2 == 0 else "B",
                    "seed": seed + 1_700_000 + len(scenarios),
                    "scenario": len(scenarios),
                    "scope": "main",
                }
            )
    audits = [
        {
            "attack": "adaptive",
            "goal": "A" if index % 2 == 0 else "B",
            "seed": seed + 1_800_000 + index,
            "scenario": 168 + index,
            "scope": "supplementary",
        }
        for index in range(8)
    ]
    assert len(scenarios) * 3 + len(audits) == 512
    return scenarios, audits


def run_episode(controller, attacker, controller_name: str, **scenario):
    env = GridWorldParallelEnv()
    observations, _ = env.reset(seed=scenario["seed"], options={"goal": scenario["goal"]})
    controller.reset()
    attacker.reset(seed=scenario["seed"])
    timestep = 0
    detection_t5 = None
    decision_score = None
    trace = []
    components = {
        "interception_reward": 0.0,
        "observation_cost": 0.0,
        "false_commitment_penalty": 0.0,
    }
    intercepted = escaped = False
    while env.agents:
        previous_goal = controller.goal
        defender_action = controller.act(observations["defender"], timestep)
        info = getattr(controller, "info", None)
        score = (
            info["deception_probability"]
            if info
            else trajectory_evidence(observations["defender"], timestep).deception_probability
        )
        if timestep == 5:
            detection_t5 = score
        new_commitment = previous_goal is None and controller.goal is not None
        if new_commitment:
            decision_score = score
        deceiver_action = attacker.act(observations["deceiver"])
        attacker_rule = getattr(attacker, "last_rule", "fixed")
        before_mask = np.asarray(observations["defender"]["trajectory_mask"], dtype=bool)
        trace.append(
            {
                "timestep": timestep,
                "defender_action": ACTION_NAMES[defender_action],
                "deceiver_action": int(deceiver_action),
                "attacker_rule": attacker_rule,
                "P_deceptive": float(score),
                "attacker_profile": attacker.profile() if isinstance(attacker, AdaptiveDeceiver) else None,
                "visible_points_before": int(before_mask.sum()),
            }
        )
        observations, rewards, _, _, _ = env.step(
            {"deceiver": deceiver_action, "defender": defender_action}
        )
        intercepted = rewards["defender"] > 0
        escaped = rewards["deceiver"] > 0
        parts = DefenderReward().components(
            defender_action,
            intercepted,
            new_commitment,
            controller.goal,
            int(scenario["goal"] == "B"),
        )
        for key, value in parts.items():
            components[key] += value
        if isinstance(attacker, AdaptiveDeceiver):
            after_mask = np.asarray(observations["defender"]["trajectory_mask"], dtype=bool)
            attacker.observe_defender(
                defender_action,
                timestep,
                bool(after_mask.sum() > before_mask.sum()),
            )
        timestep += 1
    env.close()
    assert detection_t5 is not None
    episode_return = (
        components["interception_reward"]
        - components["observation_cost"]
        - components["false_commitment_penalty"]
    )
    return {
        "controller": controller_name,
        **scenario,
        "deception_label": int(scenario["attack"] != "honest"),
        "detection_probability_t5": detection_t5,
        "detection_probability_at_decision": detection_t5 if decision_score is None else decision_score,
        "committed": int(controller.goal is not None),
        "false_commitment": int(
            controller.goal is not None and controller.goal != int(scenario["goal"] == "B")
        ),
        "commit_timestep": controller.commit_timestep,
        "interception_success": int(intercepted),
        "deceiver_success": int(escaped),
        "steps": timestep,
        **components,
        "episode_return": episode_return,
        "decision_regret": 1.0 - episode_return,
        "trace": trace,
        "final_attacker_profile": attacker.profile() if isinstance(attacker, AdaptiveDeceiver) else None,
    }


def summarize(rows):
    compatibility = [
        {
            **row,
            "defender_task_success": row["interception_success"],
            "deceiver_task_success": row["deceiver_success"],
        }
        for row in rows
    ]
    result = base_summary(compatibility)
    result["interception_success"] = float(np.mean([row["interception_success"] for row in rows]))
    result["decision_regret"] = float(np.mean([row["decision_regret"] for row in rows]))
    committed = [row for row in rows if row["committed"]]
    result["decision_latency"] = (
        float(np.mean([row["commit_timestep"] for row in committed])) if committed else None
    )
    return result


def auroc_against_honest(rows, attack):
    subset = [row for row in rows if row["attack"] in ("honest", attack)]
    values = binary_metrics(
        [row["deception_label"] for row in subset],
        [row["detection_probability_t5"] for row in subset],
    )
    return values["auroc"]


def render_report(destination, summary, failures):
    adaptive_attack = {
        row["controller"]: row for row in summary if row["attack"] == "adaptive"
    }
    lines = [
        "# Stage 11 Adaptive Strategic Deceiver",
        "",
        "## Protocol",
        "",
        "GridWorld and all existing Defenders are frozen. AdaptiveDeceiver exists only in the experiment layer and performs no PPO, gradient update, or reward fitting. It keeps a tiny per-Defender cross-episode profile of commitment times and targets. Within an episode it sees only completed prior Defender actions, commitment events, and whether the last action exposed the attacker; it cannot read a current simultaneous action or Defender model state.",
        "",
        "Main evaluation: 168 scenarios (56 Honest, 56 ScriptedDeception steps=3, 56 AdaptiveDeceiver), each run against three Defenders for 504 episodes. Goals are A/B balanced within every condition. Eight supplementary AdaptiveDeceiver-vs-AdaptiveDefender audits make exactly 512 episodes and are excluded from the main comparison.",
        "",
        "AUROC is computed separately for Honest+Scripted and Honest+Adaptive because an all-deceptive subset has no negative class. Regret uses the existing privileged immediate-correct-guard oracle: 1 − episode_return. It is an evaluation score, not Bayes regret under equal information.",
        "",
        "| Defender | Attack | Interception | False commitment | Detection AUROC vs Honest | Observation cost | Regret |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["attack"] not in ("scripted", "adaptive", "all_deceptive"):
            continue
        auc = "N/A" if row["detection_auroc_vs_honest"] is None else f"{row['detection_auroc_vs_honest']:.4f}"
        lines.append(
            f"|{row['controller']}|{row['attack']}|{row['interception_success']:.2%}|"
            f"{row['false_commitment_rate']:.2%}|{auc}|{row['mean_observation_cost']:.4f}|"
            f"{row['decision_regret']:.4f}|"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Under AdaptiveDeceiver, interception is Rule={adaptive_attack['rule_active']['interception_success']:.2%}, Stage10.1={adaptive_attack['stage10_1']['interception_success']:.2%}, Adaptive Defender={adaptive_attack['adaptive_defender']['interception_success']:.2%}. These are fixed smoke results, not a tuned attacker/defender equilibrium.",
            "",
            "Adaptive behavior is strategic only in the limited operational sense that action selection changes with observed sensing, learned commitment timing, and cross-episode target bias. The rules are hand-written, the attacker knows its true goal, and the experiment grants it Defender action visibility through the wrapper. This does not establish game-theoretic optimality.",
            "",
            f"Failure examples saved: {len(failures)}. Exact traces and attacker profiles are in failure_cases.json; all 512 records are in episodes.csv.",
            "",
            "Stopped after the smoke run; no further training was started.",
        ]
    )
    (destination / "stage11_report.md").write_text("\n".join(lines) + "\n")
    (destination / "comparison_table.md").write_text("\n".join(lines[8:]) + "\n")


def run(destination=OUTPUT, seed=0):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    protected = [
        ROOT / "src/deception_rl/envs/gridworld.py",
        ROOT / "src/deception_rl/agents/active_anti_deception.py",
        ROOT / "src/deception_rl/agents/decoupled_defender.py",
        ROOT / "src/deception_rl/agents/adaptive_defender.py",
        ROOT / "src/deception_rl/beliefs/decoupled.py",
        Path("/workspace/stage10-two-level/belief_model.pt"),
        Path("/workspace/stage10-1-decoupled/calibrator.pt"),
    ]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    controllers = build_controllers()
    adaptive_attackers = {name: AdaptiveDeceiver(seed) for name in CONTROLLERS}
    scenarios, audits = scenario_schedule(seed)
    rows = []
    # Attack conditions are intentionally ordered: no adaptive profile is updated
    # by Honest or Scripted control episodes.
    for scenario in scenarios:
        for name, controller in controllers.items():
            if scenario["attack"] == "honest":
                attacker = HonestShortestPath()
            elif scenario["attack"] == "scripted":
                attacker = ScriptedDeception(3)
            else:
                attacker = adaptive_attackers[name]
            rows.append(run_episode(controller, attacker, name, **scenario))
    for scenario in audits:
        rows.append(
            run_episode(
                controllers["adaptive_defender"],
                adaptive_attackers["adaptive_defender"],
                "adaptive_defender",
                **scenario,
            )
        )
    assert len(rows) == 512
    main = [row for row in rows if row["scope"] == "main"]
    summary = []
    for controller in CONTROLLERS:
        controller_rows = [row for row in main if row["controller"] == controller]
        for attack in ATTACKS:
            selected = [row for row in controller_rows if row["attack"] == attack]
            result = summarize(selected)
            result["detection_auroc_vs_honest"] = (
                None if attack == "honest" else auroc_against_honest(controller_rows, attack)
            )
            summary.append({"controller": controller, "attack": attack, **result})
        deceptive = [row for row in controller_rows if row["attack"] != "honest"]
        result = summarize(deceptive)
        all_detection = binary_metrics(
            [row["deception_label"] for row in controller_rows],
            [row["detection_probability_t5"] for row in controller_rows],
        )
        summary.append(
            {
                "controller": controller,
                "attack": "all_deceptive",
                **result,
                "detection_auroc_vs_honest": all_detection["auroc"],
            }
        )
    failures = []
    for controller in CONTROLLERS:
        for attack in ("scripted", "adaptive"):
            candidates = [
                row
                for row in main
                if row["controller"] == controller
                and row["attack"] == attack
                and not row["interception_success"]
            ]
            failures.extend(candidates[:2])
    write_csv(destination / "episodes.csv", rows)
    write_csv(destination / "summary.csv", summary)
    (destination / "failure_cases.json").write_text(json.dumps(failures, indent=2))
    profiles = {
        name: attacker.profile() for name, attacker in adaptive_attackers.items()
    }
    (destination / "attacker_profiles.json").write_text(json.dumps(profiles, indent=2))
    config = {
        "seed": seed,
        "total_episodes": 512,
        "main_paired_scenarios": 168,
        "main_episodes": 504,
        "supplementary_episodes": 8,
        "main_per_attack_per_controller": 56,
        "scripted_deception_steps": 3,
        "adaptive_rules": [
            "commitment-aware timing from prior commit steps",
            "target-aware opposite-to-long-run-bias induction",
            "observation-aware fake trajectory from completed observe/track actions",
        ],
        "information_boundary": "completed prior Defender actions/commit and visibility only; no current simultaneous action or model state",
        "no_training": True,
        "no_ppo": True,
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
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir, arguments.seed), indent=2))

