"""Unified, training-free smoke evaluation for Stage 17.1 environments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from deception_rl.external_envs import (
    CyberDeceptionEnv,
    DefenderAction,
    DefenderEvaluationEnv,
    DefenderObservation,
    PettingZooDeceptionAdapter,
)
from deception_rl.external_envs.interface import ACTION_NAMES, validate_defender_environment


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/stage17_1_external_validation"


class DefenderPolicy(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, observation: DefenderObservation) -> DefenderAction: ...


class BeliefThresholdPolicy:
    """Fixed evaluator wiring policy; it is never fitted or optimized."""

    name = "fixed_belief_threshold"

    def __init__(self, confidence: float = 0.8, earliest_commit: int = 2, deadline: int = 6):
        if not 0.5 < confidence < 1 or earliest_commit < 0 or deadline < earliest_commit:
            raise ValueError("Invalid fixed policy settings")
        self.confidence = float(confidence)
        self.earliest_commit = int(earliest_commit)
        self.deadline = int(deadline)

    def reset(self) -> None:
        return None

    def act(self, observation: DefenderObservation) -> DefenderAction:
        probability_b = observation.belief[1]
        confidence = max(observation.belief)
        if observation.step >= self.deadline or (
            observation.step >= self.earliest_commit and confidence >= self.confidence
        ):
            return DefenderAction.COMMIT_B if probability_b > 0.5 else DefenderAction.COMMIT_A
        entropy = -sum(value * np.log2(max(value, 1e-12)) for value in observation.belief)
        return DefenderAction.TRACK if entropy > 0.65 else DefenderAction.OBSERVE


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def run_episode(
    env: DefenderEvaluationEnv,
    policy: DefenderPolicy,
    *,
    condition: str,
    episode: int,
    seed: int,
    goal: str,
) -> dict[str, object]:
    observation = env.reset(seed=seed, options={"goal": goal})
    policy.reset()
    trace = [{"initial": observation.as_dict()}]
    total_cost = 0.0
    success = None
    terminated = truncated = False
    actions = Counter()
    while not (terminated or truncated):
        action = DefenderAction(policy.act(observation))
        if not observation.action_mask[int(action)]:
            raise RuntimeError("Policy selected an unavailable action")
        transition = env.step(action)
        total_cost += transition.cost
        actions[ACTION_NAMES[action]] += 1
        trace.append(transition.as_dict())
        observation = transition.observation
        success = transition.success
        terminated = transition.terminated
        truncated = transition.truncated
    env.close()
    if success is None:
        raise AssertionError("Terminal transition must define success")
    return {
        "environment": env.name,
        "condition": condition,
        "policy": policy.name,
        "episode": episode,
        "seed": seed,
        "true_goal": goal,
        "success": int(success),
        "total_cost": total_cost,
        "steps": observation.step,
        "committed": int(observation.committed_goal is not None),
        "committed_goal": observation.committed_goal,
        "final_belief_A": observation.belief[0],
        "final_belief_B": observation.belief[1],
        "terminated": int(terminated),
        "truncated": int(truncated),
        "action_counts": dict(actions),
        "trace": trace,
    }


def evaluate(
    environment_factories: dict[str, Callable[[], DefenderEvaluationEnv]],
    policy_factory: Callable[[], DefenderPolicy],
    episodes_per_condition: int = 16,
    seed: int = 0,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if episodes_per_condition < 2 or episodes_per_condition % 2:
        raise ValueError("episodes_per_condition must be a positive even integer")
    rows = []
    for condition_index, (condition, factory) in enumerate(environment_factories.items()):
        for episode in range(episodes_per_condition):
            rows.append(
                run_episode(
                    factory(),
                    policy_factory(),
                    condition=condition,
                    episode=episode,
                    seed=seed + 17_000_000 + condition_index * 10_000 + episode,
                    goal="A" if episode % 2 == 0 else "B",
                )
            )
    summary = []
    for condition in environment_factories:
        selected = [row for row in rows if row["condition"] == condition]
        summary.append(
            {
                "environment": selected[0]["environment"],
                "condition": condition,
                "policy": selected[0]["policy"],
                "episodes": len(selected),
                "success_rate": float(np.mean([row["success"] for row in selected])),
                "mean_cost": float(np.mean([row["total_cost"] for row in selected])),
                "mean_steps": float(np.mean([row["steps"] for row in selected])),
                "commitment_coverage": float(np.mean([row["committed"] for row in selected])),
                "goal_A_success": float(
                    np.mean([row["success"] for row in selected if row["true_goal"] == "A"])
                ),
                "goal_B_success": float(
                    np.mean([row["success"] for row in selected if row["true_goal"] == "B"])
                ),
            }
        )
    return rows, summary


def environment_factories() -> dict[str, Callable[[], DefenderEvaluationEnv]]:
    return {
        "pettingzoo_honest": lambda: PettingZooDeceptionAdapter("honest"),
        "pettingzoo_deceptive": lambda: PettingZooDeceptionAdapter("deceptive"),
        "pettingzoo_adaptive": lambda: PettingZooDeceptionAdapter("adaptive"),
        "cyber_honest": lambda: CyberDeceptionEnv("honest_intrusion"),
        "cyber_credential_decoy": lambda: CyberDeceptionEnv("credential_decoy"),
        "cyber_lateral_switch": lambda: CyberDeceptionEnv("lateral_switch"),
        "cyber_low_and_slow": lambda: CyberDeceptionEnv("low_and_slow"),
    }


def protected_hashes() -> dict[str, str]:
    paths = [
        ROOT / "src/deception_rl/envs/gridworld.py",
        ROOT / "experiments/stage13_learned_controller.py",
        ROOT / "experiments/analyze_stage16_failures.py",
        ROOT / "outputs/stage13_learned_controller/stage13_controller.pt",
        Path("/workspace/stage10-two-level/belief_model.pt"),
        Path("/workspace/stage10-1-decoupled/calibrator.pt"),
    ]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run(destination=OUTPUT, episodes_per_condition: int = 16, seed: int = 0):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    before = protected_hashes()
    factories = environment_factories()
    manifests = [
        validate_defender_environment(PettingZooDeceptionAdapter("deceptive")),
        validate_defender_environment(CyberDeceptionEnv("credential_decoy")),
    ]
    episodes, summary = evaluate(
        factories, BeliefThresholdPolicy, episodes_per_condition, seed
    )
    write_csv(destination / "episodes.csv", episodes)
    write_csv(destination / "summary.csv", summary)
    (destination / "interface_manifest.json").write_text(json.dumps(manifests, indent=2) + "\n")
    config = {
        "training": False,
        "model_loading": False,
        "episodes_per_condition": episodes_per_condition,
        "conditions": list(factories),
        "goal_balance": "A/B equal within every condition",
        "policy": {
            "name": BeliefThresholdPolicy.name,
            "confidence": 0.8,
            "earliest_commit": 2,
            "deadline": 6,
            "purpose": "interface smoke only; not a learned or tuned baseline",
        },
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    report = [
        "# Stage 17.1 External Validation Environments",
        "",
        "Two independent validation backends expose the same DefenderObservation/DefenderTransition protocol. Every transition contains observation, belief, action, incremental cost, and terminal success. The PettingZoo adapter wraps a two-agent signal-deception ParallelEnv; CyberDeceptionEnv simulates alerts, forensic tracking, deceptive credentials, lateral switching, and containment of one of two assets.",
        "",
        "The smoke uses one fixed analytic-belief threshold policy only to verify evaluator wiring. It does not load, train, fine-tune, or compare a learned model. Goals are balanced within each condition.",
        "",
        "|Environment|Condition|Episodes|Success|Cost|Steps|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"|{row['environment']}|{row['condition']}|{row['episodes']}|{row['success_rate']:.2%}|{row['mean_cost']:.4f}|{row['mean_steps']:.2f}|"
        )
    report += [
        "",
        "These smoke numbers validate contracts and deterministic execution; they are not external-performance claims.",
    ]
    (destination / "stage17_1_report.md").write_text("\n".join(report) + "\n")
    after = protected_hashes()
    unchanged = {path: before[path] == after[path] for path in before}
    if not all(unchanged.values()):
        raise AssertionError("A Stage1-16 model or code input changed")
    (destination / "preserved_inputs.json").write_text(
        json.dumps({"sha256_before": before, "sha256_after": after, "unchanged": unchanged}, indent=2)
        + "\n"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--episodes-per-condition", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(arguments.output_dir, arguments.episodes_per_condition, arguments.seed),
            indent=2,
        )
    )
