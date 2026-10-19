"""Five-seed Stage 15 robustness and cross-environment evaluation.

This runner is evaluation-only.  It loads frozen Defender checkpoints, uses
fixed attacker rules, and never performs optimization or parameter selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from deception_rl.agents import HonestShortestPath, ScriptedDeception
from deception_rl.agents.active_anti_deception import (
    ActiveAntiDeception,
    DefenderReward,
    trajectory_evidence,
)
from deception_rl.agents.decoupled_defender import DecoupledAntiDeception
from deception_rl.envs import ForkedRoutesAdapter, GridWorldAdapter
from deception_rl.envs.gridworld import TRACK
from experiments.run_stage13_learned_controller import load_inference
from experiments.stage103_deception_families import policy as grid_family_policy
from experiments.stage11_adaptive_deceiver import AdaptiveDeceiver
from experiments.stage13_learned_controller import LearnedDecisionController
from experiments.stage14_classical_baselines import (
    FormalBayesianDefender,
    FormalFiniteHorizonPOMDPDefender,
)
from experiments.stage15_attackers import AdaptiveDeceiverV2, RouteAwareFamily
from experiments.train_anti_deception import binary_metrics
from experiments.train_learned_defender import write_csv
from models.stage13_decision import Stage13DecisionMLP


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "stage15_robustness"
SEEDS = (0, 1, 2, 3, 4)
GRID_FAMILIES = (
    "honest",
    "scripted_3",
    "adaptive_v1",
    "adaptive_v2",
    "scripted_8",
    "scripted_10",
    "delay_detour",
    "detour_switch",
    "stochastic_deception",
)
FORK_FAMILIES = (
    "honest",
    "scripted_3",
    "adaptive_v2",
    "scripted_8",
    "scripted_10",
    "delay_detour",
    "detour_switch",
    "stochastic_deception",
)
FORK_CONTROLLERS = (
    "rule_active",
    "formal_bayesian",
    "formal_pomdp",
    "stage10_1",
    "stage13_learned",
)
ACTION_NAMES = ("commit_A", "commit_B", "observe", "track")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_paths() -> list[Path]:
    return [
        ROOT / "src/deception_rl/envs/gridworld.py",
        ROOT / "experiments/stage13_learned_controller.py",
        ROOT / "outputs/stage13_learned_controller/stage13_controller.pt",
        Path("/workspace/stage10-two-level/belief_model.pt"),
        Path("/workspace/stage10-1-decoupled/calibrator.pt"),
    ]


def load_stage13_controller() -> LearnedDecisionController:
    checkpoint = torch.load(
        ROOT / "outputs/stage13_learned_controller/stage13_controller.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = Stage13DecisionMLP(checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return LearnedDecisionController(load_inference(), model)


def build_fork_controllers():
    model = ForkedRoutesAdapter().model
    return {
        "rule_active": ActiveAntiDeception(),
        "formal_bayesian": FormalBayesianDefender(model),
        "formal_pomdp": FormalFiniteHorizonPOMDPDefender(model),
        "stage10_1": DecoupledAntiDeception(load_inference()),
        "stage13_learned": load_stage13_controller(),
    }


def fixed_attacker(family: str, environment: str, model):
    if environment == "gridworld":
        if family == "honest":
            return HonestShortestPath()
        if family == "scripted_3":
            return ScriptedDeception(3)
        if family in {
            "scripted_8",
            "scripted_10",
            "delay_detour",
            "detour_switch",
            "stochastic_deception",
        }:
            return grid_family_policy(family)
        raise ValueError(f"Not a fixed GridWorld family: {family}")
    return RouteAwareFamily(family, model)


def _score(controller, observation, timestep: int) -> float:
    info = getattr(controller, "info", None)
    if info is not None and "deception_probability" in info:
        return float(info["deception_probability"])
    return float(trajectory_evidence(observation, timestep).deception_probability)


def evaluate_episode(
    adapter,
    controller,
    attacker,
    *,
    environment: str,
    controller_name: str,
    family: str,
    evaluation_seed: int,
    episode_index: int,
    goal: str,
    simulator_seed: int,
) -> dict[str, object]:
    env = adapter.make_env()
    observations, _ = env.reset(seed=simulator_seed, options={"goal": goal})
    controller.reset()
    attacker.reset(seed=simulator_seed)
    components = {
        "interception_reward": 0.0,
        "observation_cost": 0.0,
        "false_commitment_penalty": 0.0,
    }
    reward = DefenderReward()
    timestep = 0
    score_t5 = None
    score_at_decision = None
    intercepted = escaped = False
    action_counts = np.zeros(4, dtype=np.int64)
    deceiver_action_counts = np.zeros(5, dtype=np.int64)
    trajectory = []
    while env.agents:
        previous_goal = controller.goal
        visible_history = bool(
            np.asarray(observations["defender"]["trajectory_mask"], dtype=bool).any()
        )
        # The frozen neural inference contract requires one visible point.  The
        # Forked Routes start is outside default vision, so every controller is
        # given the same charged TRACK bootstrap until the first point exists.
        if not visible_history:
            defender_action = TRACK
            score = 0.5
        else:
            defender_action = int(controller.act(observations["defender"], timestep))
            score = _score(controller, observations["defender"], timestep)
        if timestep == 5:
            score_t5 = score
        new_commitment = previous_goal is None and controller.goal is not None
        if new_commitment:
            score_at_decision = score
        deceiver_action = int(attacker.act(observations["deceiver"]))
        before_mask = np.asarray(
            observations["defender"]["trajectory_mask"], dtype=bool
        )
        trajectory.append(
            {
                "t": timestep,
                "position": np.asarray(observations["deceiver"]["position"]).tolist(),
                "defender_action": ACTION_NAMES[defender_action],
                "deceiver_action": deceiver_action,
                "detection_score": score,
                "attacker_rule": getattr(attacker, "last_rule", family),
            }
        )
        observations, rewards, _, _, _ = env.step(
            {"deceiver": deceiver_action, "defender": defender_action}
        )
        intercepted = bool(rewards["defender"] > 0)
        escaped = bool(rewards["deceiver"] > 0)
        for key, value in reward.components(
            defender_action,
            intercepted,
            new_commitment,
            controller.goal,
            int(goal == "B"),
        ).items():
            components[key] += value
        action_counts[defender_action] += 1
        deceiver_action_counts[deceiver_action] += 1
        if hasattr(attacker, "observe_defender"):
            after_mask = np.asarray(
                observations["defender"]["trajectory_mask"], dtype=bool
            )
            attacker.observe_defender(
                defender_action,
                timestep,
                bool(after_mask.sum() > before_mask.sum()),
            )
        timestep += 1
    env.close()
    if score_t5 is None:
        score_t5 = _score(controller, observations["defender"], timestep)
    if score_at_decision is None:
        score_at_decision = score_t5
    episode_return = (
        components["interception_reward"]
        - components["observation_cost"]
        - components["false_commitment_penalty"]
    )
    committed = controller.goal is not None
    return {
        "environment": environment,
        "controller": controller_name,
        "family": family,
        "evaluation_seed": evaluation_seed,
        "episode": episode_index,
        "simulator_seed": simulator_seed,
        "true_goal": goal,
        "deception_label": int(family != "honest"),
        "detection_probability_t5": score_t5,
        "detection_probability_at_decision": score_at_decision,
        "committed": int(committed),
        "false_commitment": int(
            committed and controller.goal != int(goal == "B")
        ),
        "commit_timestep": controller.commit_timestep,
        "interception_success": int(intercepted),
        "deceiver_success": int(escaped),
        "truncated": int(not intercepted and not escaped),
        "steps": timestep,
        **components,
        "episode_return": episode_return,
        "decision_regret": 1.0 - episode_return,
        "defender_action_counts": action_counts.tolist(),
        "deceiver_action_counts": deceiver_action_counts.tolist(),
        "trajectory": trajectory,
        "attacker_profile": attacker.profile() if hasattr(attacker, "profile") else None,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    committed = [row for row in rows if row["committed"]]
    latencies = [float(row["commit_timestep"]) for row in committed]
    return {
        "episodes": len(rows),
        "interception_success": float(np.mean([row["interception_success"] for row in rows])),
        "false_commitment_rate": (
            float(np.mean([row["false_commitment"] for row in committed]))
            if committed
            else None
        ),
        "false_commitment_episode_rate": float(
            np.mean([row["false_commitment"] for row in rows])
        ),
        "commitment_coverage": len(committed) / len(rows),
        "observation_cost": float(np.mean([row["observation_cost"] for row in rows])),
        "decision_regret": float(np.mean([row["decision_regret"] for row in rows])),
        "decision_latency": float(np.mean(latencies)) if latencies else None,
        "truncation_rate": float(np.mean([row["truncated"] for row in rows])),
        "goal_A_success": float(
            np.mean([row["interception_success"] for row in rows if row["true_goal"] == "A"])
        ),
        "goal_B_success": float(
            np.mean([row["interception_success"] for row in rows if row["true_goal"] == "B"])
        ),
    }


def add_aurocs(seed_summaries, episodes):
    index = {
        (row["environment"], row["controller"], row["evaluation_seed"], row["family"]): row
        for row in seed_summaries
    }
    keys = sorted(
        {
            (row["environment"], row["controller"], row["evaluation_seed"])
            for row in episodes
        }
    )
    for environment, controller, seed in keys:
        available = [
            row
            for row in episodes
            if row["environment"] == environment
            and row["controller"] == controller
            and row["evaluation_seed"] == seed
        ]
        honest = [row for row in available if row["family"] == "honest"]
        for family in sorted({row["family"] for row in available}):
            target = index[(environment, controller, seed, family)]
            if family == "honest":
                target["detection_auroc_vs_honest"] = None
                continue
            selected = honest + [row for row in available if row["family"] == family]
            target["detection_auroc_vs_honest"] = binary_metrics(
                [row["deception_label"] for row in selected],
                [row["detection_probability_t5"] for row in selected],
            )["auroc"]


def run(destination=OUTPUT, seeds=SEEDS, grid_episodes=40, fork_episodes=20):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if grid_episodes < 2 or fork_episodes < 2 or grid_episodes % 2 or fork_episodes % 2:
        raise ValueError("Per-family episode counts must be positive even integers")
    paths = protected_paths()
    before = {str(path): sha256(path) for path in paths}
    protocol = {
        "evaluation_only": True,
        "seeds": list(seeds),
        "gridworld": {
            "controller": "frozen Stage13",
            "families": list(GRID_FAMILIES),
            "episodes_per_family_per_seed": grid_episodes,
        },
        "forked_routes": {
            "controllers": list(FORK_CONTROLLERS),
            "families": list(FORK_FAMILIES),
            "episodes_per_family_per_seed_per_controller": fork_episodes,
        },
        "goal_balance": "exactly half A and half B in every cell",
        "adaptive_v2": "fixed experiment-layer rules declared before rollout; no optimizer, reward fitting, or result-based selection",
        "unseen_families": [
            "scripted_8",
            "scripted_10",
            "delay_detour",
            "detour_switch",
            "stochastic_deception",
        ],
        "stage13_checkpoint_frozen": True,
        "deceiver_training": False,
        "hyperparameter_tuning": False,
        "forked_routes_bootstrap": "all controllers take charged TRACK while visible history is empty",
    }
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    grid_adapter = GridWorldAdapter()
    fork_adapter = ForkedRoutesAdapter()
    for evaluation_seed in seeds:
        controller = load_stage13_controller()
        adaptive = {
            "adaptive_v1": AdaptiveDeceiver(evaluation_seed),
            "adaptive_v2": AdaptiveDeceiverV2(grid_adapter.model, evaluation_seed),
        }
        for family_index, family in enumerate(GRID_FAMILIES):
            for episode_index in range(grid_episodes):
                attacker = adaptive.get(family)
                if attacker is None:
                    attacker = fixed_attacker(family, "gridworld", grid_adapter.model)
                simulator_seed = (
                    15_000_000
                    + evaluation_seed * 100_000
                    + family_index * 1_000
                    + episode_index
                )
                rows.append(
                    evaluate_episode(
                        grid_adapter,
                        controller,
                        attacker,
                        environment="gridworld",
                        controller_name="stage13_learned",
                        family=family,
                        evaluation_seed=evaluation_seed,
                        episode_index=episode_index,
                        goal="A" if episode_index % 2 == 0 else "B",
                        simulator_seed=simulator_seed,
                    )
                )
        controllers = build_fork_controllers()
        adaptive_fork = {
            name: AdaptiveDeceiverV2(fork_adapter.model, evaluation_seed)
            for name in controllers
        }
        for family_index, family in enumerate(FORK_FAMILIES):
            for episode_index in range(fork_episodes):
                simulator_seed = (
                    16_000_000
                    + evaluation_seed * 100_000
                    + family_index * 1_000
                    + episode_index
                )
                for name, fork_controller in controllers.items():
                    attacker = adaptive_fork[name] if family == "adaptive_v2" else fixed_attacker(
                        family, "forked_routes", fork_adapter.model
                    )
                    rows.append(
                        evaluate_episode(
                            fork_adapter,
                            fork_controller,
                            attacker,
                            environment="forked_routes",
                            controller_name=name,
                            family=family,
                            evaluation_seed=evaluation_seed,
                            episode_index=episode_index,
                            goal="A" if episode_index % 2 == 0 else "B",
                            simulator_seed=simulator_seed,
                        )
                    )
    expected = len(seeds) * (
        len(GRID_FAMILIES) * grid_episodes
        + len(FORK_FAMILIES) * fork_episodes * len(FORK_CONTROLLERS)
    )
    assert len(rows) == expected
    summaries = []
    groups = sorted(
        {
            (row["environment"], row["controller"], row["evaluation_seed"], row["family"])
            for row in rows
        }
    )
    for environment, controller, seed, family in groups:
        selected = [
            row
            for row in rows
            if row["environment"] == environment
            and row["controller"] == controller
            and row["evaluation_seed"] == seed
            and row["family"] == family
        ]
        summaries.append(
            {
                "environment": environment,
                "controller": controller,
                "evaluation_seed": seed,
                "family": family,
                **summarize(selected),
            }
        )
    add_aurocs(summaries, rows)
    write_csv(destination / "episodes.csv", rows)
    write_csv(destination / "per_seed_summary.csv", summaries)
    after = {str(path): sha256(path) for path in paths}
    unchanged = {path: before[path] == after[path] for path in before}
    assert all(unchanged.values())
    (destination / "preserved_inputs.json").write_text(
        json.dumps({"sha256_before": before, "sha256_after": after, "unchanged": unchanged}, indent=2)
        + "\n"
    )
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--grid-episodes", type=int, default=40)
    parser.add_argument("--fork-episodes", type=int, default=20)
    arguments = parser.parse_args()
    selected_seeds = tuple(int(value) for value in arguments.seeds.split(","))
    result = run(arguments.output_dir, selected_seeds, arguments.grid_episodes, arguments.fork_episodes)
    print(json.dumps({"rows": len(result), "output": str(arguments.output_dir)}, indent=2))
