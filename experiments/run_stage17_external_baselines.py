"""Stage 17.2 external baseline suite and fixed-budget MAPPO smoke run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch

from deception_rl.external_baselines import (
    ActiveInferenceDefender,
    BayesianDefenderAdapter,
    MAPPOActorCritic,
    MAPPODefenderPolicy,
    POMCPBeliefPlanner,
)
from deception_rl.external_envs import (
    CyberDeceptionEnv,
    DefenderAction,
    DefenderEvaluationEnv,
    DefenderObservation,
    ForkedRoutesDefenderAdapter,
    GridWorldDefenderAdapter,
    PettingZooDeceptionAdapter,
)
from deception_rl.external_envs.interface import ACTION_NAMES


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/stage17_2_external_baselines"
CSV_COLUMNS = (
    "method",
    "environment",
    "attack_family",
    "interception",
    "regret",
    "observation_cost",
    "calibration",
)


class DefenderPolicy(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, observation: DefenderObservation) -> DefenderAction: ...


@dataclass(frozen=True)
class EnvironmentSpec:
    environment: str
    attack_family: str
    context_index: int
    factory: Callable[[], DefenderEvaluationEnv]

    @property
    def key(self) -> str:
        return f"{self.environment}:{self.attack_family}"


def environment_specs() -> list[EnvironmentSpec]:
    specs: list[EnvironmentSpec] = []
    for family in ("honest", "scripted_3", "scripted_8"):
        specs.append(EnvironmentSpec("gridworld", family, 0, lambda family=family: GridWorldDefenderAdapter(family)))
        specs.append(EnvironmentSpec("forked_routes", family, 1, lambda family=family: ForkedRoutesDefenderAdapter(family)))
    for family in ("honest", "deceptive", "adaptive"):
        specs.append(EnvironmentSpec("pettingzoo_deception", family, 2, lambda family=family: PettingZooDeceptionAdapter(family)))
    for family in ("honest_intrusion", "credential_decoy", "lateral_switch", "low_and_slow"):
        specs.append(EnvironmentSpec("cyber_deception", family, 3, lambda family=family: CyberDeceptionEnv(family)))
    return specs


def _write_csv(path: Path, rows: list[dict[str, object]], columns=None) -> None:
    fieldnames = list(columns or rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], separators=(",", ":"))
                    if isinstance(row[key], (dict, list))
                    else row[key]
                    for key in fieldnames
                }
            )


def _context(index: int) -> torch.Tensor:
    result = torch.zeros(MAPPOActorCritic.context_size, dtype=torch.float32)
    result[index] = 1.0
    return result


def train_mappo_smoke(
    specs: list[EnvironmentSpec],
    destination: Path,
    *,
    seed: int = 0,
    episodes_per_condition: int = 4,
    epochs: int = 4,
) -> tuple[MAPPOActorCritic, list[dict[str, object]]]:
    """Run a deliberately small shared-policy PPO update.

    The actor is shared across environments.  Only the critic receives the
    environment one-hot context, matching the centralized-training convention.
    """
    if episodes_per_condition < 2 or episodes_per_condition % 2:
        raise ValueError("MAPPO smoke episodes must be a positive even number")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MAPPOActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    records: list[dict[str, object]] = []
    episode_number = 0
    for spec_index, spec in enumerate(specs):
        for local_episode in range(episodes_per_condition):
            episode_seed = seed + 3_000_000 + spec_index * 10_000 + local_episode
            env = spec.factory()
            observation = env.reset(
                seed=episode_seed,
                options={"goal": "A" if local_episode % 2 == 0 else "B"},
            )
            policy = MAPPODefenderPolicy(
                model, spec.context_index, deterministic=False, seed=episode_seed
            )
            trajectory = []
            done = False
            success = False
            while not done:
                with torch.no_grad():
                    distribution, value, features, context = policy.distribution(observation)
                    action_index = int(
                        torch.multinomial(
                            distribution.probs[0], 1, generator=policy._generator
                        )
                    )
                    log_probability = float(
                        distribution.log_prob(torch.tensor([action_index]))[0]
                    )
                transition = env.step(DefenderAction(action_index))
                done = transition.terminated or transition.truncated
                success = bool(transition.success) if done else False
                reward = -transition.cost
                if done:
                    reward += 1.0 if success else -1.0
                trajectory.append(
                    {
                        "features": features,
                        "context": context,
                        "mask": torch.tensor(observation.action_mask, dtype=torch.bool),
                        "action": action_index,
                        "old_log_probability": log_probability,
                        "old_value": float(value),
                        "reward": reward,
                    }
                )
                observation = transition.observation
            env.close()
            discounted_return = 0.0
            for item in reversed(trajectory):
                discounted_return = float(item["reward"]) + 0.99 * discounted_return
                item["return"] = discounted_return
                item["episode"] = episode_number
            records.extend(trajectory)
            episode_number += 1

    features = torch.stack([item["features"] for item in records])
    contexts = torch.stack([item["context"] for item in records])
    masks = torch.stack([item["mask"] for item in records])
    actions = torch.tensor([item["action"] for item in records], dtype=torch.long)
    old_log_probabilities = torch.tensor(
        [item["old_log_probability"] for item in records], dtype=torch.float32
    )
    returns = torch.tensor([item["return"] for item in records], dtype=torch.float32)
    old_values = torch.tensor([item["old_value"] for item in records], dtype=torch.float32)
    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    logs: list[dict[str, object]] = []
    model.train()
    for epoch in range(epochs):
        logits, values = model(features, contexts)
        logits = logits.masked_fill(~masks, -1e9)
        distribution = torch.distributions.Categorical(logits=logits)
        log_probabilities = distribution.log_prob(actions)
        ratio = torch.exp(log_probabilities - old_log_probabilities)
        surrogate = torch.minimum(
            ratio * advantages,
            torch.clamp(ratio, 0.8, 1.2) * advantages,
        )
        actor_loss = -surrogate.mean()
        critic_loss = torch.mean((values - returns) ** 2)
        entropy = distribution.entropy().mean()
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        logs.append(
            {
                "epoch": epoch + 1,
                "transitions": len(records),
                "episodes": episode_number,
                "actor_loss": float(actor_loss.detach()),
                "critic_loss": float(critic_loss.detach()),
                "entropy": float(entropy.detach()),
                "total_loss": float(loss.detach()),
            }
        )
    model.eval()
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_size": model.hidden_size,
            "seed": seed,
            "episodes": episode_number,
            "algorithm": "shared actor, centralized environment-context critic, PPO smoke",
        },
        destination / "mappo_smoke.pt",
    )
    _write_csv(destination / "mappo_training.csv", logs)
    return model, logs


def run_episode(
    spec: EnvironmentSpec,
    policy: DefenderPolicy,
    *,
    episode: int,
    seed: int,
    goal: str,
) -> dict[str, object]:
    env = spec.factory()
    observation = env.reset(seed=seed, options={"goal": goal})
    policy.reset()
    trace = []
    total_cost = 0.0
    first_commit_belief = None
    done = False
    success = False
    while not done:
        action = DefenderAction(policy.act(observation))
        if not observation.action_mask[int(action)]:
            raise RuntimeError(f"{policy.name} selected masked action {action.name}")
        if action in (DefenderAction.COMMIT_A, DefenderAction.COMMIT_B) and first_commit_belief is None:
            first_commit_belief = observation.belief
        transition = env.step(action)
        total_cost += transition.cost
        trace.append(
            {
                "step": observation.step,
                "belief": list(observation.belief),
                "action": ACTION_NAMES[action],
                "cost": transition.cost,
            }
        )
        observation = transition.observation
        done = transition.terminated or transition.truncated
        if done:
            success = bool(transition.success)
    env.close()
    decision_belief = first_commit_belief or observation.belief
    true_b = float(goal == "B")
    calibration = (float(decision_belief[1]) - true_b) ** 2
    regret = 1.0 - (float(success) - total_cost)
    return {
        "method": policy.name,
        "environment": spec.environment,
        "attack_family": spec.attack_family,
        "episode": episode,
        "seed": seed,
        "true_goal": goal,
        "interception": int(success),
        "regret": regret,
        "observation_cost": total_cost,
        "calibration": calibration,
        "steps": observation.step,
        "committed_goal": observation.committed_goal,
        "decision_belief_A": decision_belief[0],
        "decision_belief_B": decision_belief[1],
        "trace": trace,
    }


def evaluate_suite(
    specs: list[EnvironmentSpec],
    model: MAPPOActorCritic,
    *,
    episodes_per_condition: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if episodes_per_condition < 2 or episodes_per_condition % 2:
        raise ValueError("Evaluation episodes must be a positive even number")
    rows: list[dict[str, object]] = []
    for spec_index, spec in enumerate(specs):
        constructors: list[Callable[[int], DefenderPolicy]] = [
            lambda policy_seed: BayesianDefenderAdapter(),
            lambda policy_seed: POMCPBeliefPlanner(seed=policy_seed),
            lambda policy_seed: ActiveInferenceDefender(),
            lambda policy_seed, spec=spec: MAPPODefenderPolicy(
                model, spec.context_index, deterministic=True, seed=policy_seed
            ),
        ]
        for method_index, constructor in enumerate(constructors):
            for episode in range(episodes_per_condition):
                episode_seed = (
                    seed + 11_000_000 + spec_index * 100_000 + method_index * 10_000 + episode
                )
                rows.append(
                    run_episode(
                        spec,
                        constructor(episode_seed),
                        episode=episode,
                        seed=episode_seed,
                        goal="A" if episode % 2 == 0 else "B",
                    )
                )
    summary: list[dict[str, object]] = []
    for spec in specs:
        for method in ("bayesian", "pomcp", "active_inference", "mappo"):
            selected = [
                row
                for row in rows
                if row["environment"] == spec.environment
                and row["attack_family"] == spec.attack_family
                and row["method"] == method
            ]
            summary.append(
                {
                    "method": method,
                    "environment": spec.environment,
                    "attack_family": spec.attack_family,
                    "interception": float(np.mean([row["interception"] for row in selected])),
                    "regret": float(np.mean([row["regret"] for row in selected])),
                    "observation_cost": float(np.mean([row["observation_cost"] for row in selected])),
                    "calibration": float(np.mean([row["calibration"] for row in selected])),
                }
            )
    return rows, summary


def protected_hashes() -> dict[str, str]:
    candidates = [
        ROOT / "src/deception_rl/envs/gridworld.py",
        ROOT / "src/deception_rl/envs/forked_routes.py",
        ROOT / "experiments/stage13_learned_controller.py",
        ROOT / "outputs/stage13_learned_controller/stage13_controller.pt",
        Path("/workspace/stage10-two-level/belief_model.pt"),
        Path("/workspace/stage10-1-decoupled/calibrator.pt"),
    ]
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in candidates
        if path.exists()
    }


def run(
    destination: Path = OUTPUT,
    *,
    episodes_per_condition: int = 12,
    mappo_episodes_per_condition: int = 4,
    seed: int = 0,
) -> list[dict[str, object]]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    before = protected_hashes()
    specs = environment_specs()
    model, training_log = train_mappo_smoke(
        specs,
        destination,
        seed=seed,
        episodes_per_condition=mappo_episodes_per_condition,
    )
    episodes, summary = evaluate_suite(
        specs, model, episodes_per_condition=episodes_per_condition, seed=seed
    )
    _write_csv(destination / "episodes.csv", episodes)
    _write_csv(destination / "comparison.csv", summary, CSV_COLUMNS)
    config = {
        "seed": seed,
        "evaluation_episodes_per_condition": episodes_per_condition,
        "goal_balance": "equal A/B within each method/environment/attack condition",
        "mappo": {
            "training_scale": "smoke only",
            "episodes_per_condition": mappo_episodes_per_condition,
            "total_episodes": len(specs) * mappo_episodes_per_condition,
            "epochs": len(training_log),
            "large_training": False,
        },
        "metrics": {
            "interception": "terminal success mean",
            "regret": "1 - (terminal success - cumulative observation cost)",
            "observation_cost": "cumulative observe/track action cost",
            "calibration": "Brier score of P(goal=B) at first commitment or terminal horizon",
        },
        "conditions": [spec.key for spec in specs],
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    report = [
        "# Stage 17.2 External Baseline Suite",
        "",
        "This smoke compares Bayesian stopping, online POMCP belief planning, expected-free-energy active inference, and a small shared-policy MAPPO implementation through one Defender interface. MAPPO uses a centralized environment-aware critic during a fixed, small training budget; its actor sees only the common observation and belief. The results validate wiring and baseline behavior, not convergence.",
        "",
        "|Method|Environment|Attack|Interception|Regret|Obs. cost|Brier|",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"|{row['method']}|{row['environment']}|{row['attack_family']}|{row['interception']:.3f}|{row['regret']:.3f}|{row['observation_cost']:.3f}|{row['calibration']:.3f}|"
        )
    report += [
        "",
        "All four baselines expose observe, track, commit_A, and commit_B. GridWorld, Forked Routes, prior checkpoints, and Stage 1-16 sources were read only.",
    ]
    (destination / "stage17_2_report.md").write_text("\n".join(report) + "\n")
    after = protected_hashes()
    if before != after:
        raise AssertionError("A protected environment, controller, or checkpoint changed")
    (destination / "preserved_inputs.json").write_text(
        json.dumps({"sha256_before": before, "sha256_after": after, "unchanged": True}, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--episodes-per-condition", type=int, default=12)
    parser.add_argument("--mappo-episodes-per-condition", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.output_dir,
                episodes_per_condition=arguments.episodes_per_condition,
                mappo_episodes_per_condition=arguments.mappo_episodes_per_condition,
                seed=arguments.seed,
            ),
            indent=2,
        )
    )
