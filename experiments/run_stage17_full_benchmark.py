"""Stage 17.3 five-seed full benchmark over four Defender environments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from deception_rl.external_baselines import (
    ActiveInferenceDefender,
    BayesianDefenderAdapter,
    MAPPOActorCritic,
    MAPPODefenderPolicy,
    POMCPBeliefPlanner,
    RIDeDefenderAdapter,
    RuleActiveDefender,
)
from deception_rl.external_envs import (
    ATTACK_SETTINGS,
    BenchmarkCyberAdapter,
    BenchmarkNavigationAdapter,
    BenchmarkPettingZooAdapter,
    DefenderAction,
    DefenderEvaluationEnv,
)
from deception_rl.external_envs.interface import ACTION_NAMES


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/stage17_3_full_benchmark"
RIDE_CHECKPOINT = ROOT / "outputs/stage13_learned_controller/stage13_controller.pt"
METHODS = ("rule_active", "bayesian", "pomcp", "active_inference", "mappo", "ride")
ENVIRONMENTS = ("gridworld", "forked_routes", "pettingzoo_deception", "cyber_deception")
SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class BenchmarkSpec:
    environment: str
    attack_family: str
    context_index: int
    factory: Callable[[], DefenderEvaluationEnv]

    @property
    def key(self) -> str:
        return f"{self.environment}:{self.attack_family}"


def benchmark_specs() -> list[BenchmarkSpec]:
    specs: list[BenchmarkSpec] = []
    for family in ATTACK_SETTINGS:
        specs.extend(
            [
                BenchmarkSpec(
                    "gridworld",
                    family,
                    0,
                    lambda family=family: BenchmarkNavigationAdapter("gridworld", family),
                ),
                BenchmarkSpec(
                    "forked_routes",
                    family,
                    1,
                    lambda family=family: BenchmarkNavigationAdapter("forked_routes", family),
                ),
                BenchmarkSpec(
                    "pettingzoo_deception",
                    family,
                    2,
                    lambda family=family: BenchmarkPettingZooAdapter(family),
                ),
                BenchmarkSpec(
                    "cyber_deception",
                    family,
                    3,
                    lambda family=family: BenchmarkCyberAdapter(family),
                ),
            ]
        )
    return specs


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames or rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], separators=(",", ":"))
                    if isinstance(row[key], (list, dict, tuple))
                    else row[key]
                    for key in columns
                }
            )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_device(device: str | torch.device = "cpu") -> torch.device:
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Stage17 MAPPO supports only cpu or cuda devices")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; refusing CPU fallback")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable")
        device = torch.device("cuda", index)
    return device


def _context(index: int, device: str | torch.device = "cpu") -> torch.Tensor:
    vector = torch.zeros(MAPPOActorCritic.context_size, dtype=torch.float32, device=device)
    vector[index] = 1.0
    return vector


def _collect_rollout(
    model: MAPPOActorCritic,
    specs: list[BenchmarkSpec],
    rng: np.random.Generator,
    *,
    seed: int,
    update: int,
    minimum_steps: int,
    episode_counter: int,
) -> tuple[list[dict[str, object]], list[dict[str, float]], int]:
    device = next(model.parameters()).device
    records: list[dict[str, object]] = []
    outcomes: list[dict[str, float]] = []
    while len(records) < minimum_steps:
        spec = specs[episode_counter % len(specs)]
        goal_index = episode_counter % 2
        episode_seed = seed * 100_000_000 + update * 100_000 + episode_counter
        env = spec.factory()
        observation = env.reset(seed=episode_seed, options={"goal": "B" if goal_index else "A"})
        context = _context(spec.context_index, device)
        trajectory: list[dict[str, object]] = []
        done = False
        total_cost = 0.0
        terminal_success = False
        while not done:
            features = np.concatenate(
                (
                    np.asarray(observation.observation, dtype=np.float32),
                    np.asarray(observation.belief, dtype=np.float32),
                )
            )
            feature_tensor = torch.from_numpy(features).to(device)
            mask = torch.tensor(observation.action_mask, dtype=torch.bool, device=device)
            with torch.no_grad():
                logits, value = model(feature_tensor.unsqueeze(0), context.unsqueeze(0))
                logits = logits[0].masked_fill(~mask, -1e9)
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
                old_log_probability = distribution.log_prob(action)
            transition = env.step(DefenderAction(int(action)))
            done = transition.terminated or transition.truncated
            total_cost += transition.cost
            terminal_success = bool(transition.success) if done else False
            reward = -transition.cost
            if done:
                reward += 1.0 if terminal_success else -1.0
            trajectory.append(
                {
                    "features": feature_tensor,
                    "context": context,
                    "mask": mask,
                    "action": int(action),
                    "old_log_probability": float(old_log_probability),
                    "old_value": float(value[0]),
                    "reward": reward,
                }
            )
            observation = transition.observation
        env.close()
        discounted_return = 0.0
        for item in reversed(trajectory):
            discounted_return = float(item["reward"]) + 0.99 * discounted_return
            item["return"] = discounted_return
        records.extend(trajectory)
        outcomes.append(
            {
                "success": float(terminal_success),
                "return": float(sum(item["reward"] for item in trajectory)),
                "cost": total_cost,
            }
        )
        episode_counter += 1
    return records, outcomes, episode_counter


def _ppo_update(
    model: MAPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    records: list[dict[str, object]],
    rng: np.random.Generator,
    *,
    epochs: int = 4,
    minibatch_size: int = 256,
) -> dict[str, float]:
    device = next(model.parameters()).device
    features = torch.stack([item["features"] for item in records]).to(device)
    contexts = torch.stack([item["context"] for item in records]).to(device)
    masks = torch.stack([item["mask"] for item in records]).to(device)
    actions = torch.tensor([item["action"] for item in records], dtype=torch.long, device=device)
    old_log_probabilities = torch.tensor(
        [item["old_log_probability"] for item in records], dtype=torch.float32, device=device
    )
    old_values = torch.tensor([item["old_value"] for item in records], dtype=torch.float32, device=device)
    returns = torch.tensor([item["return"] for item in records], dtype=torch.float32, device=device)
    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    metrics: defaultdict[str, list[float]] = defaultdict(list)
    model.train()
    for _ in range(epochs):
        permutation = rng.permutation(len(records))
        for start in range(0, len(records), minibatch_size):
            indices = permutation[start : start + minibatch_size]
            index = torch.as_tensor(indices, dtype=torch.long, device=device)
            logits, values = model(features[index], contexts[index])
            logits = logits.masked_fill(~masks[index], -1e9)
            distribution = torch.distributions.Categorical(logits=logits)
            log_probabilities = distribution.log_prob(actions[index])
            ratio = torch.exp(log_probabilities - old_log_probabilities[index])
            unclipped = ratio * advantages[index]
            clipped = torch.clamp(ratio, 0.8, 1.2) * advantages[index]
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            critic_loss = torch.mean((values - returns[index]) ** 2)
            entropy = distribution.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            metrics["actor_loss"].append(float(actor_loss.detach()))
            metrics["critic_loss"].append(float(critic_loss.detach()))
            metrics["entropy"].append(float(entropy.detach()))
            metrics["gradient_norm"].append(float(gradient_norm))
            metrics["clip_fraction"].append(float((torch.abs(ratio - 1) > 0.2).float().mean()))
    model.eval()
    variance = float(torch.var(returns, unbiased=False))
    explained_variance = 1 - float(torch.var(returns - old_values, unbiased=False)) / max(variance, 1e-8)
    result = {key: float(np.mean(values)) for key, values in metrics.items()}
    result["explained_variance"] = explained_variance
    return result


def _quick_mappo_evaluation(
    model: MAPPOActorCritic,
    specs: list[BenchmarkSpec],
    *,
    seed: int,
    update: int,
) -> tuple[float, float]:
    successes = []
    returns = []
    for spec_index, spec in enumerate(specs):
        for episode in range(4):
            env = spec.factory()
            observation = env.reset(
                seed=seed * 10_000_000 + update * 10_000 + spec_index * 100 + episode,
                options={"goal": "A" if episode % 2 == 0 else "B"},
            )
            policy = MAPPODefenderPolicy(model, spec.context_index, deterministic=True)
            done = False
            total_reward = 0.0
            success = False
            while not done:
                transition = env.step(policy.act(observation))
                done = transition.terminated or transition.truncated
                total_reward -= transition.cost
                observation = transition.observation
                if done:
                    success = bool(transition.success)
                    total_reward += 1.0 if success else -1.0
            env.close()
            successes.append(float(success))
            returns.append(total_reward)
    return float(np.mean(successes)), float(np.mean(returns))


def train_mappo_seed(
    seed: int,
    specs: list[BenchmarkSpec],
    destination: Path,
    *,
    training_steps: int,
    rollout_steps: int = 2048,
    device: str | torch.device = "cpu",
) -> tuple[MAPPOActorCritic, list[dict[str, object]]]:
    device = resolve_device(device)
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / f"mappo_seed{seed}_latest.pt"
    final_path = checkpoint_dir / f"mappo_seed{seed}.pt"
    curve_path = destination / "training_curves" / f"seed{seed}.csv"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = MAPPOActorCritic(hidden_size=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    completed_steps = 0
    update = 0
    episode_counter = 0
    curves: list[dict[str, object]] = []
    if latest_path.exists():
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        if int(payload["target_training_steps"]) == training_steps:
            model.load_state_dict(payload["state_dict"])
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            completed_steps = int(payload["completed_steps"])
            update = int(payload["update"])
            episode_counter = int(payload["episode_counter"])
            rng.bit_generator.state = payload["numpy_rng_state"]
            torch.set_rng_state(payload["torch_rng_state"])
            if device.type == "cuda" and "cuda_rng_state" in payload:
                torch.cuda.set_rng_state(payload["cuda_rng_state"], device)
            if curve_path.exists():
                curves = [dict(row) for row in read_csv(curve_path)]
    if final_path.exists() and completed_steps >= training_steps:
        model.eval()
        return model, curves

    train_specs = [spec for spec in specs if spec.attack_family != "unseen_family"]
    while completed_steps < training_steps:
        update += 1
        records, outcomes, episode_counter = _collect_rollout(
            model,
            train_specs,
            rng,
            seed=seed,
            update=update,
            minimum_steps=min(rollout_steps, training_steps - completed_steps),
            episode_counter=episode_counter,
        )
        metrics = _ppo_update(model, optimizer, records, rng)
        completed_steps += len(records)
        evaluation_success, evaluation_return = _quick_mappo_evaluation(
            model, train_specs, seed=seed, update=update
        )
        row = {
            "seed": seed,
            "update": update,
            "timesteps": completed_steps,
            "episodes": episode_counter,
            "rollout_success": float(np.mean([item["success"] for item in outcomes])),
            "rollout_return": float(np.mean([item["return"] for item in outcomes])),
            "rollout_cost": float(np.mean([item["cost"] for item in outcomes])),
            "evaluation_success": evaluation_success,
            "evaluation_return": evaluation_return,
            **metrics,
        }
        curves.append(row)
        write_csv(curve_path, curves)
        payload = {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hidden_size": model.hidden_size,
            "seed": seed,
            "completed_steps": completed_steps,
            "target_training_steps": training_steps,
            "update": update,
            "episode_counter": episode_counter,
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "training_families": ("scripted", "adaptive"),
            "unseen_family_excluded": True,
        }
        if device.type == "cuda":
            payload["device"] = str(device)
            payload["cuda_rng_state"] = torch.cuda.get_rng_state(device)
        torch.save(payload, latest_path)
        if update == 1 or update % 10 == 0 or completed_steps >= training_steps:
            print(
                f"MAPPO seed={seed} steps={completed_steps}/{training_steps} "
                f"eval_success={evaluation_success:.3f}",
                flush=True,
            )
    shutil.copyfile(latest_path, final_path)
    model.eval()
    return model, curves


def _policy(
    method: str,
    *,
    episode_seed: int,
    spec: BenchmarkSpec,
    mappo_model: MAPPOActorCritic,
    ride_policy: RIDeDefenderAdapter,
):
    if method == "rule_active":
        return RuleActiveDefender()
    if method == "bayesian":
        return BayesianDefenderAdapter()
    if method == "pomcp":
        return POMCPBeliefPlanner(seed=episode_seed)
    if method == "active_inference":
        return ActiveInferenceDefender()
    if method == "mappo":
        return MAPPODefenderPolicy(mappo_model, spec.context_index, deterministic=True)
    if method == "ride":
        return ride_policy
    raise ValueError(f"Unknown method: {method}")


def evaluate_episode(
    spec: BenchmarkSpec,
    method: str,
    policy,
    *,
    seed: int,
    episode: int,
    episode_seed: int,
) -> dict[str, object]:
    goal = "A" if episode % 2 == 0 else "B"
    env = spec.factory()
    observation = env.reset(seed=episode_seed, options={"goal": goal})
    policy.reset()
    total_cost = 0.0
    actions = defaultdict(int)
    first_commit_belief = None
    done = False
    success = False
    while not done:
        action = DefenderAction(policy.act(observation))
        if not observation.action_mask[int(action)]:
            raise RuntimeError(f"{method} selected unavailable {action.name}")
        if action in (DefenderAction.COMMIT_A, DefenderAction.COMMIT_B) and first_commit_belief is None:
            first_commit_belief = observation.belief
        transition = env.step(action)
        actions[ACTION_NAMES[action]] += 1
        total_cost += transition.cost
        observation = transition.observation
        done = transition.terminated or transition.truncated
        if done:
            success = bool(transition.success)
    env.close()
    belief = first_commit_belief or observation.belief
    true_b = float(goal == "B")
    probability_b = float(np.clip(belief[1], 1e-8, 1 - 1e-8))
    return {
        "seed": seed,
        "method": method,
        "environment": spec.environment,
        "attack_family": spec.attack_family,
        "episode": episode,
        "episode_seed": episode_seed,
        "true_goal": goal,
        "interception": int(success),
        "regret": 1.0 - (float(success) - total_cost),
        "observation_cost": total_cost,
        "calibration": (probability_b - true_b) ** 2,
        "probability_B": probability_b,
        "negative_log_likelihood": -math.log(probability_b if true_b else 1 - probability_b),
        "steps": observation.step,
        "committed": int(observation.committed_goal is not None),
        "committed_goal": observation.committed_goal,
        "observe_count": actions["observe"],
        "track_count": actions["track"],
        "commit_A_count": actions["commit_A"],
        "commit_B_count": actions["commit_B"],
    }


def run_evaluation(
    specs: list[BenchmarkSpec],
    models: dict[int, MAPPOActorCritic],
    destination: Path,
    *,
    episodes_per_seed_condition: int,
) -> list[dict[str, object]]:
    if episodes_per_seed_condition < 2 or episodes_per_seed_condition % 2:
        raise ValueError("episodes_per_seed_condition must be a positive even number")
    episodes_path = destination / "evaluation_episodes.csv"
    rows: list[dict[str, object]] = []
    ride_policies = {seed: RIDeDefenderAdapter(RIDE_CHECKPOINT) for seed in SEEDS}
    for seed in SEEDS:
        for spec_index, spec in enumerate(specs):
            for method_index, method in enumerate(METHODS):
                for episode in range(episodes_per_seed_condition):
                    # Paired across methods; attacker randomness differs only through
                    # controller-dependent adaptive behavior.
                    episode_seed = seed * 100_000_000 + spec_index * 100_000 + episode
                    policy = _policy(
                        method,
                        episode_seed=episode_seed,
                        spec=spec,
                        mappo_model=models[seed],
                        ride_policy=ride_policies[seed],
                    )
                    rows.append(
                        evaluate_episode(
                            spec,
                            method,
                            policy,
                            seed=seed,
                            episode=episode,
                            episode_seed=episode_seed,
                        )
                    )
            print(f"evaluation seed={seed} condition={spec.key} complete", flush=True)
    write_csv(episodes_path, rows)
    return rows


def _mean_std(values) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _seed_metrics(rows: list[dict[str, object]], group_keys: tuple[str, ...]):
    grouped: defaultdict[tuple, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in ("seed",) + group_keys)
        grouped[key].append(row)
    result = []
    for key, selected in grouped.items():
        output = {name: value for name, value in zip(("seed",) + group_keys, key)}
        for metric in ("interception", "regret", "observation_cost", "calibration"):
            output[metric] = float(np.mean([float(item[metric]) for item in selected]))
        output["episodes"] = len(selected)
        result.append(output)
    return result


def _aggregate_seed_metrics(seed_rows, group_keys):
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    outputs = []
    for key, selected in sorted(grouped.items()):
        output = {name: value for name, value in zip(group_keys, key)}
        output["seeds"] = len(selected)
        output["episodes"] = sum(int(item["episodes"]) for item in selected)
        for metric in ("interception", "regret", "observation_cost", "calibration"):
            mean, std = _mean_std(item[metric] for item in selected)
            output[f"{metric}_mean"] = mean
            output[f"{metric}_std"] = std
            output[f"{metric}_mean_std"] = f"{mean:.3f}±{std:.3f}"
        outputs.append(output)
    return outputs


def _ece(rows: list[dict[str, object]], bins: int = 10) -> float:
    probabilities = np.asarray([float(row["probability_B"]) for row in rows])
    labels = np.asarray([float(row["true_goal"] == "B") for row in rows])
    edges = np.linspace(0, 1, bins + 1)
    total = len(rows)
    value = 0.0
    for index in range(bins):
        selected = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1] if index == bins - 1 else probabilities < edges[index + 1]
        )
        if selected.any():
            value += float(selected.mean()) * abs(float(probabilities[selected].mean() - labels[selected].mean()))
    return value


def build_results(rows: list[dict[str, object]], destination: Path):
    per_family_seed = _seed_metrics(rows, ("method", "environment", "attack_family"))
    per_family = _aggregate_seed_metrics(
        per_family_seed, ("method", "environment", "attack_family")
    )
    main_seed = _seed_metrics(rows, ("method", "environment"))
    main = _aggregate_seed_metrics(main_seed, ("method", "environment"))
    calibration = []
    regret = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["environment"], row["attack_family"])].append(row)
    for key, selected in sorted(grouped.items()):
        calibration.append(
            {
                "method": key[0],
                "environment": key[1],
                "attack_family": key[2],
                "episodes": len(selected),
                "brier_score": float(np.mean([float(row["calibration"]) for row in selected])),
                "expected_calibration_error": _ece(selected),
                "negative_log_likelihood": float(
                    np.mean([float(row["negative_log_likelihood"]) for row in selected])
                ),
            }
        )
        values = np.asarray([float(row["regret"]) for row in selected])
        successful = [float(row["regret"]) for row in selected if row["interception"]]
        failed = [float(row["regret"]) for row in selected if not row["interception"]]
        regret.append(
            {
                "method": key[0],
                "environment": key[1],
                "attack_family": key[2],
                "episodes": len(selected),
                "regret_mean": float(values.mean()),
                "regret_std": float(values.std(ddof=1)),
                "regret_median": float(np.median(values)),
                "regret_p90": float(np.quantile(values, 0.9)),
                "successful_episode_regret": float(np.mean(successful)) if successful else None,
                "failed_episode_regret": float(np.mean(failed)) if failed else None,
            }
        )
    write_csv(destination / "main_results.csv", main)
    write_csv(destination / "per_family_results.csv", per_family)
    write_csv(destination / "calibration_results.csv", calibration)
    write_csv(destination / "regret_analysis.csv", regret)
    write_csv(destination / "per_seed_results.csv", per_family_seed)
    return main, per_family, calibration, regret


def convergence_rows(all_curves: dict[int, list[dict[str, object]]]):
    rows = []
    for seed, curves in sorted(all_curves.items()):
        numeric = [float(row["evaluation_success"]) for row in curves]
        window = min(5, max(1, len(numeric) // 2))
        final = float(np.mean(numeric[-window:]))
        previous = float(np.mean(numeric[-2 * window : -window])) if len(numeric) >= 2 * window else numeric[0]
        rows.append(
            {
                "seed": seed,
                "timesteps": int(float(curves[-1]["timesteps"])),
                "updates": len(curves),
                "final_window_success": final,
                "previous_window_success": previous,
                "absolute_change": abs(final - previous),
                "converged_at_0.02_tolerance": int(abs(final - previous) <= 0.02),
            }
        )
    return rows


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def generate_tables(main, per_family, destination: Path) -> None:
    table_dir = destination / "paper_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Method & Environment & Interception & Regret & Cost & Brier \\",
        r"\midrule",
    ]
    for row in main:
        lines.append(
            f"{_latex_escape(row['method'])} & {_latex_escape(row['environment'])} & "
            f"{row['interception_mean_std']} & {row['regret_mean_std']} & "
            f"{row['observation_cost_mean_std']} & {row['calibration_mean_std']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (table_dir / "main_results.tex").write_text("\n".join(lines) + "\n")
    lines = [
        r"\begin{tabular}{lllrr}",
        r"\toprule",
        r"Method & Environment & Family & Interception & Regret \\",
        r"\midrule",
    ]
    for row in per_family:
        lines.append(
            f"{_latex_escape(row['method'])} & {_latex_escape(row['environment'])} & "
            f"{_latex_escape(row['attack_family'])} & {row['interception_mean_std']} & "
            f"{row['regret_mean_std']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (table_dir / "per_family_results.tex").write_text("\n".join(lines) + "\n")


def generate_figures(main, per_family, calibration, curves, destination: Path) -> None:
    figure_dir = destination / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(METHODS)))
    for metric, label in (("interception", "Interception success"), ("regret", "Decision regret")):
        fig, axes = plt.subplots(1, len(ENVIRONMENTS), figsize=(16, 4), sharey=True)
        for axis, environment in zip(axes, ENVIRONMENTS):
            selected = {row["method"]: row for row in main if row["environment"] == environment}
            values = [selected[method][f"{metric}_mean"] for method in METHODS]
            errors = [selected[method][f"{metric}_std"] for method in METHODS]
            axis.bar(range(len(METHODS)), values, yerr=errors, color=colors, capsize=2)
            axis.set_title(environment.replace("_", " "))
            axis.set_xticks(range(len(METHODS)), [name.replace("_", "\n") for name in METHODS], rotation=35)
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel(label)
        fig.tight_layout()
        fig.savefig(figure_dir / f"main_{metric}.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(ENVIRONMENTS), figsize=(16, 4), sharey=True)
    for axis, environment in zip(axes, ENVIRONMENTS):
        for method, color in zip(METHODS, colors):
            selected = [
                row for row in per_family if row["environment"] == environment and row["method"] == method
            ]
            selected.sort(key=lambda row: ATTACK_SETTINGS.index(row["attack_family"]))
            axis.plot(
                ATTACK_SETTINGS,
                [row["interception_mean"] for row in selected],
                marker="o",
                label=method,
                color=color,
            )
        axis.set_title(environment.replace("_", " "))
        axis.tick_params(axis="x", rotation=25)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Interception success")
    axes[-1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(figure_dir / "per_family_interception.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    for seed, rows in sorted(curves.items()):
        axis.plot(
            [float(row["timesteps"]) for row in rows],
            [float(row["evaluation_success"]) for row in rows],
            alpha=0.55,
            label=f"seed {seed}",
        )
    axis.set_xlabel("MAPPO training timesteps")
    axis.set_ylabel("Deterministic validation success")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "mappo_training_curves.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(METHODS))
    values = []
    for method in METHODS:
        selected = [row for row in calibration if row["method"] == method]
        values.append(float(np.mean([row["expected_calibration_error"] for row in selected])))
    axis.bar(positions, values, color=colors)
    axis.set_xticks(positions, METHODS, rotation=25)
    axis.set_ylabel("Expected calibration error")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "calibration_comparison.png", dpi=180)
    plt.close(fig)


def protected_hashes() -> dict[str, str]:
    candidates = [
        ROOT / "src/deception_rl/envs/gridworld.py",
        ROOT / "src/deception_rl/envs/forked_routes.py",
        ROOT / "src/deception_rl/external_baselines/policies.py",
        ROOT / "outputs/stage13_learned_controller/stage13_controller.pt",
        ROOT / "outputs/data_scaling/modela.pt",
        ROOT / "outputs/data_scaling/modelb.pt",
        Path("/workspace/stage10-two-level/belief_model.pt"),
        Path("/workspace/stage10-1-decoupled/calibrator.pt"),
    ]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in candidates}


def run(
    destination: Path = OUTPUT,
    *,
    training_steps: int = 100_000,
    episodes_per_seed_condition: int = 100,
    device: str | torch.device = "cpu",
) -> list[dict[str, object]]:
    device = resolve_device(device)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    before = protected_hashes()
    specs = benchmark_specs()
    models = {}
    curves = {}
    for seed in SEEDS:
        models[seed], curves[seed] = train_mappo_seed(
            seed, specs, destination, training_steps=training_steps, device=device
        )
    all_curves = [row for seed in SEEDS for row in curves[seed]]
    write_csv(destination / "training_curves.csv", all_curves)
    convergence = convergence_rows(curves)
    write_csv(destination / "convergence.csv", convergence)
    episodes_path = destination / "evaluation_episodes.csv"
    if episodes_path.exists():
        rows = read_csv(episodes_path)
        expected = len(SEEDS) * len(specs) * len(METHODS) * episodes_per_seed_condition
        if len(rows) != expected:
            rows = run_evaluation(
                specs,
                models,
                destination,
                episodes_per_seed_condition=episodes_per_seed_condition,
            )
    else:
        rows = run_evaluation(
            specs,
            models,
            destination,
            episodes_per_seed_condition=episodes_per_seed_condition,
        )
    main, per_family, calibration, regret = build_results(rows, destination)
    generate_tables(main, per_family, destination)
    generate_figures(main, per_family, calibration, curves, destination)
    config = {
        "stage": "17.3",
        "seeds": list(SEEDS),
        "environments": list(ENVIRONMENTS),
        "methods": list(METHODS),
        "attack_settings": list(ATTACK_SETTINGS),
        "episodes_per_seed_method_environment_attack": episodes_per_seed_condition,
        "episodes_per_method_environment_attack": episodes_per_seed_condition * len(SEEDS),
        "total_evaluation_episodes": len(rows),
        "goal_balance": "50/50 A/B within every seed, method, environment, and attack",
        "mappo": {
            "training_steps_per_seed": training_steps,
            "total_training_budget": training_steps * len(SEEDS),
            "rollout_steps": 2048,
            "ppo_epochs": 4,
            "minibatch_size": 256,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "clip_range": 0.2,
            "entropy_coefficient": 0.01,
            "training_families": ["scripted", "adaptive"],
            "unseen_family_used_for_training": False,
        },
        "ride": {
            "checkpoint": str(RIDE_CHECKPOINT),
            "checkpoint_frozen": True,
            "external_counterfactual_input": "fixed belief-confidence/reversal likelihood-ratio proxy",
            "fine_tuning": False,
        },
        "regret_definition": "1 - (interception - cumulative observation cost)",
        "calibration_primary": "Brier score for P(goal=B) at first commitment or terminal horizon",
        "paper_writing": False,
    }
    if device.type == "cuda":
        config["mappo"]["device"] = str(device)
        config["mappo"]["gpu_name"] = torch.cuda.get_device_name(device)
        config["mappo"]["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    after = protected_hashes()
    if before != after:
        raise AssertionError("A protected environment, algorithm, or RIDe checkpoint changed")
    (destination / "preserved_inputs.json").write_text(
        json.dumps({"sha256_before": before, "sha256_after": after, "unchanged": True}, indent=2) + "\n"
    )
    return main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="MAPPO device: cpu (default), cuda, or cuda:N; other methods remain on CPU")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--training-steps", type=int, default=100_000)
    parser.add_argument("--episodes-per-seed-condition", type=int, default=100)
    arguments = parser.parse_args()
    result = run(
        arguments.output_dir,
        training_steps=arguments.training_steps,
        episodes_per_seed_condition=arguments.episodes_per_seed_condition,
        device=arguments.device,
    )
    print(json.dumps(result, indent=2))
