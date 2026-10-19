"""Evaluate red deception and blue commitment baselines on intelligence reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from deception_rl.agents.intelligence_policies import (
    AdaptiveDeceptionRedPolicy,
    AdaptiveVerificationBluePolicy,
    ConcealmentRedPolicy,
    DecoyRedPolicy,
    DelayedDeceptionRedPolicy,
    FixedTimeBluePolicy,
    HonestRedPolicy,
    SwitchDeceptionRedPolicy,
    policy_action_name,
)
from deception_rl.envs.deceptive_intelligence import (
    NUM_HYPOTHESES,
    DeceptiveIntelligenceEnv,
)


def default_red_policies() -> list[Any]:
    return [
        HonestRedPolicy(),
        ConcealmentRedPolicy(),
        DecoyRedPolicy(),
        DecoyRedPolicy(multisource=True),
        DelayedDeceptionRedPolicy(onset=3),
        SwitchDeceptionRedPolicy(switch_step=3),
        AdaptiveDeceptionRedPolicy(),
    ]


def default_blue_policies(max_cycles: int) -> list[Any]:
    return [
        FixedTimeBluePolicy(commit_step=4),
        AdaptiveVerificationBluePolicy(max_cycles=max_cycles),
    ]


def run_episode(
    red_policy: Any,
    blue_policy: Any,
    *,
    hypothesis: int,
    seed: int,
    max_cycles: int,
) -> dict[str, Any]:
    env = DeceptiveIntelligenceEnv(max_cycles=max_cycles)
    red_policy.reset()
    blue_policy.reset()
    observations, _ = env.reset(seed=seed, options={"hypothesis": hypothesis})
    action_trace: list[dict[str, Any]] = []
    belief_trace: list[dict[str, Any]] = []
    total_rewards = {"red": 0.0, "blue": 0.0}
    metrics: dict[str, Any] | None = None

    while env.agents:
        red_action = red_policy.act(observations["red"])
        blue_action = blue_policy.act(observations["blue"])
        belief = getattr(blue_policy, "last_belief", None)
        if belief is not None:
            belief_trace.append(
                {
                    "step": int(observations["blue"]["current_step"][0]),
                    "hypotheses": belief.hypothesis_probabilities.round(6).tolist(),
                    "deception_probability": round(belief.deception_probability, 6),
                    "confidence": round(belief.confidence, 6),
                    "entropy": round(belief.entropy, 6),
                }
            )
        action_trace.append(
            {
                "step": len(action_trace),
                "red": policy_action_name(red_action, "red"),
                "blue": policy_action_name(blue_action, "blue"),
            }
        )
        observations, rewards, _, _, infos = env.step(
            {"red": red_action, "blue": blue_action}
        )
        for agent in total_rewards:
            total_rewards[agent] += rewards[agent]
        if "episode_metrics" in infos["blue"]:
            metrics = infos["blue"]["episode_metrics"]

    if metrics is None:
        raise RuntimeError("episode ended without terminal metrics")
    final_belief = belief_trace[-1] if belief_trace else {}
    return {
        "red_policy": red_policy.name,
        "blue_policy": blue_policy.name,
        "seed": seed,
        **metrics,
        "blue_total_reward": round(total_rewards["blue"], 6),
        "red_total_reward": round(total_rewards["red"], 6),
        "final_confidence": final_belief.get("confidence", 1 / NUM_HYPOTHESES),
        "final_deception_probability": final_belief.get(
            "deception_probability", 0.5
        ),
        "action_trace": action_trace,
        "belief_trace": belief_trace,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], json_fields=()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            serialized = row.copy()
            for field in json_fields:
                serialized[field] = json.dumps(
                    serialized[field], separators=(",", ":"), sort_keys=True
                )
            writer.writerow(serialized)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["red_policy"], row["blue_policy"])].append(row)
    summaries = []
    for (red_policy, blue_policy), group in sorted(groups.items()):
        correct = np.array([row["decision_correct"] for row in group], dtype=float)
        confidence = np.array([row["final_confidence"] for row in group], dtype=float)
        summaries.append(
            {
                "red_policy": red_policy,
                "blue_policy": blue_policy,
                "episodes": len(group),
                "decision_accuracy": round(float(correct.mean()), 6),
                "mean_decision_step": round(
                    float(np.mean([row["decision_step"] for row in group])), 6
                ),
                "deception_success_rate": round(
                    float(np.mean([row["deception_succeeded"] for row in group])), 6
                ),
                "timeout_rate": round(
                    float(np.mean([row["timed_out"] for row in group])), 6
                ),
                "mean_information_cost": round(
                    float(np.mean([row["blue_information_cost"] for row in group])), 6
                ),
                "mean_decision_regret": round(
                    float(np.mean([row["decision_regret"] for row in group])), 6
                ),
                "mean_confidence": round(float(confidence.mean()), 6),
                "confidence_absolute_error": round(
                    float(np.mean(np.abs(confidence - correct))), 6
                ),
            }
        )
    return summaries


def run_experiment(
    output_dir: str | Path,
    *,
    episodes_per_hypothesis: int = 10,
    seed: int = 0,
    max_cycles: int = 10,
    red_policies: list[Any] | None = None,
    blue_policies: list[Any] | None = None,
) -> tuple[Path, Path]:
    if episodes_per_hypothesis < 1:
        raise ValueError("episodes_per_hypothesis must be positive")
    if max_cycles < 3:
        raise ValueError("max_cycles must be at least 3")
    red_policies = red_policies or default_red_policies()
    blue_policies = blue_policies or default_blue_policies(max_cycles)
    rows = []
    for red_index, red_policy in enumerate(red_policies):
        for blue_index, blue_policy in enumerate(blue_policies):
            for hypothesis in range(NUM_HYPOTHESES):
                for episode in range(episodes_per_hypothesis):
                    episode_seed = (
                        seed
                        + red_index * 1_000_003
                        + blue_index * 100_003
                        + hypothesis * 1_003
                        + episode
                    )
                    rows.append(
                        run_episode(
                            red_policy,
                            blue_policy,
                            hypothesis=hypothesis,
                            seed=episode_seed,
                            max_cycles=max_cycles,
                        )
                    )
    destination = Path(output_dir)
    episode_path = destination / "episodes.csv"
    summary_path = destination / "summary.csv"
    _write_csv(episode_path, rows, json_fields=("action_trace", "belief_trace"))
    _write_csv(summary_path, summarize(rows))
    return episode_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/deceptive_intelligence")
    )
    parser.add_argument("--episodes-per-hypothesis", type=int, default=10)
    parser.add_argument("--max-cycles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    episode_path, summary_path = run_experiment(
        args.output_dir,
        episodes_per_hypothesis=args.episodes_per_hypothesis,
        max_cycles=args.max_cycles,
        seed=args.seed,
    )
    print(f"Wrote episode results to {episode_path}")
    print(f"Wrote aggregate results to {summary_path}")


if __name__ == "__main__":
    main()

