"""Visualize representative baseline and PPO Deceiver trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_COLORS = {
    "honest_shortest_path": "#176B87",
    "random_policy": "#888888",
    "scripted_deception": "#D97732",
    "ppo_learned_deceiver": "#568B52",
}


def plot_trajectories(
    episode_csv: str | Path,
    output_path: str | Path,
    episodes_per_goal: int = 1,
) -> Path:
    with Path(episode_csv).open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    policies = list(dict.fromkeys(row["policy"] for row in rows))
    goals = ("A", "B")
    figure, axes = plt.subplots(1, len(goals), figsize=(6.2 * len(goals), 6.0))
    if len(goals) == 1:
        axes = [axes]

    for axis, goal in zip(axes, goals):
        axis.set_xlim(-0.5, 6.5)
        axis.set_ylim(6.5, -0.5)
        axis.set_xticks(range(7))
        axis.set_yticks(range(7))
        axis.grid(True, linewidth=0.6, alpha=0.45)
        axis.set_aspect("equal")
        axis.scatter([0], [0], marker="*", s=190, color="#111111", label="A")
        axis.scatter([6], [6], marker="*", s=190, color="#C13B36", label="B")
        for policy in policies:
            examples = [
                row
                for row in rows
                if row["policy"] == policy and row["true_goal"] == goal
            ][:episodes_per_goal]
            for example_index, row in enumerate(examples):
                trajectory = np.asarray(json.loads(row["trajectory"]), dtype=np.int64)
                axis.plot(
                    trajectory[:, 1],
                    trajectory[:, 0],
                    marker="o",
                    markersize=3,
                    linewidth=1.8,
                    alpha=0.8,
                    color=_COLORS[policy],
                    label=policy if example_index == 0 else None,
                )
        axis.set_title(f"True goal {goal}")
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")
        axis.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncols=2)

    figure.suptitle("Deceiver trajectory diagnostics against frozen Model A")
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=250, bbox_inches="tight")
    plt.close(figure)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=Path("outputs/ppo_deceiver/evaluation_episodes.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ppo_deceiver/trajectory_diagnostics.png"))
    args = parser.parse_args()
    print(f"Wrote {plot_trajectories(args.episodes, args.output)}")


if __name__ == "__main__":
    main()
