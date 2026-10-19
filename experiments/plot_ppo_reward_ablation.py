"""Plot reward-ablation learning curves and learned trajectory examples."""

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
    "A_task_only": "#176B87",
    "B_task_plus_deception": "#D97732",
    "C_task_deception_cost": "#568B52",
}
_LABELS = {
    "A_task_only": "A: task only",
    "B_task_plus_deception": "B: task + deception",
    "C_task_deception_cost": "C: task + deception + cost",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _rolling_mean(values: list[float], window: int = 20) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return array
    width = min(window, array.size)
    return np.convolve(array, np.ones(width) / width, mode="valid")


def plot_learning_curves(curves_csv: Path, output_path: Path) -> Path:
    rows = _read_rows(curves_csv)
    metrics = (
        ("episode_reward", "Episode reward"),
        ("task_reward", "Task reward"),
        ("deception_reward", "Deception reward"),
        ("movement_cost", "Movement cost"),
        ("rolling_success_rate", "Rolling success rate"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    flat_axes = axes.flat
    for axis, (metric, label) in zip(flat_axes, metrics):
        for ablation in _LABELS:
            selected = sorted(
                (row for row in rows if row["ablation"] == ablation),
                key=lambda row: int(row["timesteps"]),
            )
            if not selected:
                continue
            values = [float(row[metric]) for row in selected]
            smoothed = _rolling_mean(values)
            steps = [int(row["timesteps"]) for row in selected][-len(smoothed) :]
            axis.plot(
                steps,
                smoothed,
                color=_COLORS[ablation],
                linewidth=1.8,
                label=_LABELS[ablation],
            )
        axis.set_xlabel("Training timesteps")
        axis.set_ylabel(label)
        axis.set_title(label)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    axes.flat[-1].set_visible(False)
    figure.suptitle("PPO Deceiver reward ablation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=250)
    plt.close(figure)
    return output_path


def plot_learned_trajectories(episodes_csv: Path, output_path: Path) -> Path:
    rows = [
        row
        for row in _read_rows(episodes_csv)
        if row["policy"] == "ppo_learned_deceiver"
    ]
    figure, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
    for row_index, ablation in enumerate(_LABELS):
        for column_index, goal in enumerate(("A", "B")):
            axis = axes[row_index, column_index]
            selected = next(
                row
                for row in rows
                if row["ablation"] == ablation and row["true_goal"] == goal
            )
            trajectory = np.asarray(json.loads(selected["trajectory"]), dtype=np.int64)
            axis.plot(
                trajectory[:, 1],
                trajectory[:, 0],
                marker="o",
                color=_COLORS[ablation],
                linewidth=2,
            )
            axis.scatter([0], [0], marker="*", s=150, color="#111111", label="A")
            axis.scatter([6], [6], marker="*", s=150, color="#C13B36", label="B")
            axis.set_xlim(-0.5, 6.5)
            axis.set_ylim(6.5, -0.5)
            axis.set_xticks(range(7))
            axis.set_yticks(range(7))
            axis.set_aspect("equal")
            axis.grid(alpha=0.3)
            axis.set_title(f"{_LABELS[ablation]} | true goal {goal}")
            axis.set_xlabel("Column")
            axis.set_ylabel("Row")
            axis.legend(frameon=False, loc="upper right")
    figure.suptitle("Learned trajectory examples by reward ablation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=250)
    plt.close(figure)
    return output_path


def plot_reward_ablation(
    learning_curves_csv: str | Path,
    evaluation_episodes_csv: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    return (
        plot_learning_curves(
            Path(learning_curves_csv), output_dir / "learning_curves.png"
        ),
        plot_learned_trajectories(
            Path(evaluation_episodes_csv), output_dir / "learned_trajectory_examples.png"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ppo_reward_ablation"))
    args = parser.parse_args()
    figures = plot_reward_ablation(
        args.output_dir / "learning_curves.csv",
        args.output_dir / "evaluation_episodes.csv",
        args.output_dir,
    )
    for figure in figures:
        print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
