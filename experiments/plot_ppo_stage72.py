"""Plot Stage7.2 reward curves and belief-observation ablation results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_EXPERIMENTS = (
    "Experiment1_PPO_no_belief",
    "Experiment2_PPO_with_belief",
    "Experiment3_belief_curriculum",
)
_COLORS = {
    _EXPERIMENTS[0]: "#176B87",
    _EXPERIMENTS[1]: "#D97732",
    _EXPERIMENTS[2]: "#568B52",
}
_LABELS = {
    _EXPERIMENTS[0]: "PPO-no-belief",
    _EXPERIMENTS[1]: "PPO-with-belief",
    _EXPERIMENTS[2]: "Belief + reward curriculum",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _smooth(values: list[float], window: int = 20) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return array
    width = min(window, array.size)
    return np.convolve(array, np.ones(width) / width, mode="valid")


def plot_learning_curves(curves_csv: Path, output_path: Path) -> Path:
    rows = _read_rows(curves_csv)
    metrics = (
        ("episode_reward", "Episode return"),
        ("task_reward", "Task reward"),
        ("deception_reward", "Deception reward"),
        ("movement_cost", "Movement cost"),
        ("rolling_success_rate", "Rolling task success rate"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, metrics):
        for experiment in _EXPERIMENTS:
            selected = sorted(
                (row for row in rows if row["experiment"] == experiment),
                key=lambda row: int(row["timestep"]),
            )
            if not selected:
                continue
            smoothed = _smooth([float(row[metric]) for row in selected])
            timesteps = [int(row["timestep"]) for row in selected][-len(smoothed) :]
            axis.plot(
                timesteps,
                smoothed,
                color=_COLORS[experiment],
                label=_LABELS[experiment],
                linewidth=1.8,
            )
        axis.set(xlabel="Timesteps", ylabel=label, title=label)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    axes.flat[-1].set_visible(False)
    figure.suptitle("Stage 7.2 PPO reward curriculum")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=250)
    plt.close(figure)
    return output_path


def plot_belief_ablation(episodes_csv: Path, output_path: Path) -> Path:
    rows = [
        row
        for row in _read_rows(episodes_csv)
        if row["policy"] == "ppo_learned_deceiver"
    ]
    metrics = (
        ("deceiver_success", "Task success rate"),
        ("commitment_accuracy", "Defender commitment accuracy"),
        ("misdirection_rate", "Misdirection rate"),
        ("belief_accuracy", "Defender belief accuracy"),
        ("episode_return", "Episode return"),
    )
    figure, axes = plt.subplots(1, len(metrics), figsize=(16, 4.8), constrained_layout=True)
    for axis, (metric, label) in zip(axes, metrics):
        means = []
        errors = []
        for experiment in _EXPERIMENTS:
            values = [
                float(row[metric])
                for row in rows
                if row["experiment"] == experiment
            ]
            means.append(float(np.mean(values)) if values else 0.0)
            errors.append(
                float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
        positions = np.arange(len(_EXPERIMENTS))
        bars = axis.bar(
            positions,
            means,
            yerr=errors,
            color=[_COLORS[experiment] for experiment in _EXPERIMENTS],
            capsize=3,
        )
        axis.set_xticks(positions, ["No belief", "With belief", "Belief +\ncurriculum"], rotation=0)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
        for bar, mean in zip(bars, means):
            axis.annotate(
                f"{mean:.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    figure.suptitle("PPO execution-observation ablation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=250)
    plt.close(figure)
    return output_path


def plot_stage72(
    learning_curves_csv: str | Path,
    evaluation_episodes_csv: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    return (
        plot_learning_curves(
            Path(learning_curves_csv), output_dir / "learning_curves.png"
        ),
        plot_belief_ablation(
            Path(evaluation_episodes_csv), output_dir / "belief_ablation.png"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ppo_stage72"))
    args = parser.parse_args()
    figures = plot_stage72(
        args.output_dir / "learning_curves.csv",
        args.output_dir / "evaluation_episodes.csv",
        args.output_dir,
    )
    for figure in figures:
        print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
