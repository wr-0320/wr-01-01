"""Create paper figures from the normalized evaluation CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_MODEL_COLORS = {"ModelA": "#176B87", "ModelB": "#D97732"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Run experiments.evaluate_paper first; missing {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _binary_interval(
    episodes: list[dict[str, str]], field: str
) -> tuple[float, float]:
    values = [int(row[field]) for row in episodes if row[field] != ""]
    return wilson_interval(sum(values), len(values))


def _save_generalization(
    summary: list[dict[str, str]], episodes: list[dict[str, str]], figure_dir: Path
) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    stage_rows = [
        row
        for row in summary
        if row["stage"] == "Stage 6" and row["experiment"] == "defender_adaptation"
    ]
    for model in ("ModelA", "ModelB"):
        model_rows = sorted(
            (row for row in stage_rows if row["model"] == model),
            key=lambda row: int(row["deception_steps"]),
        )
        x_values = [int(row["deception_steps"]) for row in model_rows]
        y_values = [float(row["commitment_accuracy"]) for row in model_rows]
        lower_errors = []
        upper_errors = []
        for row in model_rows:
            matched = [
                episode
                for episode in episodes
                if episode["stage"] == "Stage 6"
                and episode["model"] == model
                and episode["condition"] == row["condition"]
            ]
            lower, upper = _binary_interval(matched, "commitment_accuracy")
            mean = float(row["commitment_accuracy"])
            lower_errors.append(mean - lower)
            upper_errors.append(upper - mean)
        axis.errorbar(
            x_values,
            y_values,
            yerr=[lower_errors, upper_errors],
            marker="o",
            linewidth=2,
            capsize=3,
            color=_MODEL_COLORS[model],
            label=model,
        )

    axis.axvspan(4.5, 5.5, color="#777777", alpha=0.10, label="Unseen parameter")
    axis.set(xlabel="Scripted deception steps", ylabel="Commitment accuracy")
    axis.set_xticks([1, 2, 3, 4, 5])
    axis.set_ylim(-0.04, 1.04)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncols=3, loc="lower left")
    axis.set_title("Generalization across deception duration")
    path = figure_dir / "model_a_vs_b_generalization.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def _save_success_curve(
    summary: list[dict[str, str]], episodes: list[dict[str, str]], figure_dir: Path
) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    stage_rows = [
        row
        for row in summary
        if row["stage"] == "Stage 6" and row["experiment"] == "defender_adaptation"
    ]
    for model in ("ModelA", "ModelB"):
        model_rows = sorted(
            (row for row in stage_rows if row["model"] == model),
            key=lambda row: int(row["deception_steps"]),
        )
        x_values = [int(row["deception_steps"]) for row in model_rows]
        y_values = [float(row["deception_success_rate"]) for row in model_rows]
        lower_errors = []
        upper_errors = []
        for row in model_rows:
            matched = [
                episode
                for episode in episodes
                if episode["stage"] == "Stage 6"
                and episode["model"] == model
                and episode["condition"] == row["condition"]
            ]
            lower, upper = _binary_interval(matched, "deceiver_success")
            mean = float(row["deception_success_rate"])
            lower_errors.append(mean - lower)
            upper_errors.append(upper - mean)
        axis.errorbar(
            x_values,
            y_values,
            yerr=[lower_errors, upper_errors],
            marker="s",
            linewidth=2,
            capsize=3,
            color=_MODEL_COLORS[model],
            label=model,
        )

    axis.axvspan(4.5, 5.5, color="#777777", alpha=0.10, label="Unseen parameter")
    axis.set(xlabel="Scripted deception steps", ylabel="Deceiver success rate")
    axis.set_xticks([1, 2, 3, 4, 5])
    axis.set_ylim(-0.04, 1.04)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncols=3, loc="upper left")
    axis.set_title("Deception success by feint duration")
    path = figure_dir / "deception_steps_vs_success_rate.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def _save_exposure_robustness(
    summary: list[dict[str, str]], episodes: list[dict[str, str]], figure_dir: Path
) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    stage_styles = {"Stage 5": "o", "Stage 6": "s"}
    plotted = False
    for row in summary:
        if not row["condition"].startswith("unseen_"):
            continue
        robustness = row["robustness_under_unseen_deception"]
        if robustness == "":
            continue
        matched = [
            episode
            for episode in episodes
            if episode["stage"] == row["stage"]
            and episode["model"] == row["model"]
            and episode["condition"] == row["condition"]
        ]
        lower, upper = _binary_interval(matched, "commitment_accuracy")
        mean = float(robustness)
        marker = stage_styles.get(row["stage"], "^")
        axis.errorbar(
            float(row["deception_exposure"]),
            mean,
            yerr=[[mean - lower], [upper - mean]],
            marker=marker,
            markersize=7,
            capsize=4,
            linestyle="none",
            color=_MODEL_COLORS[row["model"]],
            label=(
                f"{row['stage']} {row['model']} "
                f"(unseen steps={row['deception_steps']})"
            ),
        )
        plotted = True

    if not plotted:
        raise ValueError("No unseen-deception evaluation rows found in paper summary.")
    axis.set_xscale("symlog", linthresh=10)
    axis.set(xlabel="Scripted-deception training episodes", ylabel="Unseen commitment accuracy")
    axis.set_ylim(-0.04, 1.04)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8, loc="best")
    axis.set_title("Robustness versus deception exposure")
    path = figure_dir / "deception_exposure_vs_robustness.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def _save_family_robustness(
    summary: list[dict[str, str]], episodes: list[dict[str, str]], figure_dir: Path
) -> Path:
    family_rows = [
        row
        for row in summary
        if row["stage"] == "Stage 6B"
        and row["experiment"] == "unseen_deception_families"
    ]
    labels = []
    for family, steps in (
        ("scripted_extrapolation", "6"),
        ("scripted_extrapolation", "7"),
        ("delayed_deception", ""),
        ("detour_deception", ""),
        ("multi_switch_deception", ""),
    ):
        if any(
            row["deception_family"] == family
            and str(row["deception_steps"]) == steps
            for row in family_rows
        ):
            label = f"Scripted {steps}" if steps else family.replace("_", " ")
            labels.append((family, steps, label))
    if not labels:
        raise ValueError("No unseen-family rows found; run the family generalization experiment.")

    figure, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    x_values = list(range(len(labels)))
    offsets = {"ModelA": -0.08, "ModelB": 0.08}
    for model in ("ModelA", "ModelB"):
        means = []
        lower_errors = []
        upper_errors = []
        positions = []
        for index, (family, steps, _) in enumerate(labels):
            row = next(
                item
                for item in family_rows
                if item["model"] == model
                and item["deception_family"] == family
                and str(item["deception_steps"]) == steps
            )
            matched = [
                episode
                for episode in episodes
                if episode["stage"] == "Stage 6B"
                and episode["model"] == model
                and episode["deception_family"] == family
                and str(episode["deception_steps"]) == steps
            ]
            lower, upper = _binary_interval(matched, "commitment_accuracy")
            mean = float(row["commitment_accuracy"])
            means.append(mean)
            lower_errors.append(mean - lower)
            upper_errors.append(upper - mean)
            positions.append(x_values[index] + offsets[model])
        axis.errorbar(
            positions,
            means,
            yerr=[lower_errors, upper_errors],
            marker="o" if model == "ModelA" else "s",
            linewidth=1.8,
            capsize=3,
            color=_MODEL_COLORS[model],
            label=model,
        )

    axis.set_xticks(x_values, [label for _, _, label in labels], rotation=18, ha="right")
    axis.set(xlabel="Unseen deception family", ylabel="Commitment accuracy")
    axis.set_ylim(-0.04, 1.04)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("Robustness across unseen deception families")
    path = figure_dir / "robustness_unseen_deception_families.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def plot_paper(
    evaluation_dir: str | Path = "outputs/paper",
    figure_dir: str | Path = "outputs/figures",
) -> list[Path]:
    evaluation_dir = Path(evaluation_dir)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_csv(evaluation_dir / "evaluation_summary.csv")
    episodes = _read_csv(evaluation_dir / "evaluation_episodes.csv")
    return [
        _save_generalization(summary, episodes, figure_dir),
        _save_success_curve(summary, episodes, figure_dir),
        _save_exposure_robustness(summary, episodes, figure_dir),
        _save_family_robustness(summary, episodes, figure_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--figure-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    figures = plot_paper(args.evaluation_dir, args.figure_dir)
    for figure in figures:
        print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
