"""Plot counter-deception robustness against training diversity and exposure."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_COLORS = {"C1": "#176B87", "C2": "#D97732", "C3": "#568B52"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing exposure/diversity results: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def plot_exposure_diversity(
    diversity_csv: str | Path = "outputs/exposure_diversity/robustness_vs_training_diversity.csv",
    output_path: str | Path = "outputs/figures/robustness_vs_training_diversity.png",
) -> Path:
    rows = _read_rows(Path(diversity_csv))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    offsets = {"C1": -0.10, "C2": 0.10, "C3": 0.0}
    for row in rows:
        model = row["model"]
        family_count = int(row["training_family_count"])
        exposure = int(row["deception_training_episodes"])
        robustness = float(row["robustness_new_families"])
        seed_std = float(row["robustness_new_families_std_across_seeds"])
        marker_size = 85 + 90 * exposure / max(
            int(item["deception_training_episodes"]) for item in rows
        )
        x_value = family_count + offsets[model]
        axis.errorbar(
            x_value,
            robustness,
            yerr=seed_std,
            marker="o",
            markersize=math.sqrt(marker_size),
            capsize=4,
            linestyle="none",
            color=_COLORS[model],
            label=f"{model}: {exposure:,} deception episodes",
        )
        axis.annotate(
            f"{model} ({exposure:,})",
            (x_value, robustness),
            xytext=(4, 7),
            textcoords="offset points",
            fontsize=8,
            color=_COLORS[model],
        )

    axis.set(
        xlabel="Distinct deception families in training",
        ylabel="Unseen-family commitment accuracy",
    )
    axis.set_xticks([1, 4], ["1 family", "4 families"])
    axis.set_xlim(0.5, 4.5)
    axis.set_ylim(-0.04, 1.04)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="best")
    axis.set_title("Unseen-family robustness vs training diversity")
    figure.savefig(destination, dpi=300)
    plt.close(figure)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diversity-csv",
        type=Path,
        default=Path("outputs/exposure_diversity/robustness_vs_training_diversity.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/robustness_vs_training_diversity.png"),
    )
    args = parser.parse_args()
    print(
        f"Wrote {plot_exposure_diversity(args.diversity_csv, args.output)}"
    )


if __name__ == "__main__":
    main()
