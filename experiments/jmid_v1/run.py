"""Run the frozen exploratory JMID comparison."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .core import CONDITIONS, METHODS, diagnostics, fit_models, generate_dataset, metrics


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs" / "development" / "jmid_v1"
SEEDS = (2401, 2402, 2403)
SHIFT_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    metric_names = (
        "accuracy", "brier", "nll", "ece", "wrong_confidence",
        "high_confidence_error_rate",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = []
    for group, selected in grouped.items():
        record = dict(zip(keys, group))
        record["seeds"] = len(selected)
        for metric_name in metric_names:
            values = np.asarray([row[metric_name] for row in selected], dtype=float)
            record[f"{metric_name}_mean"] = float(values.mean())
            record[f"{metric_name}_min"] = float(values.min())
            record[f"{metric_name}_max"] = float(values.max())
        result.append(record)
    return result


def run(
    output: Path = DEFAULT_OUTPUT,
    *,
    seeds: tuple[int, ...] = SEEDS,
    train_episodes: int = 4000,
    test_episodes: int = 5000,
    smoke: bool = False,
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if smoke:
        train_episodes = min(train_episodes, 300)
        test_episodes = min(test_episodes, 400)

    condition_rows, diagnostic_rows, shift_rows = [], [], []
    for seed in seeds:
        for condition_index, (condition, strengths) in enumerate(CONDITIONS.items()):
            missing_strength, dependence_strength = strengths
            train = generate_dataset(
                train_episodes, seed * 100 + condition_index,
                missing_strength=missing_strength,
                dependence_strength=dependence_strength,
            )
            test = generate_dataset(
                test_episodes, seed * 1000 + condition_index,
                missing_strength=missing_strength,
                dependence_strength=dependence_strength,
            )
            models = fit_models(train)
            diagnostic_rows.append({
                "seed": seed, "condition": condition,
                "missing_strength": missing_strength,
                "dependence_strength": dependence_strength,
                **diagnostics(test),
            })
            for method in METHODS:
                condition_rows.append({
                    "seed": seed, "condition": condition, "method": method,
                    **metrics(test.z, models[method].predict_proba(test.y)),
                })

        shift_train = generate_dataset(
            train_episodes, seed * 100 + 90,
            missing_strength=0.2, dependence_strength=0.2,
        )
        shift_models = fit_models(shift_train)
        for shift_index, strength in enumerate(SHIFT_STRENGTHS):
            test = generate_dataset(
                test_episodes, seed * 1000 + 90 + shift_index,
                missing_strength=strength, dependence_strength=strength,
            )
            for method in METHODS:
                shift_rows.append({
                    "seed": seed, "strength": strength, "method": method,
                    **metrics(test.z, shift_models[method].predict_proba(test.y)),
                })

    condition_summary = _summary(condition_rows, ("condition", "method"))
    shift_summary = _summary(shift_rows, ("strength", "method"))
    _write_csv(output / "condition_per_seed.csv", condition_rows)
    _write_csv(output / "condition_summary.csv", condition_summary)
    _write_csv(output / "generator_diagnostics.csv", diagnostic_rows)
    _write_csv(output / "shift_per_seed.csv", shift_rows)
    _write_csv(output / "shift_summary.csv", shift_summary)
    config = {
        "status": "smoke" if smoke else "exploratory_development",
        "seeds": list(seeds), "train_episodes": train_episodes,
        "test_episodes": test_episodes, "alpha": 1.0,
        "conditions": CONDITIONS, "shift_train": [0.2, 0.2],
        "shift_strengths": SHIFT_STRENGTHS, "methods": METHODS,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _write_report(output / "report.md", condition_summary, shift_summary, diagnostic_rows, smoke)
    return {"condition_rows": len(condition_rows), "shift_rows": len(shift_rows)}


def _write_report(path: Path, condition_summary, shift_summary, diagnostics_rows, smoke: bool) -> None:
    lookup = {(r["condition"], r["method"]): r for r in condition_summary}
    d_rows = [r for r in diagnostics_rows if r["condition"] == "D_strategic_dependent"]
    lines = [
        "# JMID v1 smoke result" if smoke else "# JMID v1 exploratory development result",
        "", "This is wiring evidence only." if smoke else
        "This is three-seed exploratory evidence, not formal confirmation.", "",
        "## Condition means", "",
        "| Condition | Method | Accuracy | Brier | NLL | ECE | Wrong confidence | High-confidence error |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        for method in METHODS:
            row = lookup[(condition, method)]
            lines.append(
                f"| {condition} | {method} | {row['accuracy_mean']:.4f} | "
                f"{row['brier_mean']:.4f} | {row['nll_mean']:.4f} | "
                f"{row['ece_mean']:.4f} | {row['wrong_confidence_mean']:.4f} | "
                f"{row['high_confidence_error_rate_mean']:.4f} |"
            )
    lines += ["", "## Condition D generator check", ""]
    for key in (
        "missing_rate", "missing_rate_z0", "missing_rate_z1",
        "pair_correlation", "missing_pair_correlation",
    ):
        lines.append(f"- {key}: {np.mean([r[key] for r in d_rows]):.4f}")
    lines += ["", "## Shift endpoints", ""]
    for strength in (0.75, 1.0):
        lines.append(f"### Joint strength {strength}")
        for method in METHODS:
            row = next(r for r in shift_summary if r["strength"] == strength and r["method"] == method)
            lines.append(f"- {method}: Brier {row['brier_mean']:.4f}; NLL {row['nll_mean']:.4f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.output, smoke=args.smoke)


if __name__ == "__main__":
    main()

