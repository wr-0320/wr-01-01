"""Normalize existing Stage 1-6 outputs into paper-ready metric tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_EPISODE_FIELDS = [
    "stage",
    "experiment",
    "estimator",
    "model",
    "deceiver_policy",
    "deception_family",
    "condition",
    "deception_parameters",
    "deception_steps",
    "episode",
    "true_goal",
    "belief_accuracy",
    "deceiver_success",
    "commitment_accuracy",
    "deception_cost",
    "robustness_under_unseen_deception",
    "deception_exposure",
    "source_file",
]
_SUMMARY_FIELDS = [
    "stage",
    "experiment",
    "estimator",
    "model",
    "deceiver_policy",
    "deception_family",
    "condition",
    "deception_parameters",
    "deception_steps",
    "episodes",
    "belief_accuracy",
    "deception_success_rate",
    "commitment_accuracy",
    "deception_cost",
    "robustness_under_unseen_deception",
    "deception_exposure",
    "source_file",
]
_MANIFEST = [
    {
        "stage": "Stage 1",
        "experiment": "GridWorld MVP validation",
        "source_file": "tests/test_gridworld.py",
        "episodes": 0,
        "note": "Environment/API/information-boundary tests; no episode-level CSV.",
    },
    {
        "stage": "Stage 2",
        "experiment": "H1/H2 scripted baselines",
        "source_file": "outputs/baseline_results.csv",
        "episodes": 60,
        "note": "Rule estimator; honest, random, and scripted deception baselines.",
    },
    {
        "stage": "Stage 3",
        "experiment": "H3 commitment",
        "source_file": "outputs/h3_commitment_results.csv",
        "episodes": 40,
        "note": "Rule estimator with irreversible commitment.",
    },
    {
        "stage": "Stage 4",
        "experiment": "Rule versus GRU",
        "source_file": "outputs/estimator_comparison/",
        "episodes": 200,
        "note": "H1/H2 and H3 estimator comparison.",
    },
    {
        "stage": "Stage 5",
        "experiment": "Defender adaptation",
        "source_file": "outputs/adaptation/",
        "episodes": 160,
        "note": "Model A/B tested on seen steps=2 and unseen steps=3.",
    },
    {
        "stage": "Stage 6",
        "experiment": "Expanded data and generalization",
        "source_file": "outputs/data_scaling/",
        "episodes": 2000,
        "note": "1,000 episodes/configuration, ten seed groups, unseen steps=5.",
    },
    {
        "stage": "Stage 6B",
        "experiment": "Unseen deception families",
        "source_file": "outputs/deception_family_generalization/",
        "episodes": 2000,
        "note": "Frozen Model A/B; extrapolated steps=6/7 and three unseen families.",
    },
    {
        "stage": "Stage 6C",
        "experiment": "Exposure versus policy diversity",
        "source_file": "outputs/exposure_diversity/",
        "episodes": 10800,
        "note": "Three model seeds; C1/C2 vary scripted exposure, C2/C3 match exposure and vary family count.",
    },
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment output is missing: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _belief_accuracy(belief_json: str, true_goal: str) -> int:
    belief = json.loads(belief_json)
    predicted_goal = "A" if belief["A"] >= belief["B"] else "B"
    return int(predicted_goal == true_goal)


def _scripted_cost(deception_steps: int) -> int:
    # On the 7x7 MVP, each feint step moves one Manhattan unit away from the goal.
    return 2 * deception_steps


def _episode_row(
    *,
    stage: str,
    experiment: str,
    estimator: str = "",
    model: str = "",
    policy: str = "",
    deception_family: str = "",
    condition: str = "",
    deception_parameters: str = "",
    deception_steps: int | str = "",
    episode: int | str = "",
    true_goal: str = "",
    belief_accuracy: float | int | str = "",
    deceiver_success: float | int | str = "",
    commitment_accuracy: float | int | str = "",
    deception_cost: float | int | str = "",
    deception_exposure: int | str = "",
    source_file: str = "",
) -> dict[str, Any]:
    steps = int(deception_steps) if str(deception_steps).isdigit() else ""
    unseen_robustness: float | int | str = ""
    if condition.startswith("unseen_") and commitment_accuracy != "":
        unseen_robustness = float(commitment_accuracy)
    if deception_cost == "" and policy == "scripted_deception" and steps != "":
        deception_cost = _scripted_cost(steps)
    return {
        "stage": stage,
        "experiment": experiment,
        "estimator": estimator,
        "model": model,
        "deceiver_policy": policy,
        "deception_family": deception_family,
        "condition": condition,
        "deception_parameters": deception_parameters,
        "deception_steps": steps,
        "episode": episode,
        "true_goal": true_goal,
        "belief_accuracy": belief_accuracy,
        "deceiver_success": deceiver_success,
        "commitment_accuracy": commitment_accuracy,
        "deception_cost": deception_cost,
        "robustness_under_unseen_deception": unseen_robustness,
        "deception_exposure": deception_exposure,
        "source_file": source_file,
    }


def _from_h1_h2(
    path: Path,
    stage: str,
    estimator: str,
) -> list[dict[str, Any]]:
    normalized = []
    for row in _read_csv(path):
        policy = row["deceiver_policy"]
        steps = row.get("deception_steps") or (2 if policy == "scripted_deception" else "")
        normalized.append(
            _episode_row(
                stage=stage,
                experiment="H1_H2",
                estimator=estimator,
                policy=policy,
                condition="baseline_trajectory",
                deception_steps=steps,
                episode=row.get("episode", ""),
                true_goal=row.get("true_goal", ""),
                belief_accuracy=_belief_accuracy(
                    row["estimated_belief"], row["true_goal"]
                ),
                deceiver_success=int(row["deceiver_success"]),
                source_file=str(path),
            )
        )
    return normalized


def _from_h3(path: Path, stage: str, estimator: str) -> list[dict[str, Any]]:
    normalized = []
    for row in _read_csv(path):
        normalized.append(
            _episode_row(
                stage=stage,
                experiment="H3_commitment",
                estimator=estimator,
                policy=row["deceiver_policy"],
                condition="fixed_commitment",
                deception_steps=row.get("deception_steps", ""),
                episode=row.get("episode", ""),
                true_goal=row.get("true_goal", ""),
                belief_accuracy=_belief_accuracy(
                    row["belief_at_commitment"], row["true_goal"]
                ),
                deceiver_success=int(row["deceiver_success"]),
                commitment_accuracy=int(row["decision_correct"]),
                deception_cost=(
                    float(row["deception_cost"])
                    if row["deceiver_policy"] == "scripted_deception"
                    else ""
                ),
                source_file=str(path),
            )
        )
    return normalized


def _training_exposure(path: Path) -> dict[str, int]:
    rows = _read_csv(path)
    scripted_count = sum(row.get("deceiver_policy") == "scripted_deception" for row in rows)
    return {"ModelA": 0, "ModelB": scripted_count}


def _from_adaptation(path: Path, stage: str, exposures: dict[str, int]) -> list[dict[str, Any]]:
    normalized = []
    for row in _read_csv(path):
        steps = row.get("deception_steps", "")
        cost = _scripted_cost(int(steps)) if steps else ""
        normalized.append(
            _episode_row(
                stage=stage,
                experiment="defender_adaptation",
                model=row["model"],
                policy="scripted_deception",
                condition=row["condition"],
                deception_steps=steps,
                episode=row.get("episode", ""),
                true_goal=row.get("true_goal", ""),
                belief_accuracy=float(row["belief_accuracy"]),
                deceiver_success=int(row["deceiver_success"]),
                commitment_accuracy=int(row["commitment_accuracy"]),
                deception_cost=cost,
                deception_exposure=exposures[row["model"]],
                source_file=str(path),
            )
        )
    return normalized


def _from_family_generalization(
    path: Path,
    exposures: dict[str, int],
) -> list[dict[str, Any]]:
    normalized = []
    for row in _read_csv(path):
        family = row["deception_family"]
        normalized.append(
            _episode_row(
                stage="Stage 6B",
                experiment="unseen_deception_families",
                model=row["model"],
                policy=family,
                deception_family=family,
                condition=row["condition"],
                deception_parameters=row["deception_parameters"],
                deception_steps=row.get("deception_steps", ""),
                episode=row.get("episode", ""),
                true_goal=row.get("true_goal", ""),
                belief_accuracy=float(row["belief_accuracy"]),
                deceiver_success=int(row["deceiver_success"]),
                commitment_accuracy=int(row["commitment_accuracy"]),
                deception_exposure=exposures[row["model"]],
                source_file=str(path),
            )
        )
    return normalized


def _from_exposure_diversity(path: Path) -> list[dict[str, Any]]:
    normalized = []
    for row in _read_csv(path):
        normalized.append(
            _episode_row(
                stage="Stage 6C",
                experiment="exposure_diversity",
                model=row["model"],
                policy=row["test_family"],
                deception_family=row["test_family"],
                condition=f"unseen_{row['test_scenario']}",
                deception_parameters=row["test_parameters"],
                deception_steps=row.get("deception_steps", ""),
                episode=row["episode"],
                true_goal=row["true_goal"],
                belief_accuracy=float(row["belief_accuracy"]),
                deceiver_success=int(row["deceiver_success"]),
                commitment_accuracy=int(row["commitment_accuracy"]),
                deception_exposure=int(row["deception_training_episodes"]),
                source_file=str(path),
            )
        )
    return normalized


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_fields = (
        "stage",
        "experiment",
        "estimator",
        "model",
        "deceiver_policy",
        "deception_family",
        "condition",
        "deception_parameters",
        "deception_steps",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in key_fields)].append(row)

    summaries = []
    for key, group in groups.items():
        summary = dict(zip(key_fields, key))
        summary["episodes"] = len(group)
        for field, output_field in (
            ("belief_accuracy", "belief_accuracy"),
            ("deceiver_success", "deception_success_rate"),
            ("commitment_accuracy", "commitment_accuracy"),
            ("deception_cost", "deception_cost"),
            (
                "robustness_under_unseen_deception",
                "robustness_under_unseen_deception",
            ),
            ("deception_exposure", "deception_exposure"),
        ):
            values = [float(row[field]) for row in group if row[field] != ""]
            summary[output_field] = float(np.mean(values)) if values else ""
        summary["source_file"] = group[0]["source_file"]
        summaries.append(summary)
    return summaries


def evaluate_paper(
    workspace: str | Path = ".",
    output_dir: str | Path = "outputs/paper",
) -> dict[str, Path]:
    root = Path(workspace)
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = root / destination
    destination.mkdir(parents=True, exist_ok=True)

    episode_rows = []
    episode_rows.extend(
        _from_h1_h2(root / "outputs/baseline_results.csv", "Stage 2", "rule")
    )
    episode_rows.extend(
        _from_h3(root / "outputs/h3_commitment_results.csv", "Stage 3", "rule")
    )
    comparison_root = root / "outputs/estimator_comparison"
    for estimator in ("rule", "gru"):
        episode_rows.extend(
            _from_h1_h2(
                comparison_root / f"h1_h2_{estimator}.csv",
                "Stage 4",
                estimator,
            )
        )
        episode_rows.extend(
            _from_h3(
                comparison_root / f"h3_{estimator}.csv",
                "Stage 4",
                estimator,
            )
        )

    adaptation_root = root / "outputs/adaptation"
    episode_rows.extend(
        _from_adaptation(
            adaptation_root / "adaptation_episodes.csv",
            "Stage 5",
            _training_exposure(adaptation_root / "training_episodes.csv"),
        )
    )
    scaling_root = root / "outputs/data_scaling"
    episode_rows.extend(
        _from_adaptation(
            scaling_root / "data_scaling_episodes.csv",
            "Stage 6",
            _training_exposure(scaling_root / "training_episodes.csv"),
        )
    )
    family_root = root / "outputs/deception_family_generalization"
    episode_rows.extend(
        _from_family_generalization(
            family_root / "family_generalization_episodes.csv",
            _training_exposure(scaling_root / "training_episodes.csv"),
        )
    )
    exposure_root = root / "outputs/exposure_diversity"
    episode_rows.extend(
        _from_exposure_diversity(
            exposure_root / "exposure_diversity_episodes.csv"
        )
    )

    episode_path = destination / "evaluation_episodes.csv"
    summary_path = destination / "evaluation_summary.csv"
    manifest_path = destination / "experiment_manifest.csv"
    with episode_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(episode_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(_summarize(episode_rows))
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(_MANIFEST[0]))
        writer.writeheader()
        writer.writerows(_MANIFEST)
    return {
        "episodes": episode_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper"))
    args = parser.parse_args()
    outputs = evaluate_paper(args.workspace, args.output_dir)
    print(f"Wrote paper evaluation summary to {outputs['summary']}")


if __name__ == "__main__":
    main()
