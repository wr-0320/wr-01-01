"""Run the frozen JMID v2 falsification study."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.jmid_v1.core import metrics
from .generator import ROTATED_GROUPS, TRAIN_GROUPS, generate_v2, v2_diagnostics
from .models import V2_METHODS, fit_v2_models


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs" / "development" / "jmid_v2"
SEEDS = (3401, 3402, 3403)
CONDITIONS = {
    "A_random_independent": (0.0, 0.0),
    "B_censor_independent": (1.0, 0.0),
    "C_random_relay": (0.0, 0.85),
    "D_censor_relay": (1.0, 0.85),
}
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _summarize(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    names = ("accuracy", "brier", "nll", "ece", "wrong_confidence", "high_confidence_error_rate")
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, selected in grouped.items():
        record = dict(zip(keys, values)); record["seeds"] = len(selected)
        for name in names:
            array = np.asarray([row[name] for row in selected], dtype=float)
            record[f"{name}_mean"] = float(array.mean())
            record[f"{name}_min"] = float(array.min())
            record[f"{name}_max"] = float(array.max())
        output.append(record)
    return output


def _evaluate(seed, label, models, test, destination):
    for method in V2_METHODS:
        destination.append({
            "seed": seed, "test": label, "method": method,
            **metrics(test.z, models[method].predict_proba(test.y)),
        })


def run(output: Path = DEFAULT_OUTPUT, *, seeds=SEEDS, train_episodes=4000, test_episodes=5000, smoke=False):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if smoke:
        train_episodes=min(train_episodes,300); test_episodes=min(test_episodes,400)
    primary, diagnostics_rows, transfer = [], [], []
    for seed in seeds:
        for index, (condition, (censor, copy)) in enumerate(CONDITIONS.items()):
            train=generate_v2(train_episodes,seed*100+index,censor_strength=censor,copy_strength=copy)
            test=generate_v2(test_episodes,seed*1000+index,censor_strength=censor,copy_strength=copy)
            models=fit_v2_models(train)
            diagnostics_rows.append({"seed":seed,"condition":condition,**v2_diagnostics(test)})
            for method in V2_METHODS:
                primary.append({"seed":seed,"condition":condition,"method":method,
                    **metrics(test.z,models[method].predict_proba(test.y))})

        intensity_train=generate_v2(train_episodes,seed*100+80,censor_strength=.5,copy_strength=.5)
        intensity_models=fit_v2_models(intensity_train)
        for index,strength in enumerate(STRENGTHS):
            test=generate_v2(test_episodes,seed*1000+80+index,censor_strength=strength,copy_strength=strength)
            _evaluate(seed,f"strength_{strength}",intensity_models,test,transfer)

        transfer_train=generate_v2(train_episodes,seed*100+90,censor_strength=1.0,copy_strength=.85)
        transfer_models=fit_v2_models(transfer_train)
        for index, mode in enumerate(("negative","conflict")):
            test=generate_v2(test_episodes,seed*1000+90+index,censor_strength=1.0,copy_strength=.85,censor_mode=mode)
            _evaluate(seed,f"missing_{mode}",transfer_models,test,transfer)
        graph_test=generate_v2(test_episodes,seed*1000+99,censor_strength=1.0,copy_strength=.85,groups=ROTATED_GROUPS)
        _evaluate(seed,"rotated_dependency",transfer_models,graph_test,transfer)

    primary_summary=_summarize(primary,("condition","method"))
    transfer_summary=_summarize(transfer,("test","method"))
    for name,rows in (("primary_per_seed.csv",primary),("primary_summary.csv",primary_summary),
                      ("generator_diagnostics.csv",diagnostics_rows),("transfer_per_seed.csv",transfer),
                      ("transfer_summary.csv",transfer_summary)):
        _write_csv(output/name,rows)
    config={"status":"smoke" if smoke else "falsification_development","seeds":list(seeds),
        "train_episodes":train_episodes,"test_episodes":test_episodes,"conditions":CONDITIONS,
        "strengths":STRENGTHS,"methods":V2_METHODS,"train_groups":TRAIN_GROUPS,
        "rotated_groups":ROTATED_GROUPS,"composition":"logit(M2)+logit(M3)-logit(M1)"}
    (output/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    _write_report(output/"report.md",primary_summary,transfer_summary,diagnostics_rows,smoke)
    return {"primary_rows":len(primary),"transfer_rows":len(transfer)}


def _write_report(path, primary, transfer, diagnostics_rows, smoke):
    lines=["# JMID v2 smoke result" if smoke else "# JMID v2 falsification result","",
        "Wiring evidence only." if smoke else "Three-seed exploratory falsification; not formal confirmation.","",
        "## Primary means","","| Condition | Method | Accuracy | Brier | NLL | ECE | Wrong conf. | High-conf. error |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        for method in V2_METHODS:
            row=next(r for r in primary if r["condition"]==condition and r["method"]==method)
            lines.append(f"| {condition} | {method} | {row['accuracy_mean']:.4f} | {row['brier_mean']:.4f} | {row['nll_mean']:.4f} | {row['ece_mean']:.4f} | {row['wrong_confidence_mean']:.4f} | {row['high_confidence_error_rate_mean']:.4f} |")
    lines += ["","## Condition D diagnostics",""]
    selected=[r for r in diagnostics_rows if r["condition"]=="D_censor_relay"]
    for key in ("missing_rate","missing_rate_z0","missing_rate_z1","missing_given_raw1","missing_given_raw0","copy_rate","within_state_report_correlation","within_state_missing_correlation"):
        lines.append(f"- {key}: {np.mean([r[key] for r in selected]):.4f}")
    lines += ["","## Transfer means",""]
    for label in ("strength_0.75","strength_1.0","missing_negative","missing_conflict","rotated_dependency"):
        lines.append(f"### {label}")
        for method in V2_METHODS:
            row=next(r for r in transfer if r["test"]==label and r["method"]==method)
            lines.append(f"- {method}: Brier {row['brier_mean']:.4f}; NLL {row['nll_mean']:.4f}; accuracy {row['accuracy_mean']:.4f}")
        lines.append("")
    path.write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args(); run(args.output,smoke=args.smoke)


if __name__=="__main__": main()

