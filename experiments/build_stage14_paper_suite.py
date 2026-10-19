"""Build Stage 14 paper tables, figures, failure catalog and theory draft."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUTPUT = ROOT / "outputs" / "stage14_paper_suite"
SOURCES = {
    "stage73": WORKSPACE / "stage73-reward-debug" / "evaluation_summary.csv",
    "stage73_goals": WORKSPACE / "stage73-reward-debug" / "goal_symmetry.csv",
    "stage92_results": WORKSPACE / "stage92-identifiability" / "results.json",
    "stage92_identifiability": WORKSPACE / "stage92-identifiability" / "identifiability.json",
    "stage10": WORKSPACE / "stage10-two-level" / "comparison.csv",
    "stage101": WORKSPACE / "stage10-1-decoupled" / "comparison.csv",
    "stage102": WORKSPACE / "stage10-2-adaptive" / "comparison.csv",
    "stage103": WORKSPACE / "stage10-3-robustness" / "comparison.csv",
    "stage103_families": WORKSPACE / "stage10-3-robustness" / "family_comparison.csv",
    "stage103_goals": WORKSPACE / "stage10-3-robustness" / "goal_comparison.csv",
    "stage12": ROOT / "outputs" / "stage12_bayesian_pomdp" / "summary.csv",
    "stage13": ROOT / "outputs" / "stage13_learned_controller" / "summary.csv",
    "stage14": ROOT / "outputs" / "stage14_formal_baselines" / "summary.csv",
    "interfaces": ROOT / "outputs" / "stage14_formal_baselines" / "environment_interfaces.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict], columns: list[tuple[str, str]], percent=()) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if value in (None, ""):
                rendered = "—"
            elif key in percent:
                rendered = f"{float(value):.1%}"
            elif isinstance(value, float):
                rendered = f"{value:.4f}"
            else:
                rendered = str(value).replace("|", "/")
            values.append(rendered)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main_table() -> list[dict]:
    stage12 = read_csv(SOURCES["stage12"])
    stage14 = read_csv(SOURCES["stage14"])
    retained = {"rule_active", "stage10_1", "adaptive"}
    selected = [row for row in stage12 if row["controller"] in retained] + stage14
    labels = {
        "rule_active": "Rule Active",
        "stage10_1": "Stage10.1",
        "adaptive": "Adaptive Defender",
        "formal_bayesian": "Bayesian (formal)",
        "formal_pomdp": "POMDP (formal)",
        "stage13_learned": "Stage13 Learned",
    }
    attacks = {"honest": "Honest", "scripted": "Scripted-3", "adaptive_attacker": "Adaptive attacker"}
    rows = []
    for row in selected:
        rows.append(
            {
                "method": labels[row["controller"]],
                "attack": attacks[row["attack"]],
                "episodes": int(row["episodes"]),
                "interception_success": float(row["interception_success"]),
                "false_commitment_rate": float(row["false_commitment_rate"]),
                "detection_auroc": (
                    "" if not row.get("detection_auroc_vs_honest") else float(row["detection_auroc_vs_honest"])
                ),
                "observation_cost": float(row["mean_observation_cost"]),
                "decision_regret": float(row["decision_regret"]),
                "decision_latency": float(row["decision_latency"]),
                "protocol": "Stage12 common schedule, seed=0, A/B=17/17",
            }
        )
    method_order = list(labels.values())
    attack_order = list(attacks.values())
    return sorted(rows, key=lambda row: (method_order.index(row["method"]), attack_order.index(row["attack"])))


def ablation_table() -> list[dict]:
    rows: list[dict] = []
    identifiability = json.loads(SOURCES["stage92_results"].read_text())
    for mode, description in {
        "A": "trajectory only",
        "B": "trajectory + timing",
        "C": "trajectory + belief history",
    }.items():
        result = identifiability[mode]
        rows.append(
            {
                "ablation": "Input identifiability",
                "variant": f"{mode}: {description}",
                "split": "balanced test",
                "episodes": 96,
                "detection_f1": result["f1"],
                "auroc": result["auroc"],
                "interception_success": "",
                "false_commitment": "",
                "observation_cost": "",
                "regret": "",
                "source": "Stage9.2",
            }
        )
    for row in read_csv(SOURCES["stage101"]):
        if row["group"] == "overall" and row["controller"] in {"rule_active", "stage10", "stage10_1"}:
            rows.append(
                {
                    "ablation": "Decision architecture",
                    "variant": row["controller"],
                    "split": "seen + unseen",
                    "episodes": int(row["episodes"]),
                    "detection_f1": float(row["deception_f1"]),
                    "auroc": float(row["deception_auroc"]),
                    "interception_success": float(row["defender_task_success"]),
                    "false_commitment": float(row["false_commitment_episode_rate"]),
                    "observation_cost": float(row["mean_observation_cost"]),
                    "regret": "",
                    "source": "Stage10.1",
                }
            )
    for row in read_csv(SOURCES["stage102"]):
        if row["group"] == "unseen_only":
            rows.append(
                {
                    "ablation": "Adaptive observation",
                    "variant": row["controller"],
                    "split": "delayed/detour/multi-switch",
                    "episodes": int(row["episodes"]),
                    "detection_f1": float(row["deception_f1"]),
                    "auroc": "",
                    "interception_success": float(row["interception_success"]),
                    "false_commitment": float(row["false_commitment_episode_rate"]),
                    "observation_cost": float(row["mean_observation_cost"]),
                    "regret": float(row["decision_regret"]),
                    "source": "Stage10.2",
                }
            )
    for row in read_csv(SOURCES["stage13"]):
        if row["attack"] == "adaptive_attacker":
            rows.append(
                {
                    "ablation": "Learned controller",
                    "variant": row["controller"],
                    "split": "adaptive attacker",
                    "episodes": int(row["episodes"]),
                    "detection_f1": float(row["deception_f1"]),
                    "auroc": "",
                    "interception_success": float(row["interception_success"]),
                    "false_commitment": float(row["false_commitment_episode_rate"]),
                    "observation_cost": float(row["mean_observation_cost"]),
                    "regret": float(row["decision_regret"]),
                    "source": "Stage13",
                }
            )
    for row in read_csv(SOURCES["stage73"]):
        rows.append(
            {
                "ablation": "PPO reward",
                "variant": row["experiment"],
                "split": "A/B balanced",
                "episodes": int(row["episodes"]),
                "detection_f1": "",
                "auroc": "",
                "interception_success": float(row["task_success_rate"]),
                "false_commitment": "",
                "observation_cost": "",
                "regret": "",
                "source": "Stage7.3",
            }
        )
    return rows


def failure_table() -> list[dict]:
    stage10 = next(
        row for row in read_csv(SOURCES["stage10"]) if row["controller"] == "two_level" and row["group"] == "overall"
    )
    stage103 = read_csv(SOURCES["stage103_families"])
    stage103_goals = read_csv(SOURCES["stage103_goals"])
    stage12 = read_csv(SOURCES["stage12"])
    stage14 = read_csv(SOURCES["stage14"])
    identifiability = json.loads(SOURCES["stage92_identifiability"].read_text())[0]
    long_rows = [row for row in stage103 if row["family"] in {"scripted_8", "scripted_10"}]
    goal_a = next(row for row in stage103_goals if row["controller"] == "stage10_1" and row["goal"] == "A")
    goal_b = next(row for row in stage103_goals if row["controller"] == "stage10_1" and row["goal"] == "B")
    adaptive_collapse = next(
        row for row in stage12 if row["controller"] == "adaptive" and row["attack"] == "adaptive_attacker"
    )
    formal_bayes = next(
        row for row in stage14 if row["controller"] == "formal_bayesian" and row["attack"] == "scripted"
    )
    return [
        {
            "stage": "7.3",
            "failure_case": "PPO A/B collapse",
            "condition": "R1/R2/R3",
            "evidence": "A success 0%; B success 100% in every reward ablation",
            "mechanism": "Policy/action-boundary collapse survives reward changes",
            "source": "goal_symmetry.csv",
        },
        {
            "stage": "9.2",
            "failure_case": "Observable aliasing",
            "condition": "trajectory-only balanced dataset",
            "evidence": f"{identifiability['minimum_errors']} irreducible errors; empirical ceiling {identifiability['empirical_accuracy_ceiling']:.2%}",
            "mechanism": "Honest and scripted-5 produce identical observable histories",
            "source": "identifiability.json",
        },
        {
            "stage": "10",
            "failure_case": "Detection-decision gap",
            "condition": "seen + unseen",
            "evidence": f"F1 {float(stage10['deception_f1']):.2%}, task success {float(stage10['defender_task_success']):.2%}",
            "mechanism": "Correct detection does not guarantee correct irreversible commitment",
            "source": "Stage10 comparison.csv",
        },
        {
            "stage": "10.3",
            "failure_case": "Long-feint failure",
            "condition": "scripted steps 8 and 10",
            "evidence": f"{len(long_rows)}/{len(long_rows)} method-family rows have 0% interception",
            "mechanism": "All controllers commit before the delayed reversal",
            "source": "family_comparison.csv",
        },
        {
            "stage": "10.3",
            "failure_case": "Goal-conditioned asymmetry",
            "condition": "Stage10.1 robustness suite",
            "evidence": f"A {float(goal_a['interception_success']):.2%} vs B {float(goal_b['interception_success']):.2%}",
            "mechanism": "Controller/inference interaction is asymmetric although the map is symmetric",
            "source": "goal_comparison.csv",
        },
        {
            "stage": "12",
            "failure_case": "Adaptive-controller collapse",
            "condition": "adaptive attacker",
            "evidence": f"interception {float(adaptive_collapse['interception_success']):.2%}; false commit {float(adaptive_collapse['false_commitment_rate']):.2%}",
            "mechanism": "Observation policy enters a deterministic wrong-commit loop",
            "source": "Stage12 summary.csv",
        },
        {
            "stage": "14",
            "failure_case": "Classical model misspecification",
            "condition": "formal Bayesian vs scripted-3",
            "evidence": f"AUROC {float(formal_bayes['detection_auroc_vs_honest']):.2f}, interception {float(formal_bayes['interception_success']):.2%}",
            "mechanism": "Honest-motion likelihood assigns high posterior confidence to the feint goal",
            "source": "Stage14 formal summary.csv",
        },
    ]


def save_figures(destination: Path, main: list[dict], ablations: list[dict]) -> list[Path]:
    figure_dir = destination / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    created = []
    methods = list(dict.fromkeys(row["method"] for row in main))
    attacks = ["Honest", "Scripted-3", "Adaptive attacker"]
    x = np.arange(len(methods))
    width = 0.24
    fig, axis = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    for offset, attack in enumerate(attacks):
        values = [next(row for row in main if row["method"] == method and row["attack"] == attack)["interception_success"] for method in methods]
        axis.bar(x + (offset - 1) * width, values, width, label=attack)
    axis.set_xticks(x, methods, rotation=20, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Interception success")
    axis.set_title("Common-protocol defender comparison")
    axis.legend(frameon=False, ncols=3)
    axis.grid(axis="y", alpha=0.25)
    created += save_pair(fig, figure_dir / "main_interception")

    fig, axis = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    markers = {"Honest": "o", "Scripted-3": "s", "Adaptive attacker": "^"}
    for row in main:
        axis.scatter(
            row["observation_cost"], row["interception_success"],
            marker=markers[row["attack"]], s=55,
            label=f"{row['method']} / {row['attack']}",
        )
    axis.set(xlabel="Mean observation cost", ylabel="Interception success", ylim=(-0.04, 1.04))
    axis.set_title("Sensing-cost / interception trade-off")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=6.7, ncols=2)
    created += save_pair(fig, figure_dir / "cost_success_tradeoff")

    family_rows = read_csv(SOURCES["stage103_families"])
    family_names = list(dict.fromkeys(row["family"] for row in family_rows))
    controller_names = ["rule_active", "stage10_1", "adaptive"]
    matrix = np.asarray([
        [float(next(row for row in family_rows if row["controller"] == controller and row["family"] == family)["interception_success"]) for family in family_names]
        for controller in controller_names
    ])
    fig, axis = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.0%}", ha="center", va="center", color="white" if matrix[row, column] < .55 else "black")
    axis.set_xticks(range(len(family_names)), family_names, rotation=20, ha="right")
    axis.set_yticks(range(len(controller_names)), controller_names)
    axis.set_title("Stage10.3 unseen-family interception")
    fig.colorbar(image, ax=axis, label="Interception success")
    created += save_pair(fig, figure_dir / "robustness_heatmap")

    decision_rows = [row for row in ablations if row["ablation"] == "Decision architecture"]
    labels = [row["variant"] for row in decision_rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].bar(labels, [row["detection_f1"] for row in decision_rows], color="#4C78A8")
    axes[1].bar(labels, [row["interception_success"] for row in decision_rows], color="#F58518")
    axes[0].set_title("Detection F1")
    axes[1].set_title("Task success")
    for axis in axes:
        axis.set_ylim(0, 1.05)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=.25)
    fig.suptitle("Stage10 decision-architecture ablation")
    created += save_pair(fig, figure_dir / "decision_ablation")

    interfaces = read_csv(SOURCES["interfaces"])
    fig, axis = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    labels = [row["environment"] for row in interfaces]
    path_a = [int(row["shortest_path_A"]) for row in interfaces]
    path_b = [int(row["shortest_path_B"]) for row in interfaces]
    positions = np.arange(len(labels))
    axis.bar(positions - .18, path_a, .36, label="Goal A")
    axis.bar(positions + .18, path_b, .36, label="Goal B")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Shortest path length")
    axis.set_title("Environment-interface symmetry check")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=.25)
    created += save_pair(fig, figure_dir / "environment_symmetry")
    return created


def save_pair(figure, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    figure.savefig(paths[0], dpi=300)
    figure.savefig(paths[1])
    plt.close(figure)
    return paths


def theoretical_appendix() -> str:
    return r"""# Theoretical Appendix Draft

## A. Problem formulation

Let (G\in\{A,B\}) be the attacker's latent goal, (Z\in\{0,1\}) its honest/deceptive type, (X_t) its physical state, (O_t) the defender-visible observation, and (C_t\in\{\bot,A,B\}) the irreversible commitment state. The history is (H_t=(O_{0:t},U_{0:t-1})), where (U_t\in\{\text{observe},\text{track},\text{commit}_A,\text{commit}_B\}). The defender seeks a policy that maximizes interception utility minus sensing cost and false-commitment loss.

The experiments use the empirical objective

\[
J(\pi)=\mathbb{E}_{\pi}\left[R_{\mathrm{int}}-C_{\mathrm{obs}}-\lambda_{\mathrm{false}}\mathbf{1}\{C\ne G\}\right].
\]

Regret is reported against a privileged immediate-correct-guard oracle. It is an evaluation quantity and is not Bayes regret under equal information.

## B. Bayesian intent filtering

The formal baseline assumes a Boltzmann-rational honest transition model

\[
P(X_{t+1}=x'\mid X_t=x,G=g)\propto
\sum_{a:T(x,a)=x'}\exp[-\beta d_g(x')].
\]

For an observation gap of (k) steps, the likelihood is the (k)-step transition probability, obtained by exact propagation over the finite state space. The posterior update is

\[
b_t(g)=\frac{P(O_t\mid H_{t-1},g)b_{t-1}(g)}
{\sum_{g'}P(O_t\mid H_{t-1},g')b_{t-1}(g')}.
\]

Implementation uses log probabilities and explicit normalization. The stopping baseline observes until posterior confidence crosses a declared boundary or reaches its deadline. Its failure under feints is a model-misspecification result: a confident posterior need not be correct when the attacker violates the honest-motion likelihood.

## C. Finite-horizon belief planning

For sensing action (u), signal (y), Bayesian update (\tau(b,u,y)), and horizon (h), the implemented recursion is

\[
V_h(b)=\max\left\{Q_A(b),Q_B(b),
\max_{u\in\{o,t\}}\left[-c_u+\sum_y P(y\mid b,u)V_{h-1}(\tau(b,u,y))\right]\right\}.
\]

The commit values are (Q_g(b)=b(g)R_{\mathrm{int}}-(1-b(g))\lambda_{\mathrm{false}}). Commitment is absorbing. The binary observation kernels and costs are declared in the experiment configuration. Exact recursion therefore solves the stated reduced POMDP, although that reduced model is not the full adversarial GridWorld and does not model strategic type changes.

## D. Counterfactual evidence and decoupled control

The proposed pipeline separates goal belief from deception evidence. For observed history (h), the counterfactual module reports

\[
L(h)=\frac{P(h\mid Z=1)}{P(h\mid Z=0)}.
\]

The controller consumes calibrated goal belief, entropy, likelihood ratio and accumulated sensing cost. This separation explains why detection metrics and decision metrics must both be reported: AUROC is invariant to a score threshold and can remain high after an incorrect irreversible commitment.

## E. Identifiability limit

**Proposition 1 (observable aliasing).** Suppose two policy/type pairs induce the same distribution over defender-visible histories on an event (E): (P(H\mid Z=0,E)=P(H\mid Z=1,E)). Conditional on (E), every trajectory-only detector has Bayes error at least (\min\{P(Z=0\mid E),P(Z=1\mid E)\}).

**Proof.** Equality of the conditional likelihoods makes the likelihood ratio one for every history in (E). Bayes' rule therefore leaves the posterior class odds equal to the prior odds. The minimum-error decision is the majority class and incurs the minority prior mass. No measurable function of the same history can reduce this error. \(\square\)

Stage9.2 exhibits this condition empirically: honest and scripted-5 samples share observable inputs. Action timing and deterministic belief history add no information because both are functions of the same aliased trajectory in that dataset.

## F. Commitment-risk decomposition

For commitment to (g), expected one-step utility is

\[
U_g=b(g)R_{\mathrm{int}}-(1-b(g))\lambda_{\mathrm{false}}.
\]

The value of another observation is its expected posterior decision value minus its cost. Immediate commitment is justified only when its value exceeds that information value. A deception detector can change the observation model or increase the modeled risk, but detection alone cannot identify the correct goal. This distinction predicts the Stage10 detection-decision gap.

## G. Cross-environment interface

Stage14 exposes a finite transition model (\mathcal{M}=(S,A,T,s_0,g_A,g_B,H)) and a ParallelEnv factory. GridWorld and Forked Routes satisfy the same timed visible-trajectory contract. Both have equal A/B shortest-path lengths; Forked Routes adds obstacle-constrained topology. The interface supports environment-aware Bayesian likelihoods without changing either environment or any Stage1-13 controller.

## H. Scope of claims

The experiments support comparisons only within each declared protocol. Stage12/14 main-table rows share seeds, goals and attack labels, but an online adaptive attacker generates controller-dependent trajectories. Results from Stage7.3, Stage9.2, Stage10.1/10.2, Stage10.3 and Stage13 remain ablations with their original sample sizes. Deterministic route duplication limits effective sample diversity. The current evidence does not establish minimax optimality, equilibrium behavior, or transfer of learned neural encoders to Forked Routes.
"""


def build(destination=OUTPUT):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source_data = destination / "source_data"
    source_data.mkdir(parents=True, exist_ok=True)
    resolved = {}
    for name, path in SOURCES.items():
        snapshot = source_data / f"{name}{path.suffix}"
        if path.is_file():
            shutil.copy2(path, snapshot)
            resolved[name] = snapshot
        elif snapshot.is_file():
            resolved[name] = snapshot
        else:
            resolved[name] = path
    SOURCES.update(resolved)
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Stage14 source artifacts: " + ", ".join(missing))
    main = main_table()
    ablations = ablation_table()
    failures = failure_table()
    write_csv(destination / "main_table.csv", main)
    write_csv(destination / "ablation_table.csv", ablations)
    write_csv(destination / "failure_case_table.csv", failures)
    main_columns = [
        ("method", "Method"), ("attack", "Attack"), ("episodes", "n"),
        ("interception_success", "Interception"), ("false_commitment_rate", "False commit"),
        ("detection_auroc", "AUROC"), ("observation_cost", "Obs. cost"),
        ("decision_regret", "Regret"), ("decision_latency", "Latency"),
    ]
    ablation_columns = [
        ("ablation", "Ablation"), ("variant", "Variant"), ("split", "Split"),
        ("episodes", "n"), ("detection_f1", "F1"), ("auroc", "AUROC"),
        ("interception_success", "Task/interception"), ("false_commitment", "False commit"),
        ("observation_cost", "Obs. cost"), ("regret", "Regret"),
    ]
    failure_columns = [
        ("stage", "Stage"), ("failure_case", "Failure"), ("condition", "Condition"),
        ("evidence", "Evidence"), ("mechanism", "Diagnosis"),
    ]
    (destination / "main_table.md").write_text(
        "# Main Table\n\nCommon Stage12 scenario schedule; 34 episodes per attack/method, A/B balanced.\n\n"
        + markdown_table(main, main_columns, {"interception_success", "false_commitment_rate"}) + "\n"
    )
    (destination / "ablation_table.md").write_text(
        "# Ablation Table\n\nRows retain their original stage protocol and sample size.\n\n"
        + markdown_table(ablations, ablation_columns, {"detection_f1", "auroc", "interception_success", "false_commitment"}) + "\n"
    )
    (destination / "failure_case_table.md").write_text(
        "# Failure Case Table\n\n" + markdown_table(failures, failure_columns) + "\n"
    )
    (destination / "theoretical_appendix.md").write_text(theoretical_appendix())
    figures = save_figures(destination, main, ablations)
    manifest = {
        "generated_by": "experiments.build_stage14_paper_suite",
        "algorithms_modified": False,
        "main_protocol": "Stage12 common scenario schedule; seed=0; 34 episodes per attack/controller; A/B=17/17",
        "cross_stage_policy": "Only main-table rows use a shared protocol; ablations retain source protocol and n",
        "sources": {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in SOURCES.items()
        },
        "tables": ["main_table.csv", "ablation_table.csv", "failure_case_table.csv"],
        "figures": [str(path.relative_to(destination)) for path in figures],
        "environment_interfaces": read_csv(SOURCES["interfaces"]),
    }
    (destination / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    adaptive = next(row for row in main if row["method"] == "Stage13 Learned" and row["attack"] == "Adaptive attacker")
    formal = next(row for row in main if row["method"] == "Bayesian (formal)" and row["attack"] == "Scripted-3")
    report = f"""# Stage 14 Paper Experiment Suite

## Reproducible scope

The main table uses one common Stage12 schedule: seed 0, 34 Honest, 34 Scripted-3 and 34 Adaptive-attacker scenarios per method, with 17 A and 17 B goals in every cell. Existing Rule Active, Stage10.1 and Adaptive Defender rows are reused byte-for-byte. Stage14 evaluates only the new formal Bayesian/POMDP implementations and the frozen Stage13 checkpoint on that schedule. No Deceiver, belief encoder, controller or GridWorld algorithm was retrained or modified.

The ablation table preserves each source experiment's own protocol and sample size. It must not be interpreted as a pooled ranking across stages.

## Main findings

Stage13 Learned reaches {adaptive['interception_success']:.1%} interception against the adaptive attacker at mean observation cost {adaptive['observation_cost']:.4f}. The formal Bayesian baseline has AUROC {formal['detection_auroc']:.4f} against Scripted-3 but {formal['interception_success']:.1%} interception, directly reproducing the detection-decision gap under an honest-motion likelihood. Rule Active and Stage10.1 remain strong on the Stage12 attack set, while the Stage10.3 table shows that none of the tested controllers solves longer feints.

The second environment interface is implemented by Forked Routes. Contract checks pass for both environments, and each has symmetric A/B shortest-path length. Learned cross-environment transfer is not claimed because the Stage10 neural encoder remains frozen to its original data domain.

## Outputs

- `main_table.csv/.md`: shared-protocol comparison.
- `ablation_table.csv/.md`: input, controller, observation and reward ablations with source protocol.
- `failure_case_table.csv/.md`: empirical failure modes and source evidence.
- `theoretical_appendix.md`: formal problem, Bayes/POMDP recursions, identifiability proposition and scope.
- `figures/`: PNG and PDF paper figures generated from the tables.
- `artifact_manifest.json`: source hashes and environment contract metadata.
"""
    (destination / "stage14_report.md").write_text(report)
    return {"destination": str(destination), "tables": 3, "figures": len(figures), "main_rows": len(main)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.output_dir), indent=2))

