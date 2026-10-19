"""Build Stage 15 mean/std tables, paper figures, claims, and theory appendix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.train_learned_defender import write_csv


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/stage15_robustness"
METRICS = (
    "interception_success",
    "false_commitment_episode_rate",
    "observation_cost",
    "decision_regret",
    "decision_latency",
    "detection_auroc_vs_honest",
    "truncation_rate",
    "commitment_coverage",
)
UNSEEN = (
    "scripted_8",
    "scripted_10",
    "delay_detour",
    "detour_switch",
    "stochastic_deception",
)


def read_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            row["evaluation_seed"] = int(raw["evaluation_seed"])
            row["episodes"] = int(raw["episodes"])
            for metric in METRICS + ("goal_A_success", "goal_B_success"):
                value = raw.get(metric, "")
                row[metric] = None if value in ("", "None") else float(value)
            rows.append(row)
    return rows


def mean_std(values) -> tuple[float | None, float | None]:
    clean = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(clean):
        return None, None
    return float(clean.mean()), float(clean.std(ddof=1)) if len(clean) > 1 else 0.0


def aggregate(rows, keys) -> list[dict[str, object]]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key, selected in sorted(groups.items()):
        result = dict(zip(keys, group_key))
        result["seeds"] = len({row["evaluation_seed"] for row in selected})
        result["episodes_per_seed"] = selected[0]["episodes"]
        for metric in METRICS:
            mean, std = mean_std(row[metric] for row in selected)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
            result[f"{metric}_mean_std"] = (
                "N/A" if mean is None else f"{mean:.4f} ± {std:.4f}"
            )
        output.append(result)
    return output


def macro_rows(rows, environment, controller, families, scope):
    selected = [
        row
        for row in rows
        if row["environment"] == environment
        and row["controller"] == controller
        and row["family"] in families
    ]
    by_seed = defaultdict(list)
    for row in selected:
        by_seed[row["evaluation_seed"]].append(row)
    output = []
    for seed, seed_rows in sorted(by_seed.items()):
        result = {
            "environment": environment,
            "controller": controller,
            "scope": scope,
            "evaluation_seed": seed,
        }
        for metric in METRICS:
            values = [row[metric] for row in seed_rows if row[metric] is not None]
            result[metric] = float(np.mean(values)) if values else None
        output.append(result)
    return output


def aggregate_macro(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["environment"], row["controller"], row["scope"])].append(row)
    output = []
    for (environment, controller, scope), selected in sorted(groups.items()):
        result = {
            "environment": environment,
            "controller": controller,
            "scope": scope,
            "seeds": len(selected),
        }
        for metric in METRICS:
            mean, std = mean_std(row[metric] for row in selected)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
            result[f"{metric}_mean_std"] = (
                "N/A" if mean is None else f"{mean:.4f} ± {std:.4f}"
            )
        output.append(result)
    return output


def latex_table(path: Path, rows, columns, headers, caption, label):
    def render(value):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("_", r"\_").replace("±", r"$\pm$")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(render(row[column]) for column in columns) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table*}",
    ]
    path.write_text("\n".join(lines) + "\n")


def bar_with_error(ax, labels, means, errors, ylabel, title):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errors, capsize=3, color="#4472C4")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def build_figures(output: Path, family, macro, seed_rows):
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    grid = [row for row in family if row["environment"] == "gridworld"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bar_with_error(
        ax,
        [row["family"] for row in grid],
        [row["interception_success_mean"] for row in grid],
        [row["interception_success_std"] for row in grid],
        "Interception success",
        "Frozen Stage13 robustness across five evaluation seeds",
    )
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(figures / "stage13_five_seed_robustness.png", dpi=200)
    fig.savefig(figures / "stage13_five_seed_robustness.pdf")
    plt.close(fig)

    v = {row["family"]: row for row in grid}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, metric, title in zip(
        axes,
        ("interception_success", "false_commitment_episode_rate", "decision_regret"),
        ("Interception", "False commitment", "Regret"),
    ):
        bar_with_error(
            ax,
            ["Adaptive v1", "Adaptive v2"],
            [v[name][f"{metric}_mean"] for name in ("adaptive_v1", "adaptive_v2")],
            [v[name][f"{metric}_std"] for name in ("adaptive_v1", "adaptive_v2")],
            title,
            title,
        )
        ax.set_xticklabels(["v1", "v2"], rotation=0)
    fig.suptitle("Predeclared stronger adaptive attacker comparison")
    fig.tight_layout()
    fig.savefig(figures / "adaptive_attacker_v2.png", dpi=200)
    fig.savefig(figures / "adaptive_attacker_v2.pdf")
    plt.close(fig)

    fork = [
        row
        for row in macro
        if row["environment"] == "forked_routes" and row["scope"] == "all_deceptive_macro"
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title in zip(
        axes,
        ("interception_success", "observation_cost", "decision_regret"),
        ("Interception", "Observation cost", "Decision regret"),
    ):
        bar_with_error(
            ax,
            [row["controller"] for row in fork],
            [row[f"{metric}_mean"] for row in fork],
            [row[f"{metric}_std"] for row in fork],
            title,
            title,
        )
    fig.suptitle("Forked Routes formal evaluation: deceptive-family macro")
    fig.tight_layout()
    fig.savefig(figures / "forked_routes_main.png", dpi=200)
    fig.savefig(figures / "forked_routes_main.pdf")
    plt.close(fig)

    controllers = sorted({row["controller"] for row in seed_rows if row["environment"] == "forked_routes"})
    families = list(UNSEEN)
    matrix = np.zeros((len(controllers), len(families)))
    for i, controller in enumerate(controllers):
        for j, family_name in enumerate(families):
            values = [
                row["interception_success"]
                for row in seed_rows
                if row["environment"] == "forked_routes"
                and row["controller"] == controller
                and row["family"] == family_name
            ]
            matrix[i, j] = np.mean(values)
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(families)), families, rotation=30, ha="right")
    ax.set_yticks(range(len(controllers)), controllers)
    for i in range(len(controllers)):
        for j in range(len(families)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] < .55 else "black")
    fig.colorbar(image, ax=ax, label="Interception success")
    ax.set_title("Forked Routes unseen-family robustness")
    fig.tight_layout()
    fig.savefig(figures / "forked_routes_unseen_heatmap.png", dpi=200)
    fig.savefig(figures / "forked_routes_unseen_heatmap.pdf")
    plt.close(fig)

    gaps = defaultdict(list)
    for row in seed_rows:
        gaps[(row["environment"], row["controller"])].append(
            abs(row["goal_A_success"] - row["goal_B_success"])
        )
    labels = [f"{key[0]}:{key[1]}" for key in sorted(gaps)]
    means = [float(np.mean(gaps[key])) for key in sorted(gaps)]
    errors = [float(np.std(gaps[key], ddof=1)) for key in sorted(gaps)]
    fig, ax = plt.subplots(figsize=(9, 4))
    bar_with_error(ax, labels, means, errors, "Absolute A/B success gap", "Goal symmetry appendix diagnostic")
    fig.tight_layout()
    fig.savefig(figures / "appendix_goal_symmetry.png", dpi=200)
    fig.savefig(figures / "appendix_goal_symmetry.pdf")
    plt.close(fig)


THEORY = r"""\makeatletter
\@ifundefined{theorem}{\newtheorem{theorem}{Theorem}}{}
\makeatother
\section{Formal Results for Active Anti-Deception}
\label{app:formal-results}

\paragraph{Setup.}
Let $Z=(G,D)$ denote the latent goal and deception mode, $H_t$ the
Defender-visible history by time $t$, and $a_t\in\{\mathsf{observe},
\mathsf{track},\mathsf{commit}_A,\mathsf{commit}_B\}$.  A commitment is
irreversible.  For a latent state $z$, write $P_z^t$ for the distribution of
$H_t$ induced by the environment, attacker, and sensing policy.

\begin{proposition}[Exact observational aliasing]
Suppose two latent states $z_A,z_B$ require opposite correct commitments and
$P_{z_A}^t=P_{z_B}^t$.  Under a balanced prior, every possibly randomized
history-only rule that commits by time $t$ has commitment accuracy exactly
$1/2$ on the pair.
\end{proposition}
\begin{proof}
Condition on any history $h$.  Equality of the history laws makes the
posterior over $z_A,z_B$ equal to the balanced prior.  If the rule commits to
$A$ with probability $q(h)$, its conditional accuracy is
$\tfrac12q(h)+\tfrac12(1-q(h))=\tfrac12$.  Integrating over $h$ gives the
claim.
\end{proof}

\begin{theorem}[Approximate identifiability lower bound]
For the same pair with balanced prior, the minimum error of any commitment
rule measurable with respect to $H_t$ is
\[
  R_t^*=\frac{1-\operatorname{TV}(P_{z_A}^t,P_{z_B}^t)}{2}.
\]
Consequently, if $\operatorname{TV}(P_{z_A}^t,P_{z_B}^t)\leq\epsilon$, every
rule that commits by $t$ has error at least $(1-\epsilon)/2$.
\end{theorem}
\begin{proof}
A commitment rule is a binary hypothesis test between $P_{z_A}^t$ and
$P_{z_B}^t$.  For a decision region $S$ assigned to $z_A$, balanced-prior
error is $\tfrac12P_{z_A}^t(S^c)+\tfrac12P_{z_B}^t(S)$.  Minimizing over $S$
is equivalent to maximizing $P_{z_A}^t(S)-P_{z_B}^t(S)$, whose supremum is
total variation.  Substitution yields the displayed equality.
\end{proof}

\begin{proposition}[Net value of an observation]
At any belief state, let $V_{\rm commit}$ be the best immediate commitment
value and let $V_{\rm info}$ be the expected continuation value after an
additional signal when the controller may ignore that signal.  Then
$V_{\rm info}\geq V_{\rm commit}$.  An observation of cost $c$ is preferred
by a Bayes-optimal controller exactly when
$V_{\rm info}-V_{\rm commit}>c$ (with equality permitting a tie).
\end{proposition}
\begin{proof}
The continuation policy can ignore the signal and reproduce the best
immediate commitment, proving the first inequality.  Subtracting the sensing
cost from the continuation value gives the decision condition.
\end{proof}

\begin{proposition}[Likelihood misspecification can reverse commitment]
For every confidence level $1-\delta$ with $0<\delta<1/2$, there is an
attacker distribution and a misspecified honest likelihood under which the
Defender posterior assigns probability at least $1-\delta$ to the wrong goal.
Thus posterior confidence alone does not guarantee interception under an
adaptive attacker.
\end{proposition}
\begin{proof}[Proof sketch]
Choose a history $h$ that the assumed model assigns likelihood $1$ under goal
$A$ and likelihood at most $\delta/(1-\delta)$ under goal $B$.  Let an
adaptive goal-$B$ attacker deliberately produce $h$.  Bayes' rule under the
misspecified model assigns at least $1-\delta$ posterior mass to $A$ although
the true goal is $B$.  Smoothing zero likelihoods gives the same construction
with strictly positive densities.
\end{proof}

\paragraph{Scope.}
The theorem formalizes the trajectory-identifiability limit measured in
Stage~9.2.  The value proposition motivates adaptive sensing but does not
claim that the learned Stage13 controller is Bayes-optimal.  The
misspecification proposition explains why high detection AUROC and confident
goal inference need not imply successful irreversible commitment.
"""


def build(source=SOURCE):
    source = Path(source)
    rows = read_rows(source / "per_seed_summary.csv")
    tables = source / "tables"
    tables.mkdir(exist_ok=True)
    family = aggregate(rows, ("environment", "controller", "family"))
    write_csv(tables / "family_mean_std.csv", family)
    macro_seed_rows = []
    macro_seed_rows += macro_rows(rows, "gridworld", "stage13_learned", ("adaptive_v1",), "adaptive_v1")
    macro_seed_rows += macro_rows(rows, "gridworld", "stage13_learned", ("adaptive_v2",), "adaptive_v2")
    macro_seed_rows += macro_rows(rows, "gridworld", "stage13_learned", UNSEEN, "unseen_macro")
    macro_seed_rows += macro_rows(
        rows,
        "gridworld",
        "stage13_learned",
        tuple(family for family in {row["family"] for row in rows if row["environment"] == "gridworld"} if family != "honest"),
        "all_deceptive_macro",
    )
    for controller in sorted({row["controller"] for row in rows if row["environment"] == "forked_routes"}):
        macro_seed_rows += macro_rows(
            rows,
            "forked_routes",
            controller,
            tuple(family for family in {row["family"] for row in rows if row["environment"] == "forked_routes"} if family != "honest"),
            "all_deceptive_macro",
        )
        macro_seed_rows += macro_rows(rows, "forked_routes", controller, UNSEEN, "unseen_macro")
    macro = aggregate_macro(macro_seed_rows)
    write_csv(tables / "main_mean_std.csv", macro)
    latex_table(
        tables / "main_mean_std.tex",
        [row for row in macro if row["scope"] == "all_deceptive_macro"],
        ("environment", "controller", "interception_success_mean_std", "false_commitment_episode_rate_mean_std", "observation_cost_mean_std", "decision_regret_mean_std"),
        ("Environment", "Controller", "Interception", "False commit", "Obs. cost", "Regret"),
        "Five-seed robustness results. Values are mean $\\pm$ sample standard deviation across evaluation seeds.",
        "tab:stage15-main",
    )
    latex_table(
        tables / "family_mean_std.tex",
        [row for row in family if row["environment"] == "gridworld"],
        ("family", "interception_success_mean_std", "false_commitment_episode_rate_mean_std", "decision_regret_mean_std", "detection_auroc_vs_honest_mean_std"),
        ("Family", "Interception", "False commit", "Regret", "AUROC"),
        "Frozen Stage13 family-level evaluation over five seeds.",
        "tab:stage15-families",
    )
    build_figures(source, family, macro, rows)
    (source / "theoretical_appendix.tex").write_text(THEORY + "\n")

    lookup = {(row["environment"], row["controller"], row["scope"]): row for row in macro}
    grid_v1 = lookup[("gridworld", "stage13_learned", "adaptive_v1")]
    grid_v2 = lookup[("gridworld", "stage13_learned", "adaptive_v2")]
    fork_stage13 = lookup[("forked_routes", "stage13_learned", "all_deceptive_macro")]
    fork_rule = lookup[("forked_routes", "rule_active", "all_deceptive_macro")]
    claims = [
        {
            "claim": "Stage13 was evaluated over five fixed evaluation seeds.",
            "status": "supported",
            "evidence": "per_seed_summary.csv contains seeds 0-4 for every GridWorld family.",
        },
        {
            "claim": "Adaptive attacker v2 is empirically stronger than v1 for frozen Stage13.",
            "status": "supported" if grid_v2["interception_success_mean"] < grid_v1["interception_success_mean"] else "not_supported",
            "evidence": f"interception v1={grid_v1['interception_success_mean_std']}, v2={grid_v2['interception_success_mean_std']}; no parameter selection was performed.",
        },
        {
            "claim": "Stage13 exceeds Rule Active on Forked Routes deceptive-family macro.",
            "status": "supported" if fork_stage13["interception_success_mean"] > fork_rule["interception_success_mean"] else "not_supported",
            "evidence": f"Stage13={fork_stage13['interception_success_mean_std']}, Rule={fork_rule['interception_success_mean_std']}.",
        },
        {
            "claim": "The Forked Routes experiment establishes broad environment generalization.",
            "status": "rejected_overclaim",
            "evidence": "Only one additional 7x7 topology is tested; this supports a cross-environment stress test, not broad generalization.",
        },
    ]
    write_csv(source / "claim_validation.csv", claims)
    report = [
        "# Stage 15 Robustness and Paper Validation",
        "",
        "## Frozen protocol",
        "",
        "All results load the existing Stage13 controller, dual-head belief model, and calibrator without training. GridWorld and both frozen controller files are hash-checked before and after evaluation. Adaptive attacker v2 is a predeclared rule stress test with no optimizer or result-based parameter selection. Forked Routes uses one common charged TRACK bootstrap whenever the visible history is empty, because the frozen neural inference contract requires at least one observed point.",
        "",
        "Mean and sample standard deviation are computed from five seed-level estimates. Episodes inside a seed are not treated as independent seeds. A/B goals are exactly balanced in every controller-family-seed cell.",
        "",
        "## Key results",
        "",
        f"On GridWorld, frozen Stage13 interception against adaptive v1 is {grid_v1['interception_success_mean_std']}; against the predeclared v2 it is {grid_v2['interception_success_mean_std']}.",
        f"On Forked Routes across all deceptive families, Stage13 interception is {fork_stage13['interception_success_mean_std']} and Rule Active is {fork_rule['interception_success_mean_std']}.",
        "",
        "The complete family results are in `tables/family_mean_std.csv`; the paper main table is `tables/main_mean_std.csv`. Claim outcomes are reported even when they do not support the proposed method. No tuning or additional training follows from these results.",
        "",
        "## Statistical limits",
        "",
        "The five seeds vary simulator randomness and stochastic-family actions. Deterministic fixed families can repeat trajectories, so error bars describe this evaluation schedule rather than independent attacker populations. Adaptive attackers maintain separate online profiles per controller and seed, which prevents paired-trajectory interpretation for those conditions.",
    ]
    (source / "stage15_robustness_report.md").write_text("\n".join(report) + "\n")
    manifest = {
        "tables": sorted(str(path.relative_to(source)) for path in tables.iterdir()),
        "figures": sorted(str(path.relative_to(source)) for path in (source / "figures").iterdir()),
        "theory": "theoretical_appendix.tex",
        "claims": "claim_validation.csv",
    }
    (source / "paper_artifacts.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.source), indent=2))
