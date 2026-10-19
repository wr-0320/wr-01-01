"""Generate the Stage 15 LaTeX paper skeleton and evidence audit.

The builder consumes only frozen Stage 14 artifacts.  It does not import or
execute any environment, model, trainer, or controller.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGE14 = ROOT / "outputs" / "stage14_paper_suite"
OUTPUT = ROOT / "outputs" / "stage15_paper"
REQUIRED_SECTIONS = (
    "abstract",
    "introduction",
    "related_work",
    "method",
    "theory",
    "experiments",
    "limitations",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def percentage(value: str | float) -> str:
    if value in ("", None):
        return "--"
    return f"{100 * float(value):.1f}"


def decimal(value: str | float, digits: int = 3) -> str:
    if value in ("", None):
        return "--"
    return f"{float(value):.{digits}f}"


def main_results_table(rows: list[dict[str, str]]) -> str:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    lookup = {(row["method"], row["attack"]): row for row in rows}
    lines = [
        "% Auto-generated from Stage14 main_table.csv; do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Shared-protocol results. Each cell contains 34 episodes with 17 A and 17 B goals. Int.: interception success; FC: false-commitment rate; Cost: mean observation cost.}",
        r"\label{tab:main-results}",
        r"\small",
        r"\begin{tabular}{lrrr rrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Scripted-3} & \multicolumn{3}{c}{Adaptive attacker} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & Int. $\uparrow$ & FC $\downarrow$ & Cost $\downarrow$ & Int. $\uparrow$ & FC $\downarrow$ & Cost $\downarrow$ \\",
        r"\midrule",
    ]
    for method in methods:
        scripted = lookup[(method, "Scripted-3")]
        adaptive = lookup[(method, "Adaptive attacker")]
        lines.append(
            f"{tex_escape(method)} & {percentage(scripted['interception_success'])} & "
            f"{percentage(scripted['false_commitment_rate'])} & {decimal(scripted['observation_cost'])} & "
            f"{percentage(adaptive['interception_success'])} & "
            f"{percentage(adaptive['false_commitment_rate'])} & {decimal(adaptive['observation_cost'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def ablation_results_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "% Auto-generated from Stage14 ablation_table.csv; protocols are not pooled.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Ablations under their original protocols. Empty cells indicate metrics that were not defined for that experiment.}",
        r"\label{tab:ablations}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Ablation & Variant & Split & $n$ & F1 & AUROC & Task/Int. & FC & Cost \\",
        r"\midrule",
    ]
    previous = None
    for row in rows:
        if previous is not None and previous != row["ablation"]:
            lines.append(r"\addlinespace")
        previous = row["ablation"]
        lines.append(
            f"{tex_escape(row['ablation'])} & {tex_escape(row['variant'])} & "
            f"{tex_escape(row['split'])} & {row['episodes']} & {percentage(row['detection_f1'])} & "
            f"{percentage(row['auroc'])} & {percentage(row['interception_success'])} & "
            f"{percentage(row['false_commitment'])} & {decimal(row['observation_cost'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def failure_results_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "% Auto-generated from Stage14 failure_case_table.csv.",
        r"\begin{table*}[p]",
        r"\centering",
        r"\caption{Empirically observed failure modes.}",
        r"\label{tab:failures}",
        r"\scriptsize",
        r"\begin{tabular}{p{0.06\textwidth}p{0.16\textwidth}p{0.20\textwidth}p{0.48\textwidth}}",
        r"\toprule",
        r"Stage & Failure & Condition & Evidence and diagnosis \\",
        r"\midrule",
    ]
    for row in rows:
        evidence = f"{row['evidence']}. {row['mechanism']}."
        lines.append(
            f"{tex_escape(row['stage'])} & {tex_escape(row['failure_case'])} & "
            f"{tex_escape(row['condition'])} & {tex_escape(evidence)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def _main_lookup(rows):
    return {(row["method"], row["attack"]): row for row in rows}


def build_claim_registry(stage14: Path = STAGE14) -> list[dict]:
    main = read_csv(stage14 / "main_table.csv")
    lookup = _main_lookup(main)
    families = read_csv(stage14 / "source_data" / "stage103_families.csv")
    interfaces = read_csv(stage14 / "source_data" / "interfaces.csv")
    identity = json.loads((stage14 / "source_data" / "stage92_identifiability.json").read_text())
    identity_results = json.loads((stage14 / "source_data" / "stage92_results.json").read_text())
    goal_rows = read_csv(stage14 / "source_data" / "stage103_goals.csv")
    stage101_goal_a = next(row for row in goal_rows if row["controller"] == "stage10_1" and row["goal"] == "A")
    stage101_goal_b = next(row for row in goal_rows if row["controller"] == "stage10_1" and row["goal"] == "B")
    two_level_path = ROOT / "src" / "deception_rl" / "beliefs" / "two_level.py"
    decoupled_belief_path = ROOT / "src" / "deception_rl" / "beliefs" / "decoupled.py"
    controller_path = ROOT / "src" / "deception_rl" / "agents" / "decoupled_defender.py"
    pomdp_path = ROOT / "experiments" / "stage14_classical_baselines.py"
    adaptive_attacker_path = ROOT / "experiments" / "stage11_adaptive_deceiver.py"
    stage14_config_path = ROOT / "outputs" / "stage14_formal_baselines" / "config.json"
    two_level_source = two_level_path.read_text() if two_level_path.is_file() else ""
    decoupled_source = decoupled_belief_path.read_text() if decoupled_belief_path.is_file() else ""
    controller_source = controller_path.read_text() if controller_path.is_file() else ""
    pomdp_source = pomdp_path.read_text() if pomdp_path.is_file() else ""
    attacker_source = adaptive_attacker_path.read_text() if adaptive_attacker_path.is_file() else ""
    stage14_config = json.loads(stage14_config_path.read_text()) if stage14_config_path.is_file() else {}
    architecture_verified = all(
        (
            "DeceptionAwareBelief" in two_level_source,
            "LikelihoodRatioReasoner" in decoupled_source,
            "CalibratedGoalInference" in decoupled_source,
            "DecoupledAntiDeception" in controller_source,
        )
    )
    frozen_verified = (
        stage14_config.get("training_episodes") == 0
        and stage14_config.get("stage13_checkpoint_frozen") is True
        and stage14_config.get("deceiver_training") is False
    )
    pomdp_verified = all(
        token in pomdp_source
        for token in ("FormalFiniteHorizonPOMDPDefender", "action_values", "_value", "@lru_cache")
    )
    s13 = lookup[("Stage13 Learned", "Adaptive attacker")]
    s101 = lookup[("Stage10.1", "Adaptive attacker")]
    bayes_scripted = lookup[("Bayesian (formal)", "Scripted-3")]
    pomdp_scripted = lookup[("POMDP (formal)", "Scripted-3")]
    long_feints = [row for row in families if row["family"] in {"scripted_8", "scripted_10"}]
    symmetric = all(row["shortest_path_A"] == row["shortest_path_B"] for row in interfaces)
    equal_modes = len(
        {
            (result["accuracy"], result["f1"], result["auroc"])
            for result in identity_results.values()
        }
    ) == 1
    entries = [
        ("M01", "implementation", "Method", "The framework separates goal belief, deception belief, counterfactual evidence, and commitment control.", "verified", "src/deception_rl/beliefs/{two_level,decoupled}.py; src/deception_rl/agents/decoupled_defender.py", "required classes found", architecture_verified, True, "Architectural claim only."),
        ("M02", "implementation", "Experiments", "Stage14 evaluation freezes earlier learned components and performs no training.", "verified", "outputs/stage14_formal_baselines/config.json", "training_episodes=0, frozen checkpoint, deceiver_training=false", frozen_verified, True, "Applies to Stage14 evaluation, not historical stages."),
        ("M03", "implementation", "Limitations", "The adaptive attacker is a hand-written online rule system rather than a learned equilibrium strategy.", "verified", "experiments/stage11_adaptive_deceiver.py", "adaptive rule branches present; no optimizer", "class AdaptiveDeceiver" in attacker_source and "optimizer" not in attacker_source.lower(), True, "It still adapts to completed Defender behavior."),
        ("M04", "protocol", "Limitations", "The shared main-table protocol uses one seed and 34 episodes per attack-method cell.", "verified", "main_table.csv", "all rows episodes=34 and protocol seed=0", all(int(row["episodes"]) == 34 and "seed=0" in row["protocol"] for row in main), True, "Descriptive evaluation only."),
        ("M05", "scope", "Limitations", "Forked Routes validates the interface but not learned cross-environment transfer.", "verified", "source_data/interfaces.csv; artifact_manifest.json", "interface rows only; manifest disclaims learned transfer", len(interfaces) == 2, True, "Negative scope statement; transfer remains untested."),
        ("T01", "theory", "Theory", "Observable class-conditional aliasing imposes the stated Bayes-error lower bound.", "proved", "sections/theory.tex", "Proposition and proof", True, True, "Conditional on equal observable distributions."),
        ("T02", "theory", "Theory", "The finite-horizon recursion exactly solves the declared reduced binary-belief POMDP.", "verified", "experiments/stage14_classical_baselines.py", "memoized exhaustive belief-tree recursion", pomdp_verified, True, "Does not solve the full adversarial GridWorld POMDP."),
        ("E01", "empirical", "Abstract", f"Stage13 Learned attains {float(s13['interception_success']):.0%} interception against the adaptive attacker in the shared protocol.", "supported", "main_table.csv", "method=Stage13 Learned; attack=Adaptive attacker", float(s13["interception_success"]) == 1.0, True, "34 episodes, seed 0, 17 A/17 B."),
        ("E02", "empirical", "Experiments", f"Stage13 matches Stage10.1 interception while reducing mean observation cost from {float(s101['observation_cost']):.4f} to {float(s13['observation_cost']):.4f} against the adaptive attacker.", "supported", "main_table.csv", "Stage13 Learned versus Stage10.1; adaptive attacker", float(s13["interception_success"]) == float(s101["interception_success"]) and float(s13["observation_cost"]) < float(s101["observation_cost"]), True, "Closed-loop trajectories are controller-dependent."),
        ("E03", "empirical", "Abstract", f"Formal Bayesian reaches AUROC {float(bayes_scripted['detection_auroc']):.1f} but {float(bayes_scripted['interception_success']):.0%} interception on Scripted-3.", "supported", "main_table.csv", "method=Bayesian (formal); attack=Scripted-3", float(bayes_scripted["detection_auroc"]) == 1.0 and float(bayes_scripted["interception_success"]) == 0.0, True, "Demonstrates a detection-decision gap under this protocol."),
        ("E04", "empirical", "Experiments", "Every evaluated method-family row for scripted steps 8 and 10 has zero interception.", "supported", "source_data/stage103_families.csv", "family in {scripted_8, scripted_10}", len(long_feints) == 6 and all(float(row["interception_success"]) == 0 for row in long_feints), True, "Three controllers, 34 episodes per cell."),
        ("E05", "empirical", "Abstract", f"Stage9.2 contains {identity[0]['minimum_errors']} irreducible errors and an empirical accuracy ceiling of {identity[0]['empirical_accuracy_ceiling']:.2%} under trajectory aliasing.", "supported", "source_data/stage92_identifiability.json", "mode=A", identity[0]["minimum_errors"] == 32 and identity[0]["empirical_accuracy_ceiling"] == 0.9375, True, "Empirical ceiling for the constructed finite dataset."),
        ("E06", "empirical", "Experiments", "Adding action timing or deterministic belief history does not improve Stage9.2 test accuracy, F1, or AUROC.", "supported", "source_data/stage92_results.json", "modes A/B/C", equal_modes, True, "Specific to the constructed dataset and deterministic auxiliary features."),
        ("E07", "empirical", "Method", "Both implemented environment interfaces have equal A/B shortest-path lengths.", "supported", "source_data/interfaces.csv", "all environment rows", symmetric and len(interfaces) == 2, True, "This is a topology contract check, not learned transfer."),
        ("E08", "empirical", "Experiments", f"Formal POMDP has {float(pomdp_scripted['interception_success']):.0%} interception on Scripted-3.", "supported", "main_table.csv", "method=POMDP (formal); attack=Scripted-3", float(pomdp_scripted["interception_success"]) == 0.0, True, "The declared observation model is misspecified under strategic feints."),
        ("E09", "empirical", "Experiments", "Stage13 Pareto-dominates Stage10.1 on interception, false commitment, observation cost, regret, and latency against the adaptive attacker in the shared protocol.", "qualified", "main_table.csv", "Stage13 Learned versus Stage10.1; adaptive attacker", all([float(s13["interception_success"]) >= float(s101["interception_success"]), float(s13["false_commitment_rate"]) <= float(s101["false_commitment_rate"]), float(s13["observation_cost"]) <= float(s101["observation_cost"]), float(s13["decision_regret"]) <= float(s101["decision_regret"]), float(s13["decision_latency"]) <= float(s101["decision_latency"])]), True, "Descriptive single-seed Pareto relation; no significance claim."),
        ("E10", "empirical", "Limitations", f"Stage10.3 Stage10.1 interception differs by goal: {float(stage101_goal_a['interception_success']):.2%} for A and {float(stage101_goal_b['interception_success']):.2%} for B.", "supported", "source_data/stage103_goals.csv", "controller=stage10_1; goals A/B", float(stage101_goal_a["interception_success"]) != float(stage101_goal_b["interception_success"]), True, "Observed robustness-suite asymmetry; mechanism is unresolved."),
        ("L01", "literature", "Related Work", "Belief-state planning is a standard formulation for finite-horizon POMDP decision making.", "cited", "references.bib", "Kaelbling1998", True, True, "Literature positioning, not an experimental result."),
        ("L02", "literature", "Related Work", "PPO is a clipped policy-gradient method.", "cited", "references.bib", "Schulman2017", True, True, "Literature positioning only."),
        ("L03", "literature", "Related Work", "PettingZoo provides standard APIs for multi-agent reinforcement-learning environments.", "cited", "references.bib", "Terry2021", True, True, "Literature positioning only."),
        ("X01", "candidate", "Excluded", "The proposed defender is robust to arbitrary deception.", "rejected", "source_data/stage103_families.csv", "contradicted by scripted_8/10", False, False, "Must not appear: long-feint interception is 0%."),
        ("X02", "candidate", "Excluded", "The learned controller transfers to Forked Routes.", "rejected", "source_data/interfaces.csv", "interface-only validation", False, False, "Must not appear: no learned transfer evaluation was run."),
        ("X03", "candidate", "Excluded", "Stage13 is statistically superior to all baselines.", "rejected", "main_table.csv", "single seed and no inferential test", False, False, "Must not appear: results are descriptive."),
    ]
    fields = (
        "claim_id", "claim_type", "section", "claim", "status", "evidence_artifact",
        "evidence_selector", "verified", "in_manuscript", "qualification",
    )
    return [dict(zip(fields, entry)) for entry in entries]


def section_texts(main_rows: list[dict[str, str]]) -> dict[str, str]:
    lookup = _main_lookup(main_rows)
    s13 = lookup[("Stage13 Learned", "Adaptive attacker")]
    s101 = lookup[("Stage10.1", "Adaptive attacker")]
    bayes = lookup[("Bayesian (formal)", "Scripted-3")]
    pomdp = lookup[("POMDP (formal)", "Scripted-3")]
    return {
        "abstract": rf"""\begin{{abstract}}
We study anti-deception decisions in a partially observable two-agent navigation game, where a defender must infer a hidden goal, detect strategic misdirection, and make an irreversible interception decision. We organize the existing system into a two-level pipeline with goal and deception beliefs, counterfactual evidence, and an adaptive commitment controller. \claim{{E01}}{{On the shared Stage12/14 protocol, the frozen Stage13 decision controller achieves {100 * float(s13['interception_success']):.0f}\% interception against the adaptive attacker.}} \claim{{E03}}{{A formal Bayesian baseline reaches AUROC {float(bayes['detection_auroc']):.1f} on Scripted-3 while intercepting {100 * float(bayes['interception_success']):.0f}\%, exposing a detection--decision gap.}} \claim{{E05}}{{A balanced identifiability analysis finds 32 irreducible errors and a 93.75\% empirical accuracy ceiling when honest and deceptive policies induce identical observations.}} These results support separating detection quality from decision quality while identifying long-feint and model-misspecification failures. All comparisons are descriptive, single-seed evaluations under their declared protocols.
\end{{abstract}}
""",
        "introduction": r"""\section{Introduction}
Agents that act under partial observability may intentionally produce trajectories that support a false interpretation of their goal. A defender must therefore answer two different questions: which goal is likely, and whether the observed behavior is strategically deceptive. Collapsing both questions into one classifier can produce a high detection score without a useful interception decision.

This paper skeleton consolidates a sequence of controlled experiments into a reproducible anti-deception evaluation. The contribution is threefold: a deception-aware belief model, a counterfactual comparison against expected honest behavior, and a commitment controller that trades information against irreversible error. Every empirical statement is linked to a machine-checked claim registry. Claims unsupported by the frozen artifacts are excluded from the manuscript.

Our evidence supports a narrow conclusion: decoupled inference and decision control can improve outcomes on the evaluated attack families, while observable aliasing, long feints, and likelihood misspecification remain unresolved. We do not claim equilibrium behavior, universal robustness, statistical superiority, or learned transfer to the second environment.
""",
        "related_work": r"""\section{Related Work}
\claim{L01}{Belief-state planning is a standard formulation for finite-horizon POMDP decision making}~\cite{Kaelbling1998}. \claim{L02}{Proximal Policy Optimization is a clipped policy-gradient method}~\cite{Schulman2017}; it appears in earlier diagnostic stages but is not part of the proposed anti-deception controller. \claim{L03}{PettingZoo provides standard APIs for multi-agent reinforcement-learning environments}~\cite{Terry2021}, which motivates the shared ParallelEnv contract used here.

\paragraph{Positioning to complete.} The final submission should add a focused review of intent inference, adversarial plan recognition, active sensing, strategic deception, calibrated uncertainty, and risk-sensitive stopping. Each added literature statement should receive a citation and a claim-registry entry when it affects the paper's novelty or scope.
""",
        "method": r"""\section{Method}
\claim{M01}{The framework separates goal belief, deception belief, counterfactual evidence, and commitment control.} Given defender-visible history $H_t$, a shared GRU encoder produces
\begin{equation}
  b_t = P(G\mid H_t), \qquad q_t = P(Z=1\mid H_t).
\end{equation}
The counterfactual module compares observed behavior with honest prototypes through the likelihood ratio
\begin{equation}
  L(H_t)=\frac{P(H_t\mid Z=1)}{P(H_t\mid Z=0)}.
\end{equation}
The controller receives $(b_t,\mathcal{H}(b_t),q_t,L(H_t),c_t,C_t)$ and selects observe, track, commit-A, or commit-B. Commitment is irreversible.

\paragraph{Environment interface.} Stage14 exposes a finite transition model and a ParallelEnv factory. \claim{E07}{Both GridWorld and Forked Routes have equal shortest-path lengths to goals A and B.} Forked Routes validates the interface and topology abstraction; no neural transfer experiment is reported.
""",
        "theory": r"""\section{Theory}
Let $G\in\{A,B\}$ denote the latent goal, $Z\in\{0,1\}$ the behavior type, and $H_t$ the visible history.

\begin{proposition}[Observable aliasing]\label{prop:aliasing}
\claim{T01}{If $P(H\mid Z=0,E)=P(H\mid Z=1,E)$ on an event $E$, every history-only detector has conditional Bayes error at least $\min\{P(Z=0\mid E),P(Z=1\mid E)\}$.}
\end{proposition}
\begin{proof}
The class-conditional likelihood ratio equals one on $E$, so Bayes' rule leaves posterior odds equal to prior odds. The minimum-error decision chooses the prior majority class and incurs the minority mass. No measurable function of the same history can improve this decision.
\end{proof}

For the classical baseline, commitment to $g$ has value
\begin{equation}
Q_g(b)=b(g)R_{\mathrm{int}}-(1-b(g))\lambda_{\mathrm{false}}.
\end{equation}
Sensing action $u$ follows the finite-horizon recursion
\begin{align}
V_h(b)&=\max\{Q_A(b),Q_B(b),S_h(b)\},\\
S_h(b)&=\max_u\left[-c_u+\sum_yP(y\mid b,u)
V_{h-1}(\tau(b,u,y))\right].
\end{align}
\claim{T02}{The implementation exactly evaluates this declared reduced binary-belief tree.} It does not solve the full adversarial GridWorld POMDP.
""",
        "experiments": rf"""\section{{Experiments}}
\paragraph{{Protocol.}} \claim{{M02}}{{The Stage14 comparison freezes all earlier learned components and performs no training.}} The main table uses seed 0 and 34 episodes per method--attack cell, balanced as 17 A and 17 B goals. Adaptive attackers maintain controller-specific histories, so paired seeds do not imply identical closed-loop trajectories. Cross-stage ablations retain their original protocols and are not pooled.

\input{{tables/main_results}}

\paragraph{{Shared-protocol result.}} \claim{{E02}}{{Against the adaptive attacker, Stage13 matches Stage10.1 at 100\% interception and 0\% false commitment while reducing mean observation cost from {float(s101['observation_cost']):.4f} to {float(s13['observation_cost']):.4f}.}} \claim{{E09}}{{Under the same descriptive protocol, Stage13 weakly improves every reported decision metric over Stage10.1 and strictly improves cost, regret, and latency.}}

\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{figures/main/main_performance.pdf}}
  \caption{{Interception and false-commitment rates under the shared Stage12/14 protocol.}}
  \label{{fig:main-performance}}
\end{{figure*}}

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.98\columnwidth]{{figures/main/decision_tradeoff.pdf}}
  \caption{{Observation-cost and interception trade-off against the adaptive attacker.}}
  \label{{fig:decision-tradeoff}}
\end{{figure}}

\paragraph{{Detection is not decision success.}} \claim{{E03}}{{Formal Bayesian obtains AUROC {float(bayes['detection_auroc']):.1f} but {100 * float(bayes['interception_success']):.0f}\% interception on Scripted-3.}} \claim{{E08}}{{The formal finite-horizon POMDP likewise obtains {100 * float(pomdp['interception_success']):.0f}\% interception on Scripted-3.}} Both models become confidently wrong under a feint that violates their honest-motion assumptions.

\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.98\columnwidth]{{figures/main/detection_decision_gap.pdf}}
  \caption{{Detection AUROC versus interception. High AUROC can coexist with a failed irreversible decision.}}
  \label{{fig:detection-decision}}
\end{{figure}}

\paragraph{{Ablations and failure boundary.}} \claim{{E06}}{{Trajectory, trajectory-plus-timing, and trajectory-plus-belief modes have identical Stage9.2 accuracy, F1, and AUROC.}} \claim{{E04}}{{All six controller/family cells for scripted steps 8 and 10 have zero interception.}} These findings delimit the current method rather than support broad robustness.

\input{{tables/ablation_results}}
""",
        "limitations": r"""\section{Limitations}
\paragraph{Concern: the reported gains may be protocol-specific.}
\textbf{Response.} We agree. \claim{M04}{The common table contains one seed and 34 episodes per attack--method cell on one map.} We therefore report descriptive rates and costs and avoid significance or universal-superiority language. A multi-seed, multi-map evaluation with confidence intervals is required before stronger claims.

\paragraph{Concern: the adaptive attacker is not a strategic equilibrium.}
\textbf{Response.} \claim{M03}{The attacker uses hand-written online adaptation and controller-specific cross-episode profiles rather than learning an equilibrium strategy.} It demonstrates response-conditioned behavior, not optimal play or equilibrium convergence. We retain this distinction in the protocol and do not infer minimax robustness.

\paragraph{Concern: a second environment interface is weaker than transfer.}
\textbf{Response.} Correct. \claim{M05}{Forked Routes passes the shared interface checks, but learned cross-environment transfer was not evaluated.} The manuscript treats this as infrastructure readiness and explicitly rejects a cross-environment generalization claim.

\paragraph{Concern: detection results may overstate practical defense.}
\textbf{Response.} The Bayesian result directly shows the problem: AUROC can be perfect while interception is zero. We report detection, commitment, interception, cost, latency, and regret separately. Regret uses a privileged immediate-correct-guard oracle and should not be interpreted as Bayes regret under equal information.

\paragraph{Concern: the classical baselines are misspecified.}
\textbf{Response.} They exactly solve their declared reduced models, but those models assume approximately honest motion and fixed binary observation kernels. Their failure under strategic feints is informative about model mismatch, not evidence that Bayesian or POMDP planning is generally ineffective.

\paragraph{Concern: observable aliasing may make the task impossible.}
\textbf{Response.} Yes for aliased histories under the stated information set. \claim{E05}{Stage9.2 measures a 93.75\% finite-dataset ceiling,} and Proposition~\ref{prop:aliasing} gives the corresponding conditional lower bound. Resolving such cases requires new information, interventions, or assumptions; a larger classifier alone cannot do so.

\paragraph{Residual failure cases.} Long scripted feints, goal-conditioned asymmetry in the robustness suite, deterministic trajectory duplication, reward-driven PPO collapse in earlier stages, and controller-dependent adaptive trajectories remain open. Appendix Table~\ref{tab:failures} lists the exact evidence.
\claim{E04}{All evaluated controllers fail on scripted steps 8 and 10.} \claim{E10}{For Stage10.1 in the Stage10.3 robustness suite, interception is 53.57\% for goal A and 5.95\% for goal B.}
""",
    }


def references_bib() -> str:
    return r"""@article{Kaelbling1998,
  author = {Kaelbling, Leslie Pack and Littman, Michael L. and Cassandra, Anthony R.},
  title = {Planning and Acting in Partially Observable Stochastic Domains},
  journal = {Artificial Intelligence},
  volume = {101},
  number = {1--2},
  pages = {99--134},
  year = {1998}
}

@article{Schulman2017,
  author = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  title = {Proximal Policy Optimization Algorithms},
  journal = {arXiv preprint arXiv:1707.06347},
  year = {2017}
}

@inproceedings{Terry2021,
  author = {Terry, Justin K. and Black, Benjamin and Grammel, Nathaniel and Jayakumar, Mario and Hari, Ananth and Sullivan, Ryan and Santos, Luis and Dieffendahl, Clemens and Horsch, Caroline and Perez-Vicente, Rodrigo and Williams, Niall L. and Lokesh, Yashas and Ravi, Praveen},
  title = {PettingZoo: Gym for Multi-Agent Reinforcement Learning},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2021}
}
"""


def paper_tex() -> str:
    return r"""\documentclass[10pt,twocolumn]{article}
\usepackage[margin=0.72in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs,longtable,array,graphicx}
\usepackage{xcolor,hyperref,microtype}
\newtheorem{proposition}{Proposition}
\newcommand{\claim}[2]{#2}
\title{Decoupled Active Anti-Deception under Partial Observability}
\author{Anonymous Authors}
\date{}

\begin{document}
\maketitle
\input{sections/abstract}
\input{sections/introduction}
\input{sections/related_work}
\input{sections/method}
\input{sections/theory}
\input{sections/experiments}
\input{sections/limitations}

\bibliographystyle{plain}
\bibliography{references}

\clearpage
\appendix
\section{Additional Results}
\input{tables/failure_cases}

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.82\textwidth]{figures/appendix/robustness_family_matrix.pdf}
  \caption{Interception across Stage10.3 unseen deception families.}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.48\textwidth]{figures/appendix/goal_asymmetry.pdf}\hfill
  \includegraphics[width=0.48\textwidth]{figures/appendix/reward_goal_symmetry.pdf}
  \caption{Goal-conditioned robustness (left) and Stage7.3 PPO reward-ablation asymmetry (right).}
\end{figure*}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.98\columnwidth]{figures/appendix/identifiability_modes.pdf}
  \caption{Stage9.2 input-mode results and empirical identifiability ceiling.}
\end{figure}

\section{Claim Audit}
The machine-readable registry is distributed as \texttt{claim\_registry.csv}; \texttt{claim\_audit.md} lists evidence selectors, qualifications, and rejected candidate claims.
\end{document}
"""


def _save_pair(figure, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    figure.savefig(paths[0], dpi=300)
    figure.savefig(paths[1])
    plt.close(figure)
    return paths


def generate_figures(stage14: Path, destination: Path) -> list[Path]:
    main_dir = destination / "figures" / "main"
    appendix_dir = destination / "figures" / "appendix"
    main_dir.mkdir(parents=True, exist_ok=True)
    appendix_dir.mkdir(parents=True, exist_ok=True)
    main = read_csv(stage14 / "main_table.csv")
    methods = list(dict.fromkeys(row["method"] for row in main))
    short = {
        "Rule Active": "Rule",
        "Stage10.1": "S10.1",
        "Adaptive Defender": "Adaptive",
        "Bayesian (formal)": "Bayes",
        "POMDP (formal)": "POMDP",
        "Stage13 Learned": "S13",
    }
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.8, len(methods)))
    lookup = _main_lookup(main)
    created = []

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    x = np.arange(len(methods))
    width = 0.36
    for offset, attack in enumerate(("Scripted-3", "Adaptive attacker")):
        intercept = [float(lookup[(method, attack)]["interception_success"]) for method in methods]
        false = [float(lookup[(method, attack)]["false_commitment_rate"]) for method in methods]
        axes[0].bar(x + (offset - .5) * width, intercept, width, label=attack, alpha=.9)
        axes[1].bar(x + (offset - .5) * width, false, width, label=attack, alpha=.9)
    for axis, title, ylabel in zip(
        axes,
        ("Interception success", "False commitment"),
        ("Rate (higher is better)", "Rate (lower is better)"),
    ):
        axis.set_xticks(x, [short[m] for m in methods], rotation=18, ha="right")
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False, ncols=2, fontsize=8)
    created += _save_pair(figure, main_dir / "main_performance")

    figure, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    adaptive = [row for row in main if row["attack"] == "Adaptive attacker"]
    for color, row in zip(colors, adaptive):
        axis.scatter(float(row["observation_cost"]), float(row["interception_success"]), s=85, color=color)
        axis.annotate(short[row["method"]], (float(row["observation_cost"]), float(row["interception_success"])), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set(xlabel="Mean observation cost", ylabel="Interception success", ylim=(-.04, 1.04))
    axis.set_title("Adaptive-attacker decision trade-off")
    axis.grid(alpha=.25)
    created += _save_pair(figure, main_dir / "decision_tradeoff")

    figure, axis = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    points = [row for row in main if row["attack"] != "Honest" and row["detection_auroc"]]
    marker = {"Scripted-3": "s", "Adaptive attacker": "^"}
    for row in points:
        axis.scatter(float(row["detection_auroc"]), float(row["interception_success"]), marker=marker[row["attack"]], s=70)
        axis.annotate(f"{short[row['method']]} / {row['attack'].replace(' attacker','')}", (float(row["detection_auroc"]), float(row["interception_success"])), xytext=(4, 4), textcoords="offset points", fontsize=6.5)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="equal numerical value")
    axis.set(xlabel="Detection AUROC vs honest", ylabel="Interception success", xlim=(.45, 1.02), ylim=(-.04, 1.04))
    axis.set_title("Detection and decision quality are distinct")
    axis.grid(alpha=.25)
    created += _save_pair(figure, main_dir / "detection_decision_gap")

    families = read_csv(stage14 / "source_data" / "stage103_families.csv")
    family_names = list(dict.fromkeys(row["family"] for row in families))
    controllers = list(dict.fromkeys(row["controller"] for row in families))
    matrix = np.asarray([[float(next(row for row in families if row["controller"] == controller and row["family"] == family)["interception_success"]) for family in family_names] for controller in controllers])
    figure, axis = plt.subplots(figsize=(8.8, 3.4), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="magma")
    for row_index in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row_index, f"{matrix[row_index, column]:.0%}", ha="center", va="center", color="white" if matrix[row_index, column] < .55 else "black", fontsize=8)
    axis.set_xticks(range(len(family_names)), family_names, rotation=20, ha="right")
    axis.set_yticks(range(len(controllers)), controllers)
    axis.set_title("Robustness by unseen deception family")
    figure.colorbar(image, ax=axis, label="Interception success")
    created += _save_pair(figure, appendix_dir / "robustness_family_matrix")

    goals = read_csv(stage14 / "source_data" / "stage103_goals.csv")
    figure, axis = plt.subplots(figsize=(6.2, 4), constrained_layout=True)
    x = np.arange(len(controllers))
    for offset, goal in enumerate(("A", "B")):
        values = [float(next(row for row in goals if row["controller"] == controller and row["goal"] == goal)["interception_success"]) for controller in controllers]
        axis.bar(x + (offset - .5) * .36, values, .36, label=f"Goal {goal}")
    axis.set_xticks(x, controllers, rotation=15)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Interception success")
    axis.set_title("Stage10.3 goal-conditioned asymmetry")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=.25)
    created += _save_pair(figure, appendix_dir / "goal_asymmetry")

    modes = json.loads((stage14 / "source_data" / "stage92_results.json").read_text())
    ceiling = json.loads((stage14 / "source_data" / "stage92_identifiability.json").read_text())[0]["empirical_accuracy_ceiling"]
    figure, axis = plt.subplots(figsize=(6.2, 4), constrained_layout=True)
    mode_names = list(modes)
    x = np.arange(len(mode_names))
    axis.bar(x - .2, [modes[m]["accuracy"] for m in mode_names], .2, label="Accuracy")
    axis.bar(x, [modes[m]["f1"] for m in mode_names], .2, label="F1")
    axis.bar(x + .2, [modes[m]["auroc"] for m in mode_names], .2, label="AUROC")
    axis.axhline(ceiling, linestyle="--", color="black", label="Empirical accuracy ceiling")
    axis.set_xticks(x, ["Trajectory", "+ timing", "+ belief"])
    axis.set_ylim(0.8, .96)
    axis.set_title("Identifiability input modes")
    axis.legend(frameon=False, fontsize=8, ncols=2)
    axis.grid(axis="y", alpha=.25)
    created += _save_pair(figure, appendix_dir / "identifiability_modes")

    rewards = read_csv(stage14 / "source_data" / "stage73_goals.csv")
    experiments = list(dict.fromkeys(row["experiment"] for row in rewards))
    figure, axis = plt.subplots(figsize=(6.2, 4), constrained_layout=True)
    x = np.arange(len(experiments))
    for offset, goal in enumerate(("A", "B")):
        values = [float(next(row for row in rewards if row["experiment"] == experiment and row["goal"] == goal)["task_success_rate"]) for experiment in experiments]
        axis.bar(x + (offset - .5) * .36, values, .36, label=f"Goal {goal}")
    axis.set_xticks(x, [name.split("_", 1)[0] for name in experiments])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Task success")
    axis.set_title("Reward ablations preserve A/B collapse")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=.25)
    created += _save_pair(figure, appendix_dir / "reward_goal_symmetry")
    return created


def audit_claims(registry: list[dict], sections: dict[str, str], bibliography: str):
    manuscript = "\n".join(sections.values())
    referenced = set(re.findall(r"\\claim\{([A-Z][0-9]{2})\}", manuscript))
    known = {row["claim_id"] for row in registry}
    unknown = sorted(referenced - known)
    missing = sorted(row["claim_id"] for row in registry if row["in_manuscript"] and row["claim_id"] not in referenced)
    rejected_referenced = sorted(row["claim_id"] for row in registry if row["status"] == "rejected" and row["claim_id"] in referenced)
    unverified = sorted(row["claim_id"] for row in registry if row["in_manuscript"] and not row["verified"])
    missing_citations = sorted(row["evidence_selector"] for row in registry if row["claim_type"] == "literature" and row["evidence_selector"] not in bibliography)
    missing_proofs = []
    if "T01" in referenced and not all(
        token in sections["theory"]
        for token in (r"\begin{proposition}", r"\begin{proof}", r"\end{proof}")
    ):
        missing_proofs.append("T01")
    passed = not (
        unknown
        or missing
        or rejected_referenced
        or unverified
        or missing_citations
        or missing_proofs
    )
    return {
        "passed": passed,
        "registered_claims": len(registry),
        "manuscript_claims": len(referenced),
        "supported_manuscript_claims": len(referenced) if passed else len(referenced) - len(unverified) - len(rejected_referenced),
        "rejected_candidate_claims": sum(row["status"] == "rejected" for row in registry),
        "unknown_claim_ids": unknown,
        "missing_manuscript_claim_ids": missing,
        "rejected_claims_in_manuscript": rejected_referenced,
        "unverified_claims": unverified,
        "missing_citation_keys": missing_citations,
        "missing_proofs": missing_proofs,
    }


def claim_audit_markdown(registry: list[dict], audit: dict) -> str:
    lines = [
        "# Stage 15 Claim Audit",
        "",
        f"Audit status: **{'PASS' if audit['passed'] else 'FAIL'}**.",
        "",
        f"The manuscript contains {audit['manuscript_claims']} registered claims. "
        f"The registry also blocks {audit['rejected_candidate_claims']} candidate claims that exceed the evidence.",
        "",
        "| ID | Type | Status | Claim | Evidence | Qualification |",
        "|---|---|---|---|---|---|",
    ]
    for row in registry:
        claim = row["claim"].replace("|", "/")
        evidence = f"{row['evidence_artifact']} [{row['evidence_selector']}]".replace("|", "/")
        qualification = row["qualification"].replace("|", "/")
        lines.append(f"| {row['claim_id']} | {row['claim_type']} | {row['status']} | {claim} | {evidence} | {qualification} |")
    lines.extend(
        [
            "",
            "## Enforcement policy",
            "",
            "Every empirical, theoretical, implementation, and literature claim in the skeleton uses a `\\claim{ID}{...}` tag. The rendered PDF hides the tag, while this audit verifies registry membership, evidence checks, citation keys, and exclusion of rejected claims.",
            "",
            "Rejected claims are retained in the registry so later edits cannot silently reintroduce universal robustness, cross-environment transfer, or statistical-superiority language.",
        ]
    )
    return "\n".join(lines) + "\n"


def rebuttal_limitations_markdown() -> str:
    return """# Rebuttal-Style Limitations

## 1. “The evaluation is too small to support a broad superiority claim.”

We agree. The shared table uses one seed and 34 episodes per attack–method cell. The revision limits the claim to descriptive performance under this protocol, reports every cell size, and removes statistical-superiority language. Multi-seed confidence intervals and additional maps remain necessary. **Evidence: M04; rejected claim X03.**

## 2. “The adaptive attacker is hand-written, so this is not a strategic equilibrium.”

Correct. The attacker changes behavior using completed Defender actions and cross-episode profiles, but it is neither optimal nor learned. We describe it as an online adaptive stress test and make no equilibrium or minimax claim. **Evidence: M03.**

## 3. “Adding a second environment interface does not demonstrate transfer.”

Correct. Forked Routes validates API compatibility, finite dynamics, obstacles, and A/B path symmetry. The frozen neural encoder has not been evaluated on that observation distribution. Cross-environment transfer is therefore listed as untested and blocked by claim ID X02. **Evidence: E07 and M05.**

## 4. “Perfect AUROC with zero interception undermines the detection story.”

It narrows the story in a useful way. The result shows why detection and commitment must be evaluated separately. We report detection, false commitment, interception, sensing cost, latency, and regret rather than treating classifier quality as defense success. **Evidence: E03 and E08.**

## 5. “The Bayesian and POMDP baselines use favorable simplifications.”

They exactly solve their declared reduced models, but those models assume approximately honest motion and fixed sensing kernels. Strategic feints violate these assumptions. We present the failures as evidence of model mismatch, not as a general weakness of Bayesian inference or POMDP planning. **Evidence: T02, E03, and E08.**

## 6. “Trajectory-only deception can be unidentifiable.”

We agree and formalize this limit. When honest and deceptive policies induce the same visible-history distribution, no classifier using that history can improve on the prior Bayes decision. The dataset’s 93.75% ceiling is empirical and should not be generalized beyond the constructed distribution. **Evidence: T01 and E05.**

## 7. “Long feints and goal asymmetry remain unsolved.”

Yes. All tested controllers fail on scripted steps 8 and 10, and Stage10.3 contains a strong A/B performance gap despite symmetric map geometry. These are reported as central failure cases. Stage15 performs no tuning or additional training to hide them. **Evidence: E04 and E10.**
"""


def compile_latex(destination: Path) -> dict:
    executable = shutil.which("latexmk")
    if executable is None:
        return {"attempted": False, "passed": False, "reason": "latexmk not installed"}
    process = subprocess.run(
        [executable, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"],
        cwd=destination,
        capture_output=True,
        text=True,
        timeout=120,
    )
    log = process.stdout + "\n" + process.stderr
    (destination / "latex_build.log").write_text(log)
    for suffix in ("aux", "bbl", "blg", "fdb_latexmk", "fls", "log", "out"):
        generated = destination / f"paper.{suffix}"
        if generated.exists():
            generated.unlink()
    return {
        "attempted": True,
        "passed": process.returncode == 0 and (destination / "paper.pdf").is_file(),
        "returncode": process.returncode,
        "log": "latex_build.log",
    }


def build(destination=OUTPUT, stage14=STAGE14, compile_pdf=True):
    destination = Path(destination)
    stage14 = Path(stage14)
    required = [
        stage14 / "main_table.csv",
        stage14 / "ablation_table.csv",
        stage14 / "failure_case_table.csv",
        stage14 / "source_data" / "stage103_families.csv",
        stage14 / "source_data" / "stage103_goals.csv",
        stage14 / "source_data" / "stage73_goals.csv",
        stage14 / "source_data" / "stage92_results.json",
        stage14 / "source_data" / "stage92_identifiability.json",
        stage14 / "source_data" / "interfaces.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen Stage14 inputs: " + ", ".join(missing))
    sections_dir = destination / "sections"
    tables_dir = destination / "tables"
    source_dir = destination / "source_data"
    for directory in (destination, sections_dir, tables_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for path in required:
        target = source_dir / path.name
        if target.exists() and target.resolve() == path.resolve():
            continue
        shutil.copy2(path, target)

    main = read_csv(stage14 / "main_table.csv")
    ablations = read_csv(stage14 / "ablation_table.csv")
    failures = read_csv(stage14 / "failure_case_table.csv")
    (tables_dir / "main_results.tex").write_text(main_results_table(main))
    (tables_dir / "ablation_results.tex").write_text(ablation_results_table(ablations))
    (tables_dir / "failure_cases.tex").write_text(failure_results_table(failures))

    sections = section_texts(main)
    for name in REQUIRED_SECTIONS:
        (sections_dir / f"{name}.tex").write_text(sections[name])
    bibliography = references_bib()
    (destination / "references.bib").write_text(bibliography)
    (destination / "paper.tex").write_text(paper_tex())
    registry = build_claim_registry(stage14)
    audit = audit_claims(registry, sections, bibliography)
    if not audit["passed"]:
        raise AssertionError(f"Claim audit failed: {audit}")
    write_csv(destination / "claim_registry.csv", registry)
    (destination / "claim_audit.json").write_text(json.dumps(audit, indent=2))
    (destination / "claim_audit.md").write_text(claim_audit_markdown(registry, audit))
    (destination / "rebuttal_limitations.md").write_text(rebuttal_limitations_markdown())
    figures = generate_figures(stage14, destination)
    latex = compile_latex(destination) if compile_pdf else {"attempted": False, "passed": None, "reason": "disabled"}
    if compile_pdf and not latex["passed"]:
        raise RuntimeError(f"LaTeX compilation failed; see {destination / 'latex_build.log'}")
    inputs = {
        str(path.relative_to(stage14)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in required
    }
    manifest = {
        "generated_by": "experiments.build_stage15_paper",
        "stage14_inputs": inputs,
        "required_sections": list(REQUIRED_SECTIONS),
        "tables": [str(path.relative_to(destination)) for path in sorted(tables_dir.glob("*.tex"))],
        "figures": [str(path.relative_to(destination)) for path in figures],
        "claim_audit": audit,
        "latex": latex,
        "algorithms_modified": False,
        "training_started": False,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--stage14-dir", type=Path, default=STAGE14)
    parser.add_argument("--no-compile", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.output_dir, arguments.stage14_dir, not arguments.no_compile), indent=2))

