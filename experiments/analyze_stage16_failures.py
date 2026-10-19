"""Stage 16 failure attribution for the frozen Stage13 controller.

The script replays the existing Stage15 GridWorld schedule only to expose
diagnostic values that were not serialized there.  It performs no optimization,
checkpoint write, threshold search, or model/environment modification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from deception_rl.agents import HonestShortestPath, ScriptedDeception
from deception_rl.envs import GridWorldAdapter
from experiments.run_stage15_robustness import load_stage13_controller
from experiments.stage103_deception_families import policy as grid_family_policy
from models.stage13_decision import ACTION_NAMES


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/stage15_robustness/episodes.csv"
OUTPUT = ROOT / "outputs/stage16_failure_attribution"
SEEN = ("scripted_3",)
UNSEEN = (
    "scripted_8",
    "scripted_10",
    "delay_detour",
    "detour_switch",
    "stochastic_deception",
)
REFERENCE = ("honest",)
FAMILIES = REFERENCE + SEEN + UNSEEN
ENV_ACTION_NAMES = ("commit_A", "commit_B", "observe", "track")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def protected_hashes() -> dict[str, str]:
    paths = [
        ROOT / "src/deception_rl/envs/gridworld.py",
        ROOT / "experiments/stage13_learned_controller.py",
        ROOT / "outputs/stage13_learned_controller/stage13_controller.pt",
        Path("/workspace/stage10-two-level/belief_model.pt"),
        Path("/workspace/stage10-1-decoupled/calibrator.pt"),
    ]
    result = {str(path): file_hash(path) for path in paths}
    result.update(tree_hashes(ROOT / "outputs/stage15_paper"))
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def load_schedule(path=SOURCE) -> list[dict[str, object]]:
    selected = []
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            if (
                row["environment"] == "gridworld"
                and row["controller"] == "stage13_learned"
                and row["family"] in FAMILIES
            ):
                selected.append(
                    {
                        "family": row["family"],
                        "evaluation_seed": int(row["evaluation_seed"]),
                        "episode": int(row["episode"]),
                        "simulator_seed": int(row["simulator_seed"]),
                        "true_goal": row["true_goal"],
                        "expected_interception": int(row["interception_success"]),
                        "expected_false_commitment": int(row["false_commitment"]),
                    }
                )
    expected = 5 * len(FAMILIES) * 40
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} Stage15 scenarios, found {len(selected)}")
    return selected


def group_for(family: str) -> str:
    if family in REFERENCE:
        return "honest_reference"
    if family in SEEN:
        return "seen"
    if family in UNSEEN:
        return "unseen"
    raise ValueError(f"Unknown family: {family}")


def trajectory_signature(trajectory) -> str:
    return json.dumps(trajectory, separators=(",", ":"))


def replay(schedule):
    adapter = GridWorldAdapter()
    controller = load_stage13_controller()
    states = []
    episodes = []
    for scenario_id, scenario in enumerate(schedule):
        env = adapter.make_env()
        observations, _ = env.reset(
            seed=scenario["simulator_seed"], options={"goal": scenario["true_goal"]}
        )
        if scenario["family"] == "honest":
            attacker = HonestShortestPath()
        elif scenario["family"] == "scripted_3":
            attacker = ScriptedDeception(3)
        else:
            attacker = grid_family_policy(scenario["family"])
        attacker.reset(seed=scenario["simulator_seed"])
        controller.reset()
        timestep = 0
        intercepted = escaped = False
        decision_state_index = None
        while env.agents:
            was_committed = controller.goal is not None
            defender_action = int(controller.act(observations["defender"], timestep))
            info = controller.info
            if info is None:
                raise AssertionError("Frozen Stage13 controller did not expose diagnostics")
            belief = np.asarray(info["goal_belief"], dtype=float)
            true_index = int(scenario["true_goal"] == "B")
            ratio = max(float(info["likelihood_ratio"]), 1e-12)
            features = np.asarray(info["decision_features"], dtype=float)
            new_commitment = not was_committed and controller.goal is not None
            row = {
                "scenario_id": scenario_id,
                "family": scenario["family"],
                "group": group_for(scenario["family"]),
                "evaluation_seed": scenario["evaluation_seed"],
                "episode": scenario["episode"],
                "simulator_seed": scenario["simulator_seed"],
                "true_goal": scenario["true_goal"],
                "timestep": timestep,
                "precommit": int(not was_committed),
                "new_commitment": int(new_commitment),
                "actual_action": ENV_ACTION_NAMES[defender_action],
                "learned_action": info["learned_action"],
                "belief_A": float(belief[0]),
                "belief_B": float(belief[1]),
                "belief_true": float(belief[true_index]),
                "belief_error": float(1 - belief[true_index]),
                "goal_correct": int(int(np.argmax(belief)) == true_index),
                "confidence": float(info["confidence"]),
                "entropy": float(info["entropy"]),
                "deception_probability": float(info["deception_probability"]),
                "honest_likelihood": float(info["honest_likelihood"]),
                "deceptive_likelihood": float(info["deceptive_likelihood"]),
                "likelihood_ratio": ratio,
                "log_likelihood_ratio": float(np.log(ratio)),
                "counterfactual_score": float(features[3]),
                "observation_cost": float(features[4]),
                "decision_logits": info["decision_logits"],
                "decision_probabilities": info["decision_probabilities"],
                "trajectory_signature": trajectory_signature(info["trajectory"]),
                "visible_points": len(info["trajectory"]),
            }
            states.append(row)
            if new_commitment:
                decision_state_index = len(states) - 1
            deceiver_action = int(attacker.act(observations["deceiver"]))
            observations, rewards, _, _, _ = env.step(
                {"deceiver": deceiver_action, "defender": defender_action}
            )
            intercepted = bool(rewards["defender"] > 0)
            escaped = bool(rewards["deceiver"] > 0)
            timestep += 1
        env.close()
        if decision_state_index is None:
            candidates = [
                index
                for index, state in enumerate(states)
                if state["scenario_id"] == scenario_id and state["precommit"]
            ]
            decision_state_index = candidates[-1]
        decision = states[decision_state_index]
        episode = {
            **scenario,
            "scenario_id": scenario_id,
            "group": group_for(scenario["family"]),
            "interception_success": int(intercepted),
            "deceiver_success": int(escaped),
            "failed": int(not intercepted),
            "committed": int(controller.goal is not None),
            "false_commitment": int(
                controller.goal is not None
                and controller.goal != int(scenario["true_goal"] == "B")
            ),
            "commit_timestep": controller.commit_timestep,
            "decision_timestep": decision["timestep"],
            "decision_belief_true": decision["belief_true"],
            "decision_belief_error": decision["belief_error"],
            "decision_goal_correct": decision["goal_correct"],
            "decision_confidence": decision["confidence"],
            "decision_entropy": decision["entropy"],
            "decision_counterfactual_score": decision["counterfactual_score"],
            "decision_log_likelihood_ratio": decision["log_likelihood_ratio"],
            "decision_deception_probability": decision["deception_probability"],
            "decision_trajectory_signature": decision["trajectory_signature"],
        }
        if episode["interception_success"] != scenario["expected_interception"]:
            raise AssertionError("Diagnostic replay changed the Stage15 interception outcome")
        if episode["false_commitment"] != scenario["expected_false_commitment"]:
            raise AssertionError("Diagnostic replay changed the Stage15 commitment outcome")
        episodes.append(episode)
    return states, episodes


def state_at(states, scenario_id: int, timestep: int):
    candidates = [
        row
        for row in states
        if row["scenario_id"] == scenario_id and row["timestep"] <= timestep
    ]
    return candidates[-1]


def per_seed_gap(states, episodes):
    rows = []
    for seed in sorted({row["evaluation_seed"] for row in episodes}):
        for group in ("seen", "unseen"):
            selected_episodes = [
                row for row in episodes if row["evaluation_seed"] == seed and row["group"] == group
            ]
            t5 = [state_at(states, row["scenario_id"], 5) for row in selected_episodes]
            selected_states = [
                row
                for row in states
                if row["evaluation_seed"] == seed and row["group"] == group and row["precommit"]
            ]
            rows.append(
                {
                    "evaluation_seed": seed,
                    "group": group,
                    "episodes": len(selected_episodes),
                    "belief_error_t5": float(np.mean([row["belief_error"] for row in t5])),
                    "goal_accuracy_t5": float(np.mean([row["goal_correct"] for row in t5])),
                    "belief_error_at_decision": float(
                        np.mean([row["decision_belief_error"] for row in selected_episodes])
                    ),
                    "goal_accuracy_at_decision": float(
                        np.mean([row["decision_goal_correct"] for row in selected_episodes])
                    ),
                    "precommit_belief_error": float(
                        np.mean([row["belief_error"] for row in selected_states])
                    ),
                    "counterfactual_score_mean": float(
                        np.mean([row["counterfactual_score"] for row in selected_states])
                    ),
                    "interception_success": float(
                        np.mean([row["interception_success"] for row in selected_episodes])
                    ),
                }
            )
    return rows


def summarize_gap(seed_rows):
    metrics = [
        key
        for key in seed_rows[0]
        if key not in {"evaluation_seed", "group", "episodes"}
    ]
    output = []
    for group in ("seen", "unseen"):
        selected = [row for row in seed_rows if row["group"] == group]
        result = {"group": group, "seeds": len(selected), "episodes_per_seed": selected[0]["episodes"]}
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=1))
        output.append(result)
    seen = output[0]
    unseen = output[1]
    gap = {"group": "unseen_minus_seen", "seeds": 5, "episodes_per_seed": ""}
    for metric in metrics:
        paired = []
        for seed in range(5):
            lookup = {(row["evaluation_seed"], row["group"]): row for row in seed_rows}
            paired.append(lookup[(seed, "unseen")][metric] - lookup[(seed, "seen")][metric])
        values = np.asarray(paired)
        gap[f"{metric}_mean"] = float(values.mean())
        gap[f"{metric}_std"] = float(values.std(ddof=1))
    output.append(gap)
    return output


def counterfactual_distribution(states):
    rows = []
    for family in FAMILIES:
        selected = [
            row["counterfactual_score"]
            for row in states
            if row["family"] == family and row["precommit"]
        ]
        values = np.asarray(selected)
        rows.append(
            {
                "family": family,
                "group": group_for(family),
                "states": len(values),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "q05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "q95": float(np.quantile(values, 0.95)),
                "fraction_nonpositive": float(np.mean(values <= 0)),
            }
        )
    return rows


def attribute_failures(states, episodes):
    seen_scores = np.asarray(
        [
            row["counterfactual_score"]
            for row in states
            if row["group"] == "seen" and row["precommit"]
        ]
    )
    seen_interval = tuple(float(value) for value in np.quantile(seen_scores, (0.025, 0.975)))
    signature_goals = defaultdict(set)
    for episode in episodes:
        signature_goals[episode["decision_trajectory_signature"]].add(episode["true_goal"])
    failures = []
    for episode in episodes:
        if episode["group"] == "honest_reference" or not episode["failed"]:
            continue
        aliasing = len(signature_goals[episode["decision_trajectory_signature"]]) > 1
        shifted = episode["group"] == "unseen" and not (
            seen_interval[0]
            <= episode["decision_counterfactual_score"]
            <= seen_interval[1]
        )
        future_states = [
            row
            for row in states
            if row["scenario_id"] == episode["scenario_id"]
            and row["timestep"] > episode["decision_timestep"]
        ]
        recovery_times = [row["timestep"] for row in future_states if row["goal_correct"]]
        premature = bool(episode["false_commitment"] and recovery_times)
        wrong_counterfactual = bool(episode["decision_log_likelihood_ratio"] <= 0)
        if aliasing:
            primary = "observational_aliasing"
        elif premature:
            primary = "premature_commitment"
        elif wrong_counterfactual:
            primary = "wrong_counterfactual_assumption"
        elif episode["group"] == "unseen":
            primary = "attacker_distribution_shift"
        else:
            primary = "unattributed"
        failures.append(
            {
                **episode,
                "observational_aliasing": int(aliasing),
                "attacker_distribution_shift": int(episode["group"] == "unseen"),
                "counterfactual_outside_seen_95pct": int(shifted),
                "premature_commitment": int(premature),
                "first_postcommit_belief_recovery": min(recovery_times) if recovery_times else None,
                "wrong_counterfactual_assumption": int(wrong_counterfactual),
                "primary_attribution": primary,
            }
        )
    return failures, seen_interval


def failure_table(failures):
    rows = []
    categories = (
        "observational_aliasing",
        "attacker_distribution_shift",
        "premature_commitment",
        "wrong_counterfactual_assumption",
    )
    groups = sorted({(row["group"], row["family"]) for row in failures})
    for group, family in groups:
        family_failures = [row for row in failures if row["family"] == family]
        for category in categories:
            selected = [row for row in family_failures if row[category]]
            rows.append(
                {
                    "group": group,
                    "family": family,
                    "attribution": category,
                    "failure_cases": len(selected),
                    "share_of_family_failures": len(selected) / len(family_failures),
                    "mean_belief_error_at_decision": float(
                        np.mean([row["decision_belief_error"] for row in selected])
                    )
                    if selected
                    else None,
                    "mean_counterfactual_score": float(
                        np.mean([row["decision_counterfactual_score"] for row in selected])
                    )
                    if selected
                    else None,
                    "mean_commit_timestep": float(
                        np.mean(
                            [
                                row["commit_timestep"]
                                for row in selected
                                if row["commit_timestep"] is not None
                            ]
                        )
                    )
                    if any(row["commit_timestep"] is not None for row in selected)
                    else None,
                }
            )
    return rows


def decision_boundary(model, states, destination: Path):
    precommit = [row for row in states if row["precommit"]]
    fixed_cost = float(np.median([row["observation_cost"] for row in precommit]))
    probabilities = np.linspace(0.001, 0.999, 180)
    scores = np.linspace(-1, 1, 180)
    grid = []
    for score in scores:
        for probability_b in probabilities:
            entropy = -probability_b * np.log2(probability_b) - (1 - probability_b) * np.log2(1 - probability_b)
            grid.append([1 - probability_b, probability_b, entropy, score, fixed_cost, 0.0])
    with torch.inference_mode():
        actions = model(torch.tensor(grid, dtype=torch.float32)).argmax(dim=1).numpy()
    action_grid = actions.reshape(len(scores), len(probabilities))
    colors = plt.matplotlib.colors.ListedColormap(["#8dd3c7", "#80b1d3", "#fb8072", "#fdb462"])
    fig, ax = plt.subplots(figsize=(8, 5.4))
    image = ax.imshow(
        action_grid,
        origin="lower",
        extent=(0, 1, -1, 1),
        aspect="auto",
        interpolation="nearest",
        cmap=colors,
        vmin=-0.5,
        vmax=3.5,
        alpha=0.72,
    )
    for group, marker, edge in (("seen", "o", "black"), ("unseen", "^", "white")):
        points = [row for row in precommit if row["group"] == group and row["new_commitment"]]
        ax.scatter(
            [row["belief_B"] for row in points],
            [row["counterfactual_score"] for row in points],
            marker=marker,
            facecolors="none",
            edgecolors=edge,
            linewidths=0.7,
            s=22,
            label=f"{group} commitment states",
        )
    colorbar = fig.colorbar(image, ax=ax, ticks=range(4))
    colorbar.ax.set_yticklabels(ACTION_NAMES)
    ax.set_xlabel("Calibrated P(goal=B)")
    ax.set_ylabel("Counterfactual score tanh(log LR)")
    ax.set_title(f"Frozen Stage13 decision regions at observation cost={fixed_cost:.3f}")
    ax.legend(loc="lower center", fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "decision_boundary.png", dpi=220)
    fig.savefig(destination / "decision_boundary.pdf")
    plt.close(fig)
    return fixed_cost


def plots(destination, gap, distributions, failures):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    groups = [row for row in gap if row["group"] in ("seen", "unseen")]
    x = np.arange(2)
    ax.bar(
        x,
        [row["belief_error_at_decision_mean"] for row in groups],
        yerr=[row["belief_error_at_decision_std"] for row in groups],
        capsize=4,
        color=("#4C78A8", "#F58518"),
    )
    ax.set_xticks(x, ["Seen", "Unseen"])
    ax.set_ylabel("1 - P(true goal) at decision")
    ax.set_title("Stage13 belief generalization gap (five seeds)")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(destination / "belief_generalization_gap.png", dpi=220)
    fig.savefig(destination / "belief_generalization_gap.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(
        np.arange(len(distributions)),
        [row["median"] for row in distributions],
        yerr=[
            [row["median"] - row["q05"] for row in distributions],
            [row["q95"] - row["median"] for row in distributions],
        ],
        fmt="o",
        capsize=4,
    )
    ax.axhline(0, color="black", linewidth=.8, linestyle="--")
    ax.set_xticks(np.arange(len(distributions)), [row["family"] for row in distributions], rotation=30, ha="right")
    ax.set_ylabel("Counterfactual score (median, 5--95%)")
    ax.set_title("Counterfactual score distribution by family")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(destination / "counterfactual_score_distribution.png", dpi=220)
    fig.savefig(destination / "counterfactual_score_distribution.pdf")
    plt.close(fig)

    counts = Counter(row["primary_attribution"] for row in failures)
    categories = (
        "observational_aliasing",
        "attacker_distribution_shift",
        "premature_commitment",
        "wrong_counterfactual_assumption",
        "unattributed",
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(categories, [counts[category] for category in categories], color="#E45756")
    ax.set_ylabel("Primary-attribution failure episodes")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_title("Stage13 failure taxonomy")
    fig.tight_layout()
    fig.savefig(destination / "failure_taxonomy.png", dpi=220)
    fig.savefig(destination / "failure_taxonomy.pdf")
    plt.close(fig)


def render_reports(destination, gap, distributions, failures, table, interval, boundary_cost):
    by_group = {row["group"]: row for row in gap}
    seen = by_group["seen"]
    unseen = by_group["unseen"]
    difference = by_group["unseen_minus_seen"]
    counts = Counter(row["primary_attribution"] for row in failures)
    total = len(failures)
    lines = [
        "# Stage16 Failure Attribution",
        "",
        "## Protocol",
        "",
        "This is a frozen diagnostic replay of the Stage15 GridWorld schedule. No model is trained or changed, no threshold is selected from outcomes, and no paper source is edited. Seen means ScriptedDeception(steps=3), which belongs to the belief/counterfactual training families. Unseen means scripted steps 8/10, delay+detour, detour+switch, and stochastic deception. Stage13 imitation used an earlier Stage10.1 evaluation artifact that included some held-out-family feature states, so the labels describe representation-level exposure rather than a strict controller-level train/test split.",
        "",
        "Belief error is `1-P(true goal)`. The counterfactual score is `tanh(log(P(o|deceptive)/P(o|honest)))`. Mean±std values below use five seed-level estimates.",
        "",
        "## Generalization gap",
        "",
        f"At decision, seen belief error is {seen['belief_error_at_decision_mean']:.4f}±{seen['belief_error_at_decision_std']:.4f}; unseen is {unseen['belief_error_at_decision_mean']:.4f}±{unseen['belief_error_at_decision_std']:.4f}. The paired unseen-minus-seen gap is {difference['belief_error_at_decision_mean']:+.4f}±{difference['belief_error_at_decision_std']:.4f}.",
        f"Interception is {seen['interception_success_mean']:.4f}±{seen['interception_success_std']:.4f} on seen and {unseen['interception_success_mean']:.4f}±{unseen['interception_success_std']:.4f} on unseen families.",
        "",
        "## Attribution rules",
        "",
        "Observational aliasing requires an exactly identical timed visible-history signature at the decision state for both goals. Premature commitment requires a false irreversible commitment followed by a later state where the same frozen belief model's argmax recovers the true goal. Wrong counterfactual assumption means the deceptive episode has likelihood ratio <=1 at the decision state. Attacker distribution shift marks held-out families; `counterfactual_outside_seen_95pct` separately identifies decision scores outside the predeclared seen 2.5--97.5 percentile interval. Primary labels use the fixed precedence aliasing, premature commitment, wrong counterfactual assumption, distribution shift, unattributed; the failure table and `failure_cases.csv` retain overlapping labels.",
        "",
        f"The seen counterfactual reference interval is [{interval[0]:.4f}, {interval[1]:.4f}]. The decision-boundary plot fixes cumulative observation cost at the empirical precommit median {boundary_cost:.4f}; it is a two-dimensional slice of the unchanged six-dimensional policy.",
        "",
        "## Failure taxonomy",
        "",
        f"There are {total} failed deceptive episodes. Overlapping counts: " + ", ".join(f"{key}={sum(int(row[key]) for row in failures)}" for key in ("observational_aliasing", "attacker_distribution_shift", "premature_commitment", "wrong_counterfactual_assumption")) + ". Primary counts under the declared precedence: " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) + ".",
        "",
        "The taxonomy is diagnostic rather than causal identification: categories are observable signatures under declared rules, and several mechanisms can coexist. Exact aliasing is a sufficient information limitation for the recorded decision history; absence of an exact match does not prove identifiability.",
    ]
    (destination / "generalization_gap_analysis.md").write_text("\n".join(lines) + "\n")

    table_lines = [
        "|Group|Family|Attribution (overlap allowed)|Failures|Family share|Belief error|CF score|Commit t|",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        timing = "N/A" if row["mean_commit_timestep"] is None else f"{row['mean_commit_timestep']:.2f}"
        belief_error = (
            "N/A"
            if row["mean_belief_error_at_decision"] is None
            else f"{row['mean_belief_error_at_decision']:.4f}"
        )
        counterfactual = (
            "N/A"
            if row["mean_counterfactual_score"] is None
            else f"{row['mean_counterfactual_score']:.4f}"
        )
        table_lines.append(
            f"|{row['group']}|{row['family']}|{row['attribution']}|{row['failure_cases']}|{row['share_of_family_failures']:.2%}|{belief_error}|{counterfactual}|{timing}|"
        )
    (destination / "failure_case_table.md").write_text("\n".join(table_lines) + "\n")

    limitation = f"""# Paper Limitation Section Draft

The failure-attribution study exposes a substantial representation-level generalization gap. Across five evaluation seeds, Stage13's belief error at the decision state changes from {seen['belief_error_at_decision_mean']:.4f} ± {seen['belief_error_at_decision_std']:.4f} on scripted steps=3 to {unseen['belief_error_at_decision_mean']:.4f} ± {unseen['belief_error_at_decision_std']:.4f} over five held-out deception families; interception changes from {seen['interception_success_mean']:.4f} ± {seen['interception_success_std']:.4f} to {unseen['interception_success_mean']:.4f} ± {unseen['interception_success_std']:.4f}. The counterfactual module compares trajectories with fixed honest and scripted-1--3 prototypes, so its likelihood ratio is vulnerable when an attacker changes timing, route shape, or switching pattern. A confident decision can therefore reflect prototype mismatch rather than reliable goal evidence.

Our taxonomy should not be read as causal ground truth. Exact observable-history collisions support an observational-aliasing diagnosis, but non-colliding histories can remain statistically indistinguishable. Premature commitment and counterfactual mismatch can overlap, and the fixed precedence used for the summary table hides that overlap while the episode table preserves it. The decision-boundary figure is a two-dimensional slice at a fixed observation cost of {boundary_cost:.4f}; the deployed controller also depends on entropy, cumulative cost, and commitment status.

The seen/unseen terminology applies to the frozen belief and counterfactual training families. Stage13 learned its decision rule by imitating a Stage10.1 evaluation artifact that included feature states from some held-out families, so this experiment is not a strict end-to-end zero-shot split. Results cover one GridWorld topology, five evaluation seeds, and hand-specified attackers. They do not establish robustness to arbitrary strategic deception or identify a universally valid counterfactual model.
"""
    (destination / "paper_limitation_section.md").write_text(limitation)
    tex = limitation.replace("# Paper Limitation Section Draft\n\n", "\\section{Limitations}\n")
    tex = tex.replace("±", r"$\pm$").replace("steps=3", r"steps$=3$")
    (destination / "paper_limitation_section.tex").write_text(tex + "\n")


def run(destination=OUTPUT):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    before = protected_hashes()
    schedule = load_schedule()
    states, episodes = replay(schedule)
    seed_rows = per_seed_gap(states, episodes)
    gap = summarize_gap(seed_rows)
    distributions = counterfactual_distribution(states)
    failures, interval = attribute_failures(states, episodes)
    table = failure_table(failures)
    boundary_cost = decision_boundary(load_stage13_controller().model, states, destination)
    plots(destination, gap, distributions, failures)
    write_csv(destination / "diagnostic_states.csv", states)
    write_csv(destination / "diagnostic_episodes.csv", episodes)
    write_csv(destination / "per_seed_generalization.csv", seed_rows)
    write_csv(destination / "belief_generalization_summary.csv", gap)
    write_csv(destination / "counterfactual_distribution.csv", distributions)
    write_csv(destination / "failure_cases.csv", failures)
    write_csv(destination / "failure_case_table.csv", table)
    render_reports(destination, gap, distributions, failures, table, interval, boundary_cost)
    config = {
        "analysis_only": True,
        "training": False,
        "tuning": False,
        "paper_sources_modified": False,
        "source_schedule": str(SOURCE),
        "seen_families": list(SEEN),
        "unseen_families": list(UNSEEN),
        "honest_reference": list(REFERENCE),
        "episodes": len(episodes),
        "diagnostic_states": len(states),
        "taxonomy_precedence": [
            "observational_aliasing",
            "premature_commitment",
            "wrong_counterfactual_assumption",
            "attacker_distribution_shift",
            "unattributed",
        ],
        "premature_rule": "false irreversible commitment followed by later frozen-belief argmax recovery to the true goal",
        "wrong_counterfactual_rule": "likelihood ratio <= 1 on a deceptive episode",
        "shift_score_rule": "decision counterfactual score outside seen precommit 2.5--97.5 percentile interval",
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    after = protected_hashes()
    unchanged = {path: before[path] == after[path] for path in before}
    if not all(unchanged.values()):
        raise AssertionError("A protected model, environment, or paper file changed")
    (destination / "preserved_inputs.json").write_text(
        json.dumps({"sha256_before": before, "sha256_after": after, "unchanged": unchanged}, indent=2)
        + "\n"
    )
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir), indent=2))
