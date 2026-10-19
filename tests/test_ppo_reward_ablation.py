import csv

import pytest

from deception_rl.envs.gridworld import MOVE_UP
from experiments.ppo_deceiver_env import PPODeceiverEnv
from experiments.plot_ppo_reward_ablation import plot_learning_curves
from experiments.run_ppo_reward_ablation import (
    _summarize_evaluation,
    reward_ablation_configs,
    run_reward_ablation,
)


class WrongGoalEstimator:
    def estimate(self, trajectory, trajectory_mask, target_positions):
        return {"A": 0.1, "B": 0.9}


@pytest.mark.parametrize(
    ("config_index", "expected_deception", "expected_movement", "expected_total"),
    [
        (0, 0.0, 0.0, 0.0),
        (1, 0.25, 0.0, 0.25),
        (2, 0.25, -0.01, 0.24),
    ],
)
def test_reward_ablation_switches_components(
    config_index,
    expected_deception,
    expected_movement,
    expected_total,
):
    config = reward_ablation_configs()[config_index]
    env = PPODeceiverEnv(
        WrongGoalEstimator(),
        step_commit=0,
        misdirection_reward=config.deception_reward,
        movement_cost=config.movement_cost,
        detour_cost=config.detour_cost,
    )
    env.reset(seed=3, options={"goal": "A"})

    _, reward, _, _, info = env.step(MOVE_UP)

    assert info["deception_reward"] == expected_deception
    assert info["movement_cost"] == expected_movement
    assert reward == pytest.approx(expected_total)


def test_ablation_runner_rejects_less_than_required_timesteps(tmp_path):
    with pytest.raises(ValueError, match="50,000"):
        run_reward_ablation(
            output_dir=tmp_path,
            defender_checkpoint=tmp_path / "not-loaded.pt",
            total_timesteps=49_999,
        )


def test_ablation_summary_compares_with_random_and_scripted():
    rows = []
    for policy, success in (
        ("honest_shortest_path", 0),
        ("random_policy", 0),
        ("scripted_deception", 1),
        ("ppo_learned_deceiver", 1),
    ):
        rows.append(
            {
                "ablation": "A_task_only",
                "policy": policy,
                "deceiver_success": success,
                "commitment_accuracy": 0,
                "misdirection_rate": 1,
                "belief_accuracy": 0.5,
                "belief_shift_true_goal": 0.0,
                "deception_cost": 1,
                "path_efficiency": 0.5,
            }
        )

    summaries = _summarize_evaluation(rows)
    learned = next(row for row in summaries if row["policy"] == "ppo_learned_deceiver")

    assert learned["success_rate_delta_vs_random"] == 1.0
    assert learned["success_rate_delta_vs_scripted"] == 0.0


def test_learning_curve_plot_handles_logged_episode_components(tmp_path):
    curves = tmp_path / "curves.csv"
    fields = [
        "ablation",
        "timesteps",
        "episode_index",
        "episode_reward",
        "task_reward",
        "deception_reward",
        "movement_cost",
        "detour_cost",
        "success",
        "rolling_success_rate",
    ]
    with curves.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for ablation in ("A_task_only", "B_task_plus_deception", "C_task_deception_cost"):
            for episode in range(3):
                writer.writerow(
                    {
                        "ablation": ablation,
                        "timesteps": (episode + 1) * 30,
                        "episode_index": episode + 1,
                        "episode_reward": episode,
                        "task_reward": 1,
                        "deception_reward": 0.25,
                        "movement_cost": -0.2,
                        "detour_cost": -0.1,
                        "success": episode % 2,
                        "rolling_success_rate": 0.5,
                    }
                )

    figure = plot_learning_curves(curves, tmp_path / "learning.png")

    assert figure.is_file() and figure.stat().st_size > 0
