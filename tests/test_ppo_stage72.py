import csv
from pathlib import Path

import numpy as np
import torch

from deception_rl.beliefs.gru import GRUBeliefEstimator
from experiments.ppo_deceiver_env import PPODeceiverEnv
from experiments.run_ppo_stage72 import run_stage72
from models.intent_gru import IntentGRU


class FixedVisibleEstimator:
    def estimate(self, trajectory, trajectory_mask, target_positions):
        assert trajectory_mask.any()
        return {"A": 0.8, "B": 0.2}


def _save_model_a_checkpoint(path: Path) -> None:
    model = IntentGRU(hidden_size=8)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_size": 8,
            "num_layers": 1,
            "grid_size": 7,
            "max_time": 30,
        },
        path,
    )


def test_belief_observation_appends_only_estimated_public_state():
    env = PPODeceiverEnv(
        FixedVisibleEstimator(),
        step_commit=0,
        belief_observation=True,
    )
    initial, _ = env.reset(seed=1, options={"goal": "A"})

    assert env.observation_space.shape == (12,)
    assert initial.shape == (12,)
    np.testing.assert_allclose(initial[8:10], [0.8, 0.2])
    assert 0.0 < initial[10] < 1.0
    assert initial[11] == 0.0
    next_observation, _, _, _, info = env.step(0)
    assert next_observation[11] == 1.0
    assert "belief" not in info


def test_stage72_128_step_smoke_logs_curriculum_and_outputs(tmp_path):
    model_a = tmp_path / "model_a.pt"
    _save_model_a_checkpoint(model_a)
    outputs = run_stage72(
        output_dir=tmp_path / "stage72",
        defender_checkpoint=model_a,
        total_timesteps=128,
        evaluation_episodes=2,
        seed=5,
        step_commit=5,
        device="cpu",
    )

    with outputs["phases"].open(newline="", encoding="utf-8") as csv_file:
        phases = list(csv.DictReader(csv_file))
    curriculum_phases = [
        row for row in phases if row["experiment"] == "Experiment3_belief_curriculum"
    ]
    with outputs["summary"].open(newline="", encoding="utf-8") as csv_file:
        summaries = list(csv.DictReader(csv_file))

    assert [row["phase"] for row in curriculum_phases] == [
        "task_only",
        "task_plus_deception",
        "task_deception_cost",
    ]
    assert sum(int(row["phase_timesteps"]) for row in curriculum_phases) == 128
    assert {
        row["experiment"] for row in summaries if row["policy"] == "ppo_learned_deceiver"
    } == {
        "Experiment1_PPO_no_belief",
        "Experiment2_PPO_with_belief",
        "Experiment3_belief_curriculum",
    }
    assert all(outputs[key].is_file() for key in ("summary", "learning_curves", "episodes"))
    event_files = list(outputs["tensorboard"].rglob("events.out.tfevents.*"))
    assert len(event_files) == 3
    assert outputs["learning_curve_figure"].stat().st_size > 0
    assert outputs["belief_ablation_figure"].stat().st_size > 0
