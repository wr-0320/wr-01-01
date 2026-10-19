from collections import Counter

import numpy as np
import torch
from torch import nn

from deception_rl.envs.gridworld import GUARD_A, OBSERVE
from experiments.run_stage13_learned_controller import (
    policy_metrics,
    schedule,
    split_episodes,
    teacher_dataset,
    train_controller,
)
from experiments.stage13_learned_controller import (
    LearnedDecisionController,
    decision_features,
)


INFO = {
    "goal_belief": [0.8, 0.2],
    "entropy": 0.721928,
    "likelihood_ratio": 2.0,
    "deception_probability": 0.6,
}


def test_stage13_features_are_exactly_requested_decision_inputs():
    info = {**INFO, "hidden_goal": "B", "trajectory": [[0, 3, 3]]}
    features = decision_features(info, 0.17, True)
    assert features.shape == (6,)
    assert np.allclose(features[:3], [0.8, 0.2, 0.721928])
    assert np.isclose(features[3], np.tanh(np.log(2.0)))
    assert np.allclose(features[4:], [0.17, 1.0])
    changed_hidden = {**info, "hidden_goal": "A", "trajectory": [[0, 6, 6]]}
    assert np.array_equal(features, decision_features(changed_hidden, 0.17, True))


def test_teacher_dataset_stops_after_one_post_commit_state():
    episodes = teacher_dataset()
    assert len(episodes) == 128
    for episode in episodes:
        committed = [sample for sample in episode["samples"] if sample["features"][-1] == 1]
        assert len(committed) <= 1
        if committed:
            assert committed[0] is episode["samples"][-1]
    counts = Counter(
        sample["action"] for episode in episodes for sample in episode["samples"]
    )
    assert set(counts) == {0, 1, 2, 3}


def test_episode_split_is_disjoint_and_deterministic():
    episodes = teacher_dataset()
    training, validation = split_episodes(episodes, 0)
    again_training, again_validation = split_episodes(episodes, 0)
    assert [row["episode_id"] for row in training] == [
        row["episode_id"] for row in again_training
    ]
    assert [row["episode_id"] for row in validation] == [
        row["episode_id"] for row in again_validation
    ]
    assert not ({row["episode_id"] for row in training} & {row["episode_id"] for row in validation})


def test_regret_metric_penalizes_probability_on_low_utility_action():
    class Fixed(nn.Module):
        def forward(self, features):
            return torch.tensor([[4.0, 0.0, 0.0, 0.0]]).repeat(len(features), 1)

    features = torch.zeros((1, 6))
    action = torch.tensor([0])
    good = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    bad = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert policy_metrics(Fixed(), features, action, good)["expected_regret"] < 0.1
    assert policy_metrics(Fixed(), features, action, bad)["expected_regret"] > 0.9


def test_learned_commitment_is_irreversible_and_status_reaches_model():
    class Inference:
        def predict(self, observation, timestep):
            return dict(INFO)

    class CommitA(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, state):
            self.inputs.append(state.clone())
            logits = torch.tensor([[0.0, 0.0, 4.0, 1.0]])
            if state[0, -1] == 1:
                logits = torch.tensor([[4.0, 0.0, 0.0, 1.0]])
            return logits

    model = CommitA()
    controller = LearnedDecisionController(Inference(), model)
    assert controller.act({}, 3) == GUARD_A
    assert controller.act({}, 4) == GUARD_A
    assert model.inputs[0][0, -1] == 0
    assert model.inputs[1][0, -1] == 1
    assert controller.commit_timestep == 3


def test_short_offline_training_writes_decision_checkpoint(tmp_path):
    model, metadata = train_controller(tmp_path, seed=0, epochs=2)
    assert (tmp_path / "stage13_controller.pt").exists()
    assert metadata["training_episodes"] + metadata["validation_episodes"] == 128
    assert sum(parameter.numel() for parameter in model.parameters()) == 1412


def test_stage13_evaluation_budget_and_goal_balance():
    scenarios = schedule(0)
    assert len(scenarios) * 2 == 512
    assert Counter(row["attack"] for row in scenarios) == {
        "scripted": 128,
        "adaptive_attacker": 128,
    }
    for attack in ("scripted", "adaptive_attacker"):
        assert Counter(row["goal"] for row in scenarios if row["attack"] == attack) == {
            "A": 64,
            "B": 64,
        }

