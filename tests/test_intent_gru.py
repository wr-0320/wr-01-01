import csv
import json

import numpy as np
import torch

from deception_rl.beliefs.gru import GRUBeliefEstimator
from deception_rl.envs.gridworld import GridWorldParallelEnv
from models.intent_gru import IntentGRU, trajectory_to_features
from train_intent_model import train


def test_trajectory_features_encode_positions_and_time():
    features = trajectory_to_features([[0, 3, 3], [2, 2, 3]])

    assert features.shape == (2, 6)
    assert torch.isclose(features[1, 2], torch.tensor(2 / 30))
    assert torch.isclose(features[1, 3], torch.tensor(-1 / 6))
    assert torch.isclose(features[1, 5], torch.tensor(2 / 30))


def test_gru_forward_supports_variable_length_batches():
    model = IntentGRU(hidden_size=8)
    logits = model(
        torch.zeros((2, 3, 6)),
        torch.tensor([3, 1]),
    )

    assert logits.shape == (2, 2)


def test_gru_estimator_returns_normalized_belief_and_uniform_empty_prior():
    env = GridWorldParallelEnv()
    observations, _ = env.reset(seed=2)
    model = IntentGRU(hidden_size=8)
    estimator = GRUBeliefEstimator(model)
    observation = observations["defender"]

    empty = estimator.estimate(
        observation["trajectory"],
        np.zeros_like(observation["trajectory_mask"]),
        observation["target_positions"],
    )
    predicted = estimator.estimate(
        observation["trajectory"],
        observation["trajectory_mask"],
        observation["target_positions"],
    )

    assert empty == {"A": 0.5, "B": 0.5}
    assert set(predicted) == {"A", "B"}
    assert np.isclose(sum(predicted.values()), 1.0)


def test_training_uses_baseline_csv_and_saves_loadable_checkpoint(tmp_path):
    rows = [
        {
            "deceiver_policy": "honest_shortest_path",
            "true_goal": goal,
            "trajectory": json.dumps([[0, 3, 3], [1, row, 3]]),
        }
        for goal, row in [("A", 2), ("A", 1), ("B", 4), ("B", 5)]
    ]
    data_path = tmp_path / "baseline.csv"
    with data_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["deceiver_policy", "true_goal", "trajectory"],
        )
        writer.writeheader()
        writer.writerows(rows)

    checkpoint = tmp_path / "intent_gru.pt"
    metrics = train(
        data_path=data_path,
        output_path=checkpoint,
        epochs=3,
        patience=2,
        hidden_size=8,
        seed=1,
    )
    estimator = GRUBeliefEstimator.from_checkpoint(checkpoint)

    assert checkpoint.is_file()
    assert metrics["unique_trajectories"] == 4
    assert estimator.model.hidden_size == 8
