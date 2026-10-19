import numpy as np

from deception_rl.envs.gridworld import (
    MOVE_LEFT,
    MOVE_UP,
    MOVE_STAY,
)
from experiments.ppo_deceiver_env import (
    OBSERVATION_SIZE,
    PPODeceiverEnv,
    encode_deceiver_observation,
)


class FixedEstimator:
    def __init__(self, belief):
        self.belief = belief

    def estimate(self, trajectory, trajectory_mask, target_positions):
        return self.belief.copy()


def test_execution_encoder_uses_only_deceiver_observation_fields():
    observation = {
        "position": np.array([3, 3]),
        "goal": 0,
        "target_positions": np.array([[0, 0], [6, 6]]),
    }

    encoded = encode_deceiver_observation(observation)

    assert encoded.shape == (OBSERVATION_SIZE,)
    assert encoded.dtype == np.float32
    assert encoded[2:4].tolist() == [1.0, 0.0]


def test_wrong_public_commitment_shapes_without_belief_observation():
    env = PPODeceiverEnv(
        FixedEstimator({"A": 0.1, "B": 0.9}),
        step_commit=0,
    )
    observation, _ = env.reset(seed=1, options={"goal": "A"})
    next_observation, reward, _, _, info = env.step(MOVE_UP)

    assert observation.shape == next_observation.shape == (OBSERVATION_SIZE,)
    assert info["public_defender_action"] == 1
    assert info["deception_reward"] == 0.25
    assert info["task_reward"] == 0.0
    assert "belief" not in info
    assert np.isclose(reward, 0.25 - 0.01)


def test_stay_cost_and_blocked_arrival_are_not_counted_as_success():
    env = PPODeceiverEnv(
        FixedEstimator({"A": 0.9, "B": 0.1}),
        step_commit=0,
    )
    env.reset(seed=2, options={"goal": "A"})
    _, stay_reward, _, _, stay_info = env.step(MOVE_STAY)
    assert stay_reward == -0.01
    assert stay_info["movement_cost"] == -0.01

    env.reset(seed=3, options={"goal": "A"})
    actions = [MOVE_UP, MOVE_UP, MOVE_UP, MOVE_LEFT, MOVE_LEFT, MOVE_LEFT]
    for action in actions:
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    assert terminated
    assert info["episode_metrics"]["task_reward"] == -1.0
    assert info["episode_metrics"]["deceiver_success"] == 0
    assert info["episode_metrics"]["reached_goal"] == 1
    assert info["episode_metrics"]["path_efficiency"] == 1.0
    assert info["episode_metrics"]["deception_cost"] == 0
