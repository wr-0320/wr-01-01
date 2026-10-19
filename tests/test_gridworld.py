import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from deception_rl.envs.gridworld import (
    GUARD_A,
    GUARD_B,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
    OBSERVE,
    TRACK,
    GridWorldParallelEnv,
)


def test_reset_is_seeded_and_hides_goal_from_defender():
    env = GridWorldParallelEnv()
    first_observations, _ = env.reset(seed=7)
    first_goal = first_observations["deceiver"]["goal"]
    defender_observation = first_observations["defender"]

    assert first_observations["deceiver"]["goal"] == first_goal
    assert "goal" not in defender_observation
    assert defender_observation["trajectory_mask"][0] == 1
    assert env.observation_space("defender").contains(defender_observation)

    repeated_observations, _ = env.reset(seed=7)
    assert repeated_observations["deceiver"]["goal"] == first_goal


def test_unseen_positions_are_not_added_to_observed_trajectory():
    env = GridWorldParallelEnv()
    env.reset(seed=0, options={"goal": "A"})
    observations, *_ = env.step({"deceiver": MOVE_UP, "defender": GUARD_A})
    observations, *_ = env.step({"deceiver": MOVE_UP, "defender": GUARD_A})

    defender_observation = observations["defender"]
    assert defender_observation["trajectory_mask"][:3].tolist() == [1, 1, 0]
    np.testing.assert_array_equal(defender_observation["trajectory"][2], [-1, -1])
    assert not np.any(defender_observation["local_map"] == 4)


def test_observe_and_track_expand_trajectory_visibility():
    env = GridWorldParallelEnv()
    env.reset(seed=0, options={"goal": "B"})
    env.step({"deceiver": MOVE_UP, "defender": GUARD_A})
    observed, *_ = env.step({"deceiver": MOVE_UP, "defender": OBSERVE})
    tracked, *_ = env.step({"deceiver": MOVE_UP, "defender": TRACK})

    assert observed["defender"]["trajectory_mask"][2] == 1
    assert tracked["defender"]["trajectory_mask"][3] == 1


@pytest.mark.parametrize(
    ("goal", "defender_action", "defender_reward"),
    [("A", GUARD_A, 1.0), ("B", GUARD_B, 1.0)],
)
def test_correct_guard_blocks_arrival(goal, defender_action, defender_reward):
    env = GridWorldParallelEnv()
    env.reset(seed=0, options={"goal": goal})
    if goal == "A":
        destination_actions = [
            MOVE_UP,
            MOVE_UP,
            MOVE_UP,
            MOVE_LEFT,
            MOVE_LEFT,
            MOVE_LEFT,
        ]
    else:
        destination_actions = [
            MOVE_DOWN,
            MOVE_DOWN,
            MOVE_DOWN,
            MOVE_RIGHT,
            MOVE_RIGHT,
            MOVE_RIGHT,
        ]

    for action in destination_actions[:-1]:
        observations, *_ = env.step({"deceiver": action, "defender": defender_action})
    result = env.step(
        {"deceiver": destination_actions[-1], "defender": defender_action}
    )

    _, rewards, terminations, truncations, _ = result
    assert rewards["defender"] == defender_reward
    assert all(terminations.values())
    assert not any(truncations.values())


def test_parallel_api_contract():
    parallel_api_test(GridWorldParallelEnv(), num_cycles=30)
