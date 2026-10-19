import numpy as np

from deception_rl.envs import (
    ForkedRoutesAdapter,
    ForkedRoutesParallelEnv,
    GridWorldAdapter,
    validate_environment_interface,
)
from deception_rl.envs.gridworld import GUARD_A, GUARD_B, TRACK


def test_both_environment_adapters_satisfy_shared_contract():
    rows = [
        validate_environment_interface(GridWorldAdapter()),
        validate_environment_interface(ForkedRoutesAdapter()),
    ]
    assert [row["environment"] for row in rows] == [
        "deception_gridworld_v0",
        "forked_routes_v0",
    ]
    assert all(row["shortest_path_A"] == row["shortest_path_B"] for row in rows)


def test_forked_routes_blocks_walls_and_keeps_timed_observation_contract():
    env = ForkedRoutesParallelEnv()
    observations, _ = env.reset(seed=3, options={"goal": "A"})
    start = observations["deceiver"]["position"].copy()
    observations, _, _, _, _ = env.step({"deceiver": 2, "defender": TRACK})
    assert np.array_equal(observations["deceiver"]["position"], start)
    assert observations["defender"]["trajectory_mask"][1] == 1
    assert env.observation_space("defender").contains(observations["defender"])


def test_forked_routes_goals_are_reachable_and_guard_actions_intercept():
    routes = {
        "A": [0, 0, 0, 0, 2, 2, 0, 0],
        "B": [0, 0, 0, 0, 3, 3, 0, 0],
    }
    guards = {"A": GUARD_A, "B": GUARD_B}
    for goal, actions in routes.items():
        env = ForkedRoutesParallelEnv()
        env.reset(seed=0, options={"goal": goal})
        final_reward = 0.0
        for action in actions:
            _, rewards, _, _, _ = env.step(
                {"deceiver": action, "defender": guards[goal]}
            )
            final_reward = rewards["defender"]
        assert final_reward == 1.0
        assert not env.agents


def test_forked_routes_resizes_timed_buffers_when_cycle_budget_changes():
    env = ForkedRoutesParallelEnv(max_cycles=10)
    env.max_cycles = 12
    observations, _ = env.reset(seed=0)
    assert observations["defender"]["trajectory"].shape == (13, 2)
    assert env.observation_space("defender").contains(observations["defender"])

