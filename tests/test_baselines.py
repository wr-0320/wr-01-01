import csv
import json

import numpy as np

from deception_rl.agents import (
    BeliefBasedDefender,
    HonestShortestPath,
    RandomPolicy,
    ScriptedDeception,
)
from deception_rl.beliefs import TrajectoryBeliefEstimator
from deception_rl.envs.gridworld import (
    GUARD_A,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_UP,
    OBSERVE,
    TRACK,
    GridWorldParallelEnv,
)
from experiments.run_baseline import run_experiment


def test_honest_shortest_path_moves_toward_true_goal():
    env = GridWorldParallelEnv()
    observations, _ = env.reset(seed=0, options={"goal": "A"})
    policy = HonestShortestPath()
    start_distance = np.abs(
        observations["deceiver"]["position"]
        - observations["deceiver"]["target_positions"][0]
    ).sum()

    action = policy.act(observations["deceiver"])
    next_observation, *_ = env.step({"deceiver": action, "defender": GUARD_A})
    next_distance = np.abs(
        next_observation["deceiver"]["position"]
        - next_observation["deceiver"]["target_positions"][0]
    ).sum()
    assert next_distance == start_distance - 1


def test_random_policy_actions_are_valid():
    env = GridWorldParallelEnv()
    observation, _ = env.reset(seed=0)
    policy = RandomPolicy(seed=3)

    assert all(
        env.action_space("deceiver").contains(policy.act(observation["deceiver"]))
        for _ in range(20)
    )


def test_scripted_deception_approaches_decoy_before_true_goal():
    env = GridWorldParallelEnv()
    observations, _ = env.reset(seed=0, options={"goal": "B"})
    policy = ScriptedDeception(deception_steps=2)

    assert policy.act(observations["deceiver"]) == MOVE_UP
    observations, *_ = env.step({"deceiver": MOVE_UP, "defender": TRACK})
    assert policy.act(observations["deceiver"]) == MOVE_UP
    observations, *_ = env.step({"deceiver": MOVE_UP, "defender": TRACK})
    assert policy.act(observations["deceiver"]) == MOVE_DOWN


def test_trajectory_belief_responds_to_observed_direction():
    estimator = TrajectoryBeliefEstimator()
    targets = np.array([[0, 0], [6, 6]])
    mask = np.array([1, 1])
    toward_a = estimator.estimate(np.array([[3, 3], [2, 3]]), mask, targets)
    toward_b = estimator.estimate(np.array([[3, 3], [4, 3]]), mask, targets)

    assert toward_a["A"] > toward_a["B"]
    assert toward_b["B"] > toward_b["A"]
    assert np.isclose(toward_a["A"] + toward_a["B"], 1.0)


def test_belief_defender_uses_sensing_when_uncertain_and_guards_when_confident():
    env = GridWorldParallelEnv()
    observations, _ = env.reset(seed=0, options={"goal": "A"})
    defender = BeliefBasedDefender(estimator=TrajectoryBeliefEstimator())

    assert defender.act(observations["defender"]) == TRACK
    observations, *_ = env.step({"deceiver": MOVE_LEFT, "defender": TRACK})
    assert defender.act(observations["defender"]) == OBSERVE
    observations, *_ = env.step({"deceiver": MOVE_LEFT, "defender": OBSERVE})
    assert defender.act(observations["defender"]) == GUARD_A


def test_experiment_writes_required_csv_records(tmp_path):
    output_path = run_experiment(
        episodes_per_policy=2,
        seed=11,
        output_path=tmp_path / "results.csv",
        estimator=TrajectoryBeliefEstimator(),
    )

    with output_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 6
    assert {
        "true_goal",
        "trajectory",
        "estimated_belief",
        "belief_trace",
        "defender_action",
        "success_rate",
    }.issubset(rows[0])
    assert {row["deceiver_policy"] for row in rows} == {
        "honest_shortest_path",
        "random_policy",
        "scripted_deception",
    }
    assert json.loads(rows[0]["estimated_belief"]).keys() == {"A", "B"}
    assert isinstance(json.loads(rows[0]["belief_trace"]), list)
