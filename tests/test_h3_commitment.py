import csv
import json

from deception_rl.agents import BeliefBasedDefender, CommittedDefender
from deception_rl.beliefs import TrajectoryBeliefEstimator
from deception_rl.envs.gridworld import GUARD_A, MOVE_UP, TRACK, GridWorldParallelEnv
from experiments.run_h3_commitment import run_experiment


def test_committed_defender_locks_decision_after_commitment():
    env = GridWorldParallelEnv()
    observations, _ = env.reset(seed=0, options={"goal": "A"})
    defender = CommittedDefender(
        step_commit=2,
        defender=BeliefBasedDefender(estimator=TrajectoryBeliefEstimator()),
    )

    assert defender.act(observations["defender"]) == TRACK
    observations, *_ = env.step({"deceiver": MOVE_UP, "defender": TRACK})
    assert defender.act(observations["defender"]) != GUARD_A
    observations, *_ = env.step({"deceiver": MOVE_UP, "defender": TRACK})
    assert defender.act(observations["defender"]) == GUARD_A
    locked_belief = defender.belief_at_commitment.copy()

    assert defender.act(observations["defender"]) == GUARD_A
    assert defender.belief_at_commitment == locked_belief


def test_h3_experiment_records_wrong_commitment_and_deception_cost(tmp_path):
    output_path = run_experiment(
        episodes_per_policy=2,
        step_commit=2,
        seed=5,
        output_path=tmp_path / "h3.csv",
        estimator=TrajectoryBeliefEstimator(),
    )

    with output_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 4
    honest_rows = [
        row for row in rows if row["deceiver_policy"] == "honest_shortest_path"
    ]
    deceptive_rows = [
        row for row in rows if row["deceiver_policy"] == "scripted_deception"
    ]
    assert all(row["decision_correct"] == "1" for row in honest_rows)
    assert all(row["decision_correct"] == "0" for row in deceptive_rows)
    assert all(row["deceiver_success"] == "1" for row in deceptive_rows)
    assert all(row["deception_cost"] == "4" for row in rows)
    for row in deceptive_rows:
        belief = json.loads(row["belief_at_commitment"])
        false_goal = "B" if row["true_goal"] == "A" else "A"
        assert belief[false_goal] > belief[row["true_goal"]]
