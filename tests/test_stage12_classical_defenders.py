from collections import Counter

import numpy as np

from deception_rl.envs.gridworld import GUARD_A, GUARD_B, OBSERVE, TRACK
from experiments.run_stage12_bayesian_pomdp import schedule
from experiments.stage12_classical_defenders import (
    BayesianDefender,
    FiniteHorizonPOMDPDefender,
    TrajectoryBayesFilter,
)


def observation(points):
    trajectory = np.full((31, 2), -1, dtype=np.int64)
    mask = np.zeros(31, dtype=np.int8)
    for timestep, row, column in points:
        trajectory[timestep] = [row, column]
        mask[timestep] = 1
    return {
        "trajectory": trajectory,
        "trajectory_mask": mask,
        "target_positions": np.asarray([[0, 0], [6, 6]]),
    }


def test_bayes_update_favors_goal_consistent_motion_and_no_double_counting():
    filter_ = TrajectoryBayesFilter()
    toward_a = observation([(0, 3, 3), (1, 2, 3), (2, 1, 3)])
    first = filter_.update(toward_a, 2)
    assert first[0] > 0.9
    assert np.allclose(filter_.update(toward_a, 3), first)
    assert filter_.entropy < 0.5


def test_bayes_filter_marginalizes_timed_visibility_gap():
    filter_ = TrajectoryBayesFilter()
    belief = filter_.update(observation([(0, 3, 3), (3, 0, 3)]), 3)
    assert belief[0] > belief[1]
    assert filter_.processed_times == {0, 3}


def test_bayesian_defender_has_only_observe_or_irreversible_commit():
    defender = BayesianDefender(confidence_threshold=0.8)
    assert defender.act(observation([(0, 3, 3)]), 0) == OBSERVE
    action = defender.act(observation([(0, 3, 3), (1, 2, 3), (2, 1, 3)]), 5)
    assert action == GUARD_A
    contrary = observation([(0, 3, 3), (1, 4, 3), (2, 5, 3)])
    assert defender.act(contrary, 6) == GUARD_A


def test_pomdp_plans_sensing_for_uncertainty_and_commit_for_confidence():
    defender = FiniteHorizonPOMDPDefender()
    uncertain = defender.action_values(0.5, 3)
    assert max(uncertain, key=uncertain.get) == TRACK
    confident = defender.action_values(0.95, 3)
    assert max(confident, key=confident.get) == GUARD_A
    assert uncertain[TRACK] > uncertain[OBSERVE]


def test_pomdp_deadline_commits_and_state_reports_belief_entropy_status():
    defender = FiniteHorizonPOMDPDefender(decision_deadline=2)
    obs = observation([(0, 3, 3)])
    assert defender.act(obs, 0) in (OBSERVE, TRACK)
    assert defender.act(obs, 2) in (GUARD_A, GUARD_B)
    assert set(("goal_belief", "entropy", "pomdp_horizon", "action_values")) <= set(defender.info)
    committed = defender.goal
    assert defender.act(obs, 3) == committed


def test_stage12_fixed_budget_and_balance():
    main, audits = schedule(0)
    assert len(main) * 5 + len(audits) == 512
    assert Counter(row["attack"] for row in main) == {
        "honest": 34,
        "scripted": 34,
        "adaptive_attacker": 34,
    }
    for attack in ("honest", "scripted", "adaptive_attacker"):
        assert Counter(row["goal"] for row in main if row["attack"] == attack) == {
            "A": 17,
            "B": 17,
        }
    assert {row["controller"] for row in audits} == {"bayesian", "pomdp"}

