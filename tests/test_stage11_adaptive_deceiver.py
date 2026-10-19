from collections import Counter

import numpy as np

from deception_rl.envs.gridworld import (
    GUARD_A,
    MOVE_DOWN,
    MOVE_UP,
    OBSERVE,
    TRACK,
)
from experiments.run_stage11_adaptive_attacker import scenario_schedule
from experiments.stage11_adaptive_deceiver import AdaptiveDeceiver


def observation(position=(3, 3), goal=0):
    return {
        "position": np.asarray(position),
        "goal": goal,
        "target_positions": np.asarray([[0, 0], [6, 6]]),
    }


def test_commitment_aware_rule_moves_honestly_then_feints_before_expected_commit():
    attacker = AdaptiveDeceiver(0)
    actions = [attacker.act(observation((3 - index, 3))) for index in range(3)]
    assert actions == [MOVE_UP, MOVE_UP, MOVE_DOWN]
    assert attacker.last_rule == "commitment_aware"


def test_target_profile_persists_across_episode_resets():
    attacker = AdaptiveDeceiver(0)
    for timestep in (4, 6):
        attacker.reset(timestep)
        attacker.observe_defender(GUARD_A, timestep, True)
    attacker.reset(9)
    assert attacker.preferred_fake_target == 1
    assert attacker.expected_commit_step == 5
    assert attacker.act(observation(goal=0)) == MOVE_DOWN
    assert attacker.last_rule == "target_aware"


def test_observation_rule_uses_only_explicitly_completed_history():
    attacker = AdaptiveDeceiver(0)
    first = attacker.act(observation())
    assert first == MOVE_UP
    attacker.observe_defender(TRACK, 0, True)
    second = attacker.act(observation((2, 3)))
    assert second == MOVE_UP
    assert attacker.last_rule == "observation_aware_track"
    attacker.observe_defender(OBSERVE, 1, True)
    third = attacker.act(observation((1, 3)))
    assert third == MOVE_DOWN
    assert attacker.last_rule == "observation_aware_observe"


def test_profile_reset_is_explicit_and_complete():
    attacker = AdaptiveDeceiver(0)
    attacker.observe_defender(GUARD_A, 8, False)
    attacker.reset(1)
    assert attacker.commit_times == [8]
    assert attacker.behavior.actions == []
    attacker.reset_profile(2)
    assert attacker.commit_times == []
    assert attacker.committed_targets.tolist() == [0, 0]


def test_stage11_budget_and_main_balance():
    main, supplementary = scenario_schedule(0)
    assert len(main) * 3 + len(supplementary) == 512
    assert Counter(item["attack"] for item in main) == {
        "honest": 56,
        "scripted": 56,
        "adaptive": 56,
    }
    for attack in ("honest", "scripted", "adaptive"):
        assert Counter(item["goal"] for item in main if item["attack"] == attack) == {
            "A": 28,
            "B": 28,
        }
    assert Counter(item["goal"] for item in supplementary) == {"A": 4, "B": 4}

