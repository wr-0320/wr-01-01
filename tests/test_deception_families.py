import numpy as np

from deception_rl.envs.gridworld import MOVE_DOWN, MOVE_RIGHT, MOVE_UP
from experiments.deception_families import (
    DelayedDeception,
    DetourDeception,
    MultiSwitchDeception,
)


def _observation(position, goal=0):
    return {
        "position": np.asarray(position),
        "goal": goal,
        "target_positions": np.asarray([[0, 0], [6, 6]]),
    }


def test_delayed_deception_switches_after_honest_prefix():
    policy = DelayedDeception(delay_steps=2, feint_steps=2)
    policy.reset()

    assert policy.act(_observation([3, 3])) == MOVE_UP
    assert policy.act(_observation([2, 3])) == MOVE_UP
    assert policy.act(_observation([1, 3])) == MOVE_DOWN


def test_detour_deception_takes_lateral_false_side_step():
    policy = DetourDeception(detour_steps=2)
    policy.reset()

    assert policy.act(_observation([3, 3], goal=0)) == MOVE_RIGHT
    assert policy.act(_observation([3, 4], goal=0)) == MOVE_RIGHT
    assert policy.act(_observation([3, 5], goal=0)) == MOVE_UP


def test_multi_switch_deception_alternates_goal_direction_cues():
    policy = MultiSwitchDeception(switches=3)
    policy.reset()
    positions = [[3, 3], [4, 3], [3, 3], [4, 3], [3, 3]]
    actions = [policy.act(_observation(position, goal=0)) for position in positions]

    assert actions == [MOVE_DOWN, MOVE_UP, MOVE_DOWN, MOVE_UP, MOVE_UP]
