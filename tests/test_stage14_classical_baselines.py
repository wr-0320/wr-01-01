import numpy as np

from deception_rl.envs import ForkedRoutesAdapter, GridWorldAdapter
from deception_rl.envs.gridworld import GUARD_A, GUARD_B, OBSERVE, TRACK
from experiments.stage14_classical_baselines import (
    FormalBayesianDefender,
    FormalFiniteHorizonPOMDPDefender,
    FormalTrajectoryBayesFilter,
)


def observation(points, targets=((0, 0), (6, 6)), length=31):
    trajectory = np.full((length, 2), -1, dtype=np.int64)
    mask = np.zeros(length, dtype=np.int8)
    for timestep, row, column in points:
        trajectory[timestep] = [row, column]
        mask[timestep] = 1
    return {
        "trajectory": trajectory,
        "trajectory_mask": mask,
        "target_positions": np.asarray(targets),
    }


def test_formal_filter_is_normalized_symmetric_and_does_not_double_count():
    filter_ = FormalTrajectoryBayesFilter(GridWorldAdapter().model)
    initial = filter_.update(observation([(0, 3, 3)]), 0)
    assert np.allclose(initial, [0.5, 0.5])
    toward_a = observation([(0, 3, 3), (1, 2, 3), (2, 1, 3)])
    belief = filter_.update(toward_a, 2)
    assert belief[0] > 0.9
    assert np.isclose(belief.sum(), 1.0)
    assert np.allclose(filter_.update(toward_a, 3), belief)


def test_formal_filter_uses_second_environment_transition_model():
    adapter = ForkedRoutesAdapter()
    filter_ = FormalTrajectoryBayesFilter(adapter.model)
    points = observation(
        [(0, 6, 3), (4, 2, 3), (6, 2, 1)],
        targets=adapter.model.goals,
    )
    belief = filter_.update(points, 6)
    assert belief[0] > belief[1]
    assert np.isclose(belief.sum(), 1.0)


def test_formal_bayesian_commitment_is_irreversible():
    defender = FormalBayesianDefender()
    assert defender.act(observation([(0, 3, 3)]), 0) == OBSERVE
    action = defender.act(observation([(0, 3, 3), (1, 2, 3), (2, 1, 3)]), 5)
    assert action == GUARD_A
    assert defender.act(observation([(0, 3, 3), (1, 4, 3)]), 6) == GUARD_A


def test_formal_pomdp_values_information_and_respects_deadline():
    defender = FormalFiniteHorizonPOMDPDefender()
    uncertain = defender.action_values(0.5, defender.horizon)
    assert max(uncertain, key=uncertain.get) in (OBSERVE, TRACK)
    confident = defender.action_values(0.99, defender.horizon)
    assert max(confident, key=confident.get) == GUARD_A
    deadline = FormalFiniteHorizonPOMDPDefender(decision_deadline=1)
    obs = observation([(0, 3, 3)])
    deadline.act(obs, 0)
    assert deadline.act(obs, 1) in (GUARD_A, GUARD_B)

