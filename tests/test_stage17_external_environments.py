import numpy as np
from pettingzoo.test import parallel_api_test

from deception_rl.external_envs import (
    CyberDeceptionEnv,
    DefenderAction,
    DefenderEvaluationEnv,
    PettingZooDeceptionAdapter,
    SignalDeceptionParallelEnv,
)
from experiments.run_stage17_external_validation import (
    BeliefThresholdPolicy,
    evaluate,
)


def test_signal_deception_is_valid_parallel_env_and_hides_goal():
    env = SignalDeceptionParallelEnv()
    observations, _ = env.reset(seed=3, options={"goal": "B"})
    assert "goal" not in observations["defender"]
    assert observations["deceiver"]["goal"] == 1
    parallel_api_test(SignalDeceptionParallelEnv(), num_cycles=20)


def test_pettingzoo_adapter_exposes_common_fields_and_cost():
    env = PettingZooDeceptionAdapter("deceptive")
    assert isinstance(env, DefenderEvaluationEnv)
    initial = env.reset(seed=2, options={"goal": "A"})
    assert initial.observation.shape == (8,)
    assert initial.belief == (0.5, 0.5)
    transition = env.step(DefenderAction.TRACK)
    record = transition.as_dict()
    assert {"observation", "belief", "action", "cost", "success"} <= set(record)
    assert transition.cost == 0.03
    assert transition.success is None
    final = env.step(DefenderAction.COMMIT_A)
    assert final.success is True


def test_cyber_simulator_is_seeded_and_tracks_incremental_cost():
    left = CyberDeceptionEnv("credential_decoy")
    right = CyberDeceptionEnv("credential_decoy")
    left_observation = left.reset(seed=7, options={"goal": "B"})
    right_observation = right.reset(seed=7, options={"goal": "B"})
    assert np.array_equal(left_observation.observation, right_observation.observation)
    for action in (DefenderAction.TRACK, DefenderAction.OBSERVE):
        left_step = left.step(action)
        right_step = right.step(action)
        assert np.array_equal(left_step.observation.observation, right_step.observation.observation)
        assert left_step.belief == right_step.belief
    assert left_step.observation.cumulative_cost == 0.08
    final = left.step(DefenderAction.COMMIT_B)
    assert final.success is True
    assert final.observation.committed_goal == 1


def test_unified_evaluator_runs_both_backends_without_training():
    factories = {
        "pettingzoo": lambda: PettingZooDeceptionAdapter("adaptive"),
        "cyber": lambda: CyberDeceptionEnv("lateral_switch"),
    }
    episodes, summary = evaluate(factories, BeliefThresholdPolicy, episodes_per_condition=2)
    assert len(episodes) == 4
    assert len(summary) == 2
    assert {row["environment"] for row in summary} == {
        "pettingzoo_signal_deception",
        "cyber_deception",
    }
    assert all(0 <= row["success_rate"] <= 1 for row in summary)
    assert all(row["trace"][1]["action"] in {"observe", "track", "commit_A", "commit_B"} for row in episodes)
