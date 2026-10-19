import csv

import numpy as np
import torch

from deception_rl.external_baselines import (
    ActiveInferenceDefender,
    BayesianDefenderAdapter,
    MAPPOActorCritic,
    MAPPODefenderPolicy,
    POMCPBeliefPlanner,
)
from deception_rl.external_envs import (
    DefenderAction,
    DefenderObservation,
    ForkedRoutesDefenderAdapter,
    GridWorldDefenderAdapter,
)
from experiments.run_stage17_external_baselines import (
    CSV_COLUMNS,
    EnvironmentSpec,
    evaluate_suite,
    train_mappo_smoke,
)


def observation(belief=(0.5, 0.5), step=0, committed_goal=None):
    if committed_goal is None:
        mask = (True, True, True, True)
    else:
        mask = tuple(index == 2 + committed_goal for index in range(4))
    return DefenderObservation(
        observation=np.zeros(8, dtype=np.float32),
        belief=belief,
        action_mask=mask,
        step=step,
        cumulative_cost=0.0,
        committed_goal=committed_goal,
        source="test",
    )


def test_navigation_adapters_keep_common_actions_and_contract():
    for adapter in (GridWorldDefenderAdapter, ForkedRoutesDefenderAdapter):
        env = adapter("scripted_3")
        current = env.reset(seed=4, options={"goal": "A"})
        assert current.observation.shape == (8,)
        assert current.action_mask == (True, True, True, True)
        transition = env.step(DefenderAction.TRACK)
        assert transition.action == DefenderAction.TRACK
        assert transition.cost == 0.03
        assert np.isclose(sum(transition.belief), 1)
        env.close()


def test_classical_baselines_return_only_available_common_actions():
    policies = (
        BayesianDefenderAdapter(),
        POMCPBeliefPlanner(simulations=8, horizon=2, seed=3),
        ActiveInferenceDefender(),
    )
    for policy in policies:
        assert policy.act(observation()) in tuple(DefenderAction)
        assert policy.act(observation((0.99, 0.01), step=8)) == DefenderAction.COMMIT_A
        assert policy.act(observation((0.01, 0.99), step=8)) == DefenderAction.COMMIT_B
        assert policy.act(observation(committed_goal=1)) == DefenderAction.COMMIT_B


def test_mappo_actor_masks_actions_and_uses_central_context():
    model = MAPPOActorCritic(hidden_size=8)
    policy = MAPPODefenderPolicy(model, context_index=2)
    action = policy.act(observation(committed_goal=0))
    assert action == DefenderAction.COMMIT_A
    features, context = policy.tensors(observation())
    logits, value = model(features.unsqueeze(0), context.unsqueeze(0))
    assert logits.shape == (1, 4)
    assert value.shape == (1,)
    assert context.tolist() == [0.0, 0.0, 1.0, 0.0]


def test_smoke_training_and_unified_csv_schema(tmp_path):
    specs = [
        EnvironmentSpec(
            "gridworld", "honest", 0, lambda: GridWorldDefenderAdapter("honest")
        )
    ]
    model, logs = train_mappo_smoke(
        specs, tmp_path, seed=2, episodes_per_condition=2, epochs=1
    )
    episodes, summary = evaluate_suite(
        specs, model, episodes_per_condition=2, seed=2
    )
    assert len(logs) == 1
    assert len(episodes) == 8
    assert len(summary) == 4
    assert all(tuple(row) == CSV_COLUMNS for row in summary)
    assert (tmp_path / "mappo_smoke.pt").exists()
    with (tmp_path / "mappo_training.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1
    assert all(np.isfinite(row["regret"]) for row in summary)
    assert all(0 <= row["calibration"] <= 1 for row in summary)
