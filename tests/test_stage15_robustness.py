import numpy as np

from deception_rl.envs import ForkedRoutesAdapter
from experiments.build_stage15_validation import mean_std
from experiments.run_stage15_robustness import evaluate_episode
from experiments.stage15_attackers import AdaptiveDeceiverV2, RouteAwareFamily


class VisibleHistoryController:
    def reset(self):
        self.goal = None
        self.commit_timestep = None
        self.info = None

    def act(self, observation, timestep):
        assert np.asarray(observation["trajectory_mask"], dtype=bool).any()
        self.goal = 0
        self.commit_timestep = timestep
        return 0


def test_route_aware_honest_policy_follows_legal_fork():
    model = ForkedRoutesAdapter().model
    policy = RouteAwareFamily("honest", model)
    policy.reset(seed=0)
    position = model.start
    observation = {
        "position": np.asarray(position),
        "goal": 0,
        "target_positions": np.asarray(model.goals),
    }
    for _ in range(model.shortest_distance(model.start, model.goals[0])):
        action = policy.act(observation)
        successor = model.transition(position, action)
        assert successor in model.traversable
        position = successor
        observation["position"] = np.asarray(position)
    assert position == model.goals[0]


def test_adaptive_v2_uses_only_recorded_completed_commitment():
    model = ForkedRoutesAdapter().model
    attacker = AdaptiveDeceiverV2(model, seed=0)
    observation = {
        "position": np.asarray(model.start),
        "goal": 0,
        "target_positions": np.asarray(model.goals),
    }
    attacker.observe_defender(1, timestep=4, attacker_was_visible=True)
    attacker.act(observation)
    assert attacker.last_rule == "post_commit_reversal"
    assert attacker.profile()["commit_times"] == [4]


def test_forked_runner_bootstraps_empty_visible_history_with_charged_track():
    adapter = ForkedRoutesAdapter()
    row = evaluate_episode(
        adapter,
        VisibleHistoryController(),
        RouteAwareFamily("honest", adapter.model),
        environment="forked_routes",
        controller_name="test",
        family="honest",
        evaluation_seed=0,
        episode_index=0,
        goal="A",
        simulator_seed=0,
    )
    assert row["defender_action_counts"][3] >= 1
    assert row["observation_cost"] >= 0.03


def test_mean_std_uses_sample_standard_deviation():
    mean, std = mean_std([0.0, 1.0, 2.0])
    assert mean == 1.0
    assert std == 1.0
