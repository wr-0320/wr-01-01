import pytest
from experiments.run_ppo_stage73 import CONFIGS, RewardDebugEnv
from deception_rl.envs.gridworld import MOVE_DOWN, MOVE_RIGHT, MOVE_STAY

class FixedEstimator:
    def estimate(self, trajectory, trajectory_mask, target_positions):
        return {'A': 0.8, 'B': 0.2}

@pytest.mark.parametrize('config', CONFIGS.values(), ids=CONFIGS.keys())
def test_reward_timing_success_failure_and_timeout(config):
    env = RewardDebugEnv(FixedEstimator(), config)
    obs, _ = env.reset(seed=0, options={'goal': 'B'})
    assert obs.shape == (12,)
    env.fixed_defender.decision = 0
    total = 0.0
    bonuses = []
    for action in [MOVE_RIGHT] * 3 + [MOVE_DOWN] * 3:
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        bonuses.append(info['deception_reward'])
    assert terminated and not truncated
    assert info['task_reward'] == config[0]
    assert sum(bonuses) == pytest.approx(config[1])
    assert total == pytest.approx(config[0] + config[1] - 0.06)
    if config[2]:
        assert bonuses[:-1] == [0.0] * 5
        assert bonuses[-1] == config[1]
    else:
        assert bonuses[0] == config[1]
    metrics = info['episode_metrics']
    assert metrics['task_reward'] == config[0]
    assert metrics['deception_reward'] == config[1]

    env.reset(seed=0, options={'goal': 'B'})
    env.fixed_defender.decision = 0
    bonuses = []
    for _ in range(30):
        _, _, terminated, truncated, info = env.step(MOVE_STAY)
        bonuses.append(info['deception_reward'])
    assert truncated and not terminated
    assert sum(bonuses) == pytest.approx(0 if config[2] else config[1])
    assert info['episode_metrics']['movement_cost'] == pytest.approx(-0.3)

    env.reset(seed=0, options={'goal': 'B'})
    env.fixed_defender.decision = 1
    for action in [MOVE_RIGHT] * 3 + [MOVE_DOWN] * 3:
        _, _, _, _, info = env.step(action)
    assert info['task_reward'] == -1
    assert info['deception_reward'] == 0
    env.close()

def test_balanced_reward_preserves_base_wrapper_transitions():
    import numpy as np
    from experiments.ppo_deceiver_env import PPODeceiverEnv
    base = PPODeceiverEnv(FixedEstimator(), step_commit=5, belief_observation=True)
    balanced = RewardDebugEnv(FixedEstimator(), CONFIGS['R2_balanced'])
    obs1, _ = base.reset(seed=0, options={'goal': 'B'})
    obs2, _ = balanced.reset(seed=0, options={'goal': 'B'})
    np.testing.assert_array_equal(obs1, obs2)
    for action in [MOVE_STAY] * 6 + [MOVE_RIGHT] * 3 + [MOVE_DOWN] * 3:
        obs1, reward1, term1, trunc1, info1 = base.step(action)
        obs2, reward2, term2, trunc2, info2 = balanced.step(action)
        np.testing.assert_array_equal(obs1, obs2)
        assert (reward1, term1, trunc1) == (reward2, term2, trunc2)
        assert info1 == info2
    base.close()
    balanced.close()
