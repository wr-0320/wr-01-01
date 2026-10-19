import numpy as np
import pytest
import torch
from models.defender_actor_critic import DefenderActorCritic
from experiments.train_learned_defender import learned_episode
from experiments.train_risk_sensitive_defender import summary
from deception_rl.agents.active_anti_deception import PassiveAntiDeception

class UniformBelief:
    def estimate(self,trajectory,trajectory_mask,target_positions):return {'A':.5,'B':.5}
class CertainBelief:
    def estimate(self,trajectory,trajectory_mask,target_positions):return {'A':1.,'B':0.}

def fixed_model(action):
    model=DefenderActorCritic()
    with torch.no_grad():
        model.actor.weight.zero_();model.actor.bias.fill_(-100);model.actor.bias[action]=100
    return model

def test_entropy_penalty_once_at_first_commitment_and_no_baseline_change():
    model=fixed_model(2)
    plain=learned_episode(model,UniformBelief(),'honest','A',0,0)[0]
    risky=learned_episode(model,UniformBelief(),'honest','A',0,0,entropy_weight=.5)[0]
    assert risky['entropy_before_commitment']==pytest.approx(1.)
    assert risky['entropy_commitment_penalty']==pytest.approx(.5)
    assert risky['episode_return']==pytest.approx(plain['episode_return']-.5)
    assert risky['base_episode_return']==pytest.approx(plain['episode_return'])
    assert risky['defender_task_success']==plain['defender_task_success']==1
    zero=learned_episode(model,CertainBelief(),'honest','A',0,0,entropy_weight=.5)[0]
    assert zero['entropy_commitment_penalty']==0

def test_no_commitment_is_not_zero_entropy_or_correct_commitment():
    row=learned_episode(fixed_model(0),UniformBelief(),'honest','B',0,0,entropy_weight=.5)[0]
    assert row['committed']==0 and row['defender_task_success']==0
    assert row['entropy_before_commitment'] is None and row['entropy_commitment_penalty']==0
    result=summary([row]);assert result['false_commitment_rate'] is None
    assert result['average_entropy_before_commitment'] is None and result['commitment_coverage']==0

def test_entropy_record_is_before_action_for_rule_controller():
    row=learned_episode(None,UniformBelief(),'honest','A',0,0,entropy_weight=.5,rule_controller=PassiveAntiDeception())[0]
    assert row['commit_timestep']==5 and row['entropy_before_commitment']==pytest.approx(1.)
    assert row['observation_cost']==pytest.approx(.05)
    assert row['episode_return']==pytest.approx(.45)

def test_negative_or_nonfinite_penalties_rejected():
    for weight in [-1,np.nan,np.inf]:
        with pytest.raises(ValueError):learned_episode(fixed_model(2),UniformBelief(),'honest','A',0,0,entropy_weight=weight)
