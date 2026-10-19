import copy
import numpy as np
import pytest
import torch
from models.defender_actor_critic import DefenderActorCritic,encode_state
from deception_rl.agents.learned_defender import LearnedActiveDefender,ENV_ACTIONS
from deception_rl.envs.gridworld import OBSERVE,TRACK,GUARD_A,GUARD_B
from experiments.train_learned_defender import discounted_returns,learned_episode

class VisibleEstimator:
    def estimate(self,trajectory,trajectory_mask,target_positions):
        assert np.all(trajectory[~trajectory_mask]==-1)
        return {'A':.7,'B':.3}

def observation():
    points=np.full((31,2),-1,dtype=np.int64);points[0]=[3,3];points[1]=[2,3]
    mask=np.zeros(31,dtype=np.int8);mask[:2]=1
    return dict(trajectory=points,trajectory_mask=mask,target_positions=np.array([[0,0],[6,6]]))

def test_state_uses_requested_signals_and_rejects_hidden_future_inputs():
    o=observation();other=copy.deepcopy(o);other['trajectory'][2:]=6;other['trajectory_mask'][20]=1;other['true_goal']=1;other['family']='scripted_5'
    a=encode_state(o,2,VisibleEstimator());b=encode_state(other,2,VisibleEstimator())
    for x,y in zip(a,b):torch.testing.assert_close(x,y)
    history,scalars=a;assert history.shape==(1,3,6) and scalars.shape==(1,4)
    assert history[0,2,1]==0 and torch.all(history[0,2,2:]==0)
    assert scalars[0,0]==pytest.approx(.7) and scalars[0,2]==pytest.approx(.1)
    model=DefenderActorCritic();logits,value=model(history,scalars)
    assert logits.shape==(1,4) and value.shape==(1,)

def test_commitment_cannot_be_reversed_by_policy_logits():
    model=DefenderActorCritic()
    with torch.no_grad():model.actor.weight.zero_();model.actor.bias.copy_(torch.tensor([0.,0.,10.,0.]))
    controller=LearnedActiveDefender(model,VisibleEstimator())
    assert controller.act(observation(),1)==GUARD_A
    with torch.no_grad():model.actor.bias.copy_(torch.tensor([0.,0.,0.,10.]))
    assert controller.act(observation(),2)==GUARD_A
    controller.reset();assert controller.act(observation(),2)==GUARD_B
    assert ENV_ACTIONS==(OBSERVE,TRACK,GUARD_A,GUARD_B)

def test_delayed_interception_returns_reach_commit_decision():
    returns=discounted_returns([-.03,0,0,1],gamma=.9)
    assert returns[0]==pytest.approx(-.03+.9**3)
    assert returns[1]==pytest.approx(.9**2)

def test_learning_episode_produces_actor_and_history_gradients():
    model=DefenderActorCritic()
    with torch.no_grad():model.actor.bias.copy_(torch.tensor([0.,0.,-10.,-10.]))
    row,decisions,rewards=learned_episode(model,VisibleEstimator(),'honest','A',0,0,training=True)
    assert decisions and sum(rewards)==pytest.approx(row['episode_return'])
    loss=-torch.stack([d[1] for d in decisions]).mean()+torch.stack([d[2] for d in decisions]).square().mean()
    loss.backward()
    assert model.actor.weight.grad is not None and torch.isfinite(model.actor.weight.grad).all()
    assert model.history_gru.weight_ih_l0.grad is not None
