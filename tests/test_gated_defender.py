import numpy as np
import pytest
import torch
from models.defender_actor_critic import DefenderActorCritic,gate_commit_logits
from deception_rl.agents.learned_defender import UncertaintyGatedDefender
from experiments.train_learned_defender import learned_episode
from deception_rl.envs.gridworld import OBSERVE,TRACK

class Belief:
    def __init__(self,certain=False):self.certain=certain
    def estimate(self,trajectory,trajectory_mask,target_positions):return {'A':1.,'B':0.} if self.certain else {'A':.5,'B':.5}

def obs():
    p=np.full((31,2),-1,dtype=np.int64);p[0]=[3,3];m=np.zeros(31,dtype=np.int8);m[0]=1
    return dict(trajectory=p,trajectory_mask=m,target_positions=np.array([[0,0],[6,6]]))

def commit_model():
    model=DefenderActorCritic()
    with torch.no_grad():model.actor.weight.zero_();model.actor.bias.copy_(torch.tensor([0.,0.,100.,0.]))
    return model

def test_gate_exact_boundary_and_sampling_gradient():
    logits=torch.tensor([[0.,0.,10.,10.],[0.,0.,10.,10.]],requires_grad=True)
    gated=gate_commit_logits(logits,torch.tensor([.75001,.75]),.75)
    distribution=torch.distributions.Categorical(logits=gated)
    assert torch.all(distribution.probs[0,2:]==0)
    assert torch.all(distribution.probs[1,2:]>0)
    assert torch.equal(gated[1],logits[1])
    assert torch.isfinite(distribution.entropy()).all()
    loss=-distribution.log_prob(torch.tensor([0,2])).mean()-.01*distribution.entropy().mean();loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.all(logits.grad[0,2:]==0)
    assert gate_commit_logits(logits,.9,None) is logits

def test_gate_shared_by_deployment_and_episode_evaluation():
    model=commit_model()
    high=UncertaintyGatedDefender(model,Belief(),.75)
    assert high.act(obs(),0) in (OBSERVE,TRACK) and high.goal is None
    certain=UncertaintyGatedDefender(model,Belief(True),.75)
    assert certain.act(obs(),0)==0
    certain.estimator=Belief();assert certain.act(obs(),1)==0  # Existing commitment stays irreversible.
    row=learned_episode(model,Belief(),'honest','A',0,0,commitment_entropy_threshold=.75)[0]
    assert row['committed']==0 and row['entropy_before_commitment'] is None
    assert row['gate_blocked_decisions']==row['steps'] and row['gate_allowed_decisions']==0
    assert row['entropy_commitment_penalty']==0 and row['episode_return']==row['base_episode_return']

def test_gate_does_not_force_commitment_when_allowed():
    model=commit_model()
    with torch.no_grad():model.actor.bias.copy_(torch.tensor([100.,0.,0.,0.]))
    row=learned_episode(model,Belief(True),'honest','A',0,0,commitment_entropy_threshold=.75)[0]
    assert row['committed']==0 and row['gate_allowed_decisions']==row['steps']
    assert row['refrained_when_gate_allowed']==row['steps']

@pytest.mark.parametrize('threshold',[-.1,1.1,np.nan,np.inf])
def test_bad_threshold_rejected(threshold):
    with pytest.raises(ValueError):gate_commit_logits(torch.zeros(1,4),.5,threshold)
