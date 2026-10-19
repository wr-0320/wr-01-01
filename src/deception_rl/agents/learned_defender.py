"""Deploy a trained four-action Defender without exposing hidden state."""
import torch
from deception_rl.envs.gridworld import OBSERVE,TRACK,GUARD_A,GUARD_B
from models.defender_actor_critic import encode_state,gate_commit_logits

ENV_ACTIONS=(OBSERVE,TRACK,GUARD_A,GUARD_B)

class LearnedActiveDefender:
    def __init__(self,model,estimator,deterministic=True,commitment_entropy_threshold=None):
        self.model=model;self.estimator=estimator;self.deterministic=deterministic;self.commitment_entropy_threshold=commitment_entropy_threshold;self.reset()
    def reset(self):self.goal=None;self.commit_timestep=None
    def act(self,observation,timestep):
        if self.goal is not None:return self.goal
        with torch.inference_mode():
            history,scalars=encode_state(observation,timestep,self.estimator)
            logits,_=self.model(history,scalars)
            logits=gate_commit_logits(logits,scalars[:,3],self.commitment_entropy_threshold)
            action=int(logits.argmax(1)) if self.deterministic else int(torch.distributions.Categorical(logits=logits).sample())
        if action>=2:self.goal=action-2;self.commit_timestep=timestep
        return ENV_ACTIONS[action]

class UncertaintyGatedDefender(LearnedActiveDefender):
    def __init__(self,model,estimator,threshold=.75,deterministic=True):
        super().__init__(model,estimator,deterministic,commitment_entropy_threshold=threshold)
