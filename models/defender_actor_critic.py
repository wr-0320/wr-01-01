"""Small learned Defender: public-history GRU plus belief state actor/critic."""
import numpy as np
import torch
from torch import nn
from deception_rl.agents.active_anti_deception import trajectory_evidence

ACTION_NAMES=('observe','track','commit_A','commit_B')

def encode_state(observation,timestep,estimator):
    if not 0<=timestep<=30:raise ValueError('Public timestep outside episode horizon')
    mask=np.asarray(observation['trajectory_mask'],dtype=bool).copy()
    mask[timestep+1:]=False
    positions=np.asarray(observation['trajectory'])
    history=np.zeros((timestep+1,6),dtype=np.float32);history[:,0]=np.arange(timestep+1)/30
    # Hidden coordinates are never read, even if a caller fills masked entries.
    last=None
    for t in range(timestep+1):
        history[t,1]=float(mask[t])
        if mask[t]:
            point=positions[t].astype(np.float32)/6;history[t,2:4]=point
            if last is not None:history[t,4:6]=point-last
            last=point
    # Provide a sanitized masked trajectory to the frozen estimator as well.
    safe=np.full_like(positions,-1);safe[mask]=positions[mask]
    belief=estimator.estimate(safe,mask,observation['target_positions'])
    probs=np.array([belief['A'],belief['B']],dtype=np.float32)
    if not np.isfinite(probs).all() or np.any(probs<0) or probs.sum()<=0:raise ValueError('Invalid goal belief')
    probs/=probs.sum()
    entropy=float(-(probs*np.log2(np.clip(probs,1e-8,1))).sum())
    visible_observation=dict(observation,trajectory=safe,trajectory_mask=mask)
    deception=trajectory_evidence(visible_observation,timestep).deception_probability
    scalars=np.array([probs[0],probs[1],deception,entropy],dtype=np.float32)
    return torch.from_numpy(history).unsqueeze(0),torch.from_numpy(scalars).unsqueeze(0)

class DefenderActorCritic(nn.Module):
    def __init__(self,history_size=16):
        super().__init__();self.history_size=history_size
        self.history_gru=nn.GRU(6,history_size,batch_first=True)
        self.trunk=nn.Sequential(nn.Linear(history_size+4,32),nn.Tanh())
        self.actor=nn.Linear(32,4);self.critic=nn.Linear(32,1)
    def forward(self,history,scalars):
        _,hidden=self.history_gru(history);state=self.trunk(torch.cat((hidden[-1],scalars),dim=1))
        return self.actor(state),self.critic(state).squeeze(-1)

def gate_commit_logits(logits,goal_entropy,threshold=None):
    """Mask commit_A/B only when normalized goal uncertainty exceeds threshold."""
    if threshold is None:return logits
    if not np.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Entropy threshold must be in [0,1]')
    entropy=torch.as_tensor(goal_entropy,dtype=logits.dtype,device=logits.device)
    if not torch.isfinite(entropy).all() or (entropy<0).any() or (entropy>1+1e-6).any():raise ValueError('Invalid goal entropy')
    if entropy.ndim==0:entropy=entropy.expand(logits.shape[0])
    if entropy.shape!=(logits.shape[0],):raise ValueError('Entropy batch does not match logits')
    result=logits.clone()
    result[:,2:]=result[:,2:].masked_fill((entropy>threshold).unsqueeze(1),-torch.inf)
    return result
