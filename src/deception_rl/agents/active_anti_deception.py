"""Rule-based anti-deception controllers using Defender-visible history only."""
from dataclasses import dataclass
import numpy as np
from deception_rl.envs.gridworld import GUARD_A,GUARD_B,OBSERVE,TRACK

@dataclass(frozen=True)
class Evidence:
    deception_probability: float
    predicted_goal: int
    consistent_moves: int
    age: int
    reversals: int

def trajectory_evidence(observation,timestep):
    points=np.asarray(observation['trajectory'])[np.asarray(observation['trajectory_mask'],dtype=bool)]
    times=np.flatnonzero(np.asarray(observation['trajectory_mask'],dtype=bool))
    if not len(points):return Evidence(.5,0,0,timestep+1,0)
    moves=np.diff(points,axis=0);moves=moves[np.any(moves!=0,axis=1)]
    # Compare signs on each axis only when that axis moved; normal turns are not reversals.
    last_sign=[0,0];reversals=0
    for move in moves:
        for axis in range(2):
            sign=int(np.sign(move[axis]))
            if sign:
                reversals+=int(last_sign[axis]!=0 and sign!=last_sign[axis]);last_sign[axis]=sign
    directional=[int(np.sign(move.sum())) for move in moves if move.sum()!=0]
    direction=directional[-1] if directional else 0
    consistent=0
    for sign in reversed(directional):
        if sign!=direction:break
        consistent+=1
    target=np.asarray(observation['target_positions'])
    if direction:goal=int(direction>0)
    else:goal=int(np.argmin(np.abs(target-points[-1]).sum(axis=1)))
    # Heuristic scores, not fitted/calibrated probabilities.
    probability=.9 if reversals else (.1 if len(moves) else .5)
    return Evidence(probability,goal,consistent,max(0,timestep-int(times[-1])),reversals)

class PassiveAntiDeception:
    """Fixed observe schedule; trajectory detector never controls sensing."""
    def __init__(self,commit_step=5):
        if commit_step<0:raise ValueError('commit_step must be nonnegative')
        self.commit_step=commit_step;self.reset()
    def reset(self):self.goal=None;self.commit_timestep=None;self.last_evidence=None
    def act(self,observation,timestep):
        self.last_evidence=trajectory_evidence(observation,timestep)
        if self.goal is not None:return self.goal
        if timestep>=self.commit_step:
            self.goal=self.last_evidence.predicted_goal;self.commit_timestep=timestep;return self.goal
        return OBSERVE

class ActiveAntiDeception(PassiveAntiDeception):
    """Adaptive sensing and evidence-dependent irreversible commitment."""
    def act(self,observation,timestep):
        self.last_evidence=trajectory_evidence(observation,timestep)
        if self.goal is not None:return self.goal
        e=self.last_evidence
        required=3 if e.deception_probability>=.5 else 2
        ready=e.age<=1 and e.consistent_moves>=required
        if timestep>=self.commit_step and (ready or (timestep>=12 and e.consistent_moves>0)):
            self.goal=e.predicted_goal;self.commit_timestep=timestep;return self.goal
        visible=np.asarray(observation['trajectory'])[np.asarray(observation['trajectory_mask'],dtype=bool)]
        near_target=bool(len(visible) and np.min(np.abs(np.asarray(observation['target_positions'])-visible[-1]).sum(axis=1))<=2)
        # Before commitment, preserve wider sensing near a possible feint endpoint.
        if near_target or e.age>=1 or e.consistent_moves<2 or e.deception_probability>=.5:return TRACK
        return OBSERVE

@dataclass(frozen=True)
class DefenderReward:
    interception: float=1.0
    observe_cost: float=.01
    track_cost: float=.03
    false_commitment_penalty: float=.5
    def components(self,action,intercepted,new_commitment,committed_goal,true_goal):
        return dict(interception_reward=self.interception*int(intercepted),
                    observation_cost=self.observe_cost*int(action==OBSERVE)+self.track_cost*int(action==TRACK),
                    false_commitment_penalty=self.false_commitment_penalty*int(new_commitment and committed_goal!=true_goal))
