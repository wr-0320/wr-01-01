"""Choose actions by utility from inference output; no policy reward training."""
import numpy as np
from deception_rl.envs.gridworld import OBSERVE,TRACK

class RiskAwareActionScorer:
    def scores(self,belief,targets,timestep):
        goal=np.asarray(belief['goal_belief']);entropy=belief['entropy'];deception=belief['deception_probability']
        # Current utility assumes the chosen goal is intercepted if correct.
        commit=goal-.5*(1-goal)
        trajectory=belief['trajectory'];last=np.asarray(trajectory[-1][1:]);age=timestep-trajectory[-1][0]
        distance=np.abs(np.asarray(targets)-last).sum(axis=1)
        # Conservative probability mass that could arrive on the next action.
        arrival_risk=float(goal[distance<=age+1].sum())
        scores={'commit_A':float(commit[0]),'commit_B':float(commit[1])}
        current=float(goal.max())
        for action,cost,gain in [('observe',.01,.03),('track',.03,.08)]:
            future=min(1.,current+gain*entropy*(1+deception))
            expected_interception=future*(1-arrival_risk)
            false_commitment_risk=.5*(1-future)
            scores[action]=expected_interception-cost-false_commitment_risk
        return scores

class DecoupledAntiDeception:
    def __init__(self,inference,scorer=None):self.inference=inference;self.scorer=scorer or RiskAwareActionScorer();self.reset()
    def reset(self):self.goal=None;self.commit_timestep=None;self.info=None
    def act(self,observation,timestep):
        self.info=self.inference.predict(observation,timestep)
        if self.goal is not None:return self.goal
        scores=self.scorer.scores(self.info,observation['target_positions'],timestep);self.info['action_utilities']=scores
        action=max(scores,key=scores.get)
        if action.startswith('commit_'):
            self.goal=int(action=='commit_B');self.commit_timestep=timestep;return self.goal
        return OBSERVE if action=='observe' else TRACK
