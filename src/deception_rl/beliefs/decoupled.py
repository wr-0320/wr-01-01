"""Calibrated inference independent of risk-aware action selection."""
import numpy as np
import torch
from torch import nn
from deception_rl.beliefs.two_level import HonestPrototypeReasoner
from experiments.analyze_identifiability import policy

class LikelihoodRatioReasoner(HonestPrototypeReasoner):
    def __init__(self,records):
        super().__init__(records);self.deceptive_prototypes=[]
        # Same public candidate-goal hypotheses; no actual hidden goal.
        moves=np.array([[-1,0],[1,0],[0,-1],[0,1],[0,0]])
        for goal in (0,1):
            paths=[]
            for family in ('scripted_1','scripted_2','scripted_3'):
                agent=policy(family);position=self.prototypes[goal][0].copy();path=[position.copy()];target=np.array([[0,0],[6,6]])
                for _ in range(30):
                    action=agent.act(dict(position=position,goal=goal,target_positions=target));position=np.clip(position+moves[action],0,6);path.append(position.copy())
                    if np.array_equal(position,target[goal]):break
                paths.append(np.array(path))
            self.deceptive_prototypes.append(paths)
    def likelihoods(self,trajectory):
        points=np.asarray(trajectory,dtype=int)
        if not len(points):raise ValueError('Visible history required')
        def compatibility(path):
            expected=path[np.clip(points[:,0],0,len(path)-1)]
            return float(np.exp(-np.abs(points[:,1:]-expected).sum(axis=1).mean()/12/self.temperature))
        h=np.array([compatibility(p) for p in self.prototypes]);d=np.array([np.mean([compatibility(p) for p in paths]) for paths in self.deceptive_prototypes])
        lh=float(h.mean());ld=float(d.mean())
        return dict(honest_likelihood=lh,deceptive_likelihood=ld,likelihood_ratio=ld/max(lh,1e-8),honest_by_goal=h.tolist(),deceptive_by_goal=d.tolist())

def logit(p):return float(np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))))

def calibration_features(raw,likelihood,timestep):
    h=np.asarray(likelihood['honest_by_goal']);d=np.asarray(likelihood['deceptive_by_goal']);pd=raw['deception_probability']
    hr=float(np.log(max(h[1],1e-8)/max(h[0],1e-8)));dr=float(np.log(max(d[1],1e-8)/max(d[0],1e-8)))
    return torch.tensor([logit(raw['goal_belief'][1]),hr,dr,logit(pd),np.log(max(likelihood['likelihood_ratio'],1e-8)),timestep/30,hr*pd,dr*pd],dtype=torch.float32)

class EvidenceCalibrator(nn.Module):
    def __init__(self):super().__init__();self.heads=nn.Linear(8,2)
    def forward(self,x):return self.heads(x)  # goal-B logit, deceptive logit

class CalibratedGoalInference:
    def __init__(self,raw_belief,reasoner,calibrator):self.raw=raw_belief;self.reasoner=reasoner;self.calibrator=calibrator.eval()
    def predict(self,observation,timestep):
        raw=self.raw.predict(observation,timestep);cf=self.reasoner.likelihoods(raw['trajectory'])
        with torch.inference_mode():p=torch.sigmoid(self.calibrator(calibration_features(raw,cf,timestep))).numpy()
        goal=[float(1-p[0]),float(p[0])];entropy=float(-sum(v*np.log2(max(v,1e-8)) for v in goal))
        return dict(goal_belief=goal,confidence=max(goal),entropy=entropy,deception_probability=float(p[1]),trajectory=raw['trajectory'],raw_goal_belief=raw['goal_belief'],raw_deception_probability=raw['deception_probability'],**cf)
