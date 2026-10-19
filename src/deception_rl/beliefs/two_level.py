"""Two-head belief, honest prototypes and reward-free adaptive commitment."""
import numpy as np
import torch
from models.intent_gru import trajectory_to_features
from deception_rl.envs.gridworld import OBSERVE,TRACK
from deception_rl.agents.active_anti_deception import trajectory_evidence

class HonestPrototypeReasoner:
    def __init__(self,records,targets=((0,0),(6,6)),temperature=.2):
        if temperature<=0:raise ValueError('temperature must be positive')
        self.temperature=temperature;self.prototypes=[]
        # Fit starts from honest training data; extend with the unchanged public shortest-path rule.
        for goal in (0,1):
            examples=[r for r in records if r['family']=='honest' and r['goal_label']==goal]
            if not examples:raise ValueError('Honest training examples for both goals required')
            start=examples[0]['trajectory'][0][1:];position=np.array(start,dtype=int);target=np.array(targets[goal]);path=[position.copy()]
            while not np.array_equal(position,target):
                axis=0 if position[0]!=target[0] else 1;position=position.copy();position[axis]+=int(np.sign(target[axis]-position[axis]));path.append(position)
            self.prototypes.append(np.asarray(path))
    def score(self,trajectory):
        points=np.asarray(trajectory,dtype=int)
        if not len(points):raise ValueError('Visible trajectory required')
        distances=[]
        for prototype in self.prototypes:
            expected=prototype[np.clip(points[:,0],0,len(prototype)-1)]
            distances.append(float(np.abs(points[:,1:]-expected).sum(axis=1).mean()/12))
        likelihoods=np.exp(-np.asarray(distances)/self.temperature)
        honest=float(likelihoods.max());observed=1. # self-compatibility; not a probability density
        return dict(observed_likelihood=observed,honest_likelihood=honest,deception_score=observed-honest,distance_by_goal=distances)

class DeceptionAwareBelief:
    def __init__(self,model):self.model=model.eval()
    def predict(self,observation,timestep):
        mask=np.asarray(observation['trajectory_mask'],dtype=bool).copy();mask[timestep+1:]=False
        times=np.flatnonzero(mask);positions=np.asarray(observation['trajectory'])[times]
        points=np.column_stack((times,positions)).astype(int).tolist()
        if not points:return dict(goal_belief=[.5,.5],deception_probability=.5,entropy=1.,trajectory=[])
        x=trajectory_to_features(points).unsqueeze(0)
        with torch.inference_mode():
            goal,deception=self.model(x,torch.tensor([len(points)]));g=torch.softmax(goal,1)[0].cpu().numpy();d=float(torch.softmax(deception,1)[0,1])
        return dict(goal_belief=g.tolist(),deception_probability=d,entropy=float(-(g*np.log2(np.clip(g,1e-8,1))).sum()),trajectory=points)

class PassiveTrajectoryClassifier:
    def __init__(self,belief,commit_step=5):self.belief=belief;self.commit_step=commit_step;self.reset()
    def reset(self):self.goal=None;self.commit_timestep=None;self.info=None
    def act(self,observation,timestep):
        self.info=self.belief.predict(observation,timestep)
        if self.goal is not None:return self.goal
        if timestep>=self.commit_step:
            self.goal=int(np.argmax(self.info['goal_belief']));self.commit_timestep=timestep;return self.goal
        return OBSERVE

class TwoLevelAntiDeception(PassiveTrajectoryClassifier):
    def __init__(self,belief,reasoner,entropy_threshold=.85,deception_threshold=.6):
        super().__init__(belief);self.reasoner=reasoner;self.entropy_threshold=entropy_threshold;self.deception_threshold=deception_threshold
    def act(self,observation,timestep):
        self.info=self.belief.predict(observation,timestep)
        counter=self.reasoner.score(self.info['trajectory']);self.info.update(counter)
        self.info['model_deception_probability']=self.info['deception_probability']
        self.info['deception_probability']=.5*(self.info['model_deception_probability']+counter['deception_score'])
        if self.goal is not None:return self.goal
        evidence=trajectory_evidence(observation,timestep)
        high_uncertainty=self.info['entropy']>self.entropy_threshold
        high_deception=self.info['deception_probability']>=self.deception_threshold
        required=3 if high_deception else 2
        if timestep>=5 and not high_uncertainty and evidence.age<=1 and evidence.consistent_moves>=required:
            self.goal=int(np.argmax(self.info['goal_belief']));self.commit_timestep=timestep;return self.goal
        positions=np.asarray(observation['trajectory'])[np.asarray(observation['trajectory_mask'],dtype=bool)]
        near=bool(len(positions) and np.min(np.abs(np.asarray(observation['target_positions'])-positions[-1]).sum(axis=1))<=2)
        if high_uncertainty or high_deception or evidence.age>=1 or evidence.consistent_moves<2 or near:return TRACK
        return OBSERVE
