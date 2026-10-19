import copy
import json
import numpy as np
import pytest
import torch
from models.anti_deception_gru import AntiDeceptionGRU
from deception_rl.beliefs.two_level import HonestPrototypeReasoner,DeceptionAwareBelief,TwoLevelAntiDeception
from experiments.run_stage10_framework import build_data,SEEN,UNSEEN

RECORDS=[dict(family='honest',goal_label=i,trajectory=[[0,3,3]]) for i in (0,1)]
def observation(points):
    p=np.full((31,2),-1,dtype=np.int64);mask=np.zeros(31,dtype=np.int8)
    for t,r,c in points:p[t]=[r,c];mask[t]=1
    return dict(trajectory=p,trajectory_mask=mask,target_positions=np.array([[0,0],[6,6]]))

def test_prototype_exact_honest_and_deviation_distance_without_goal_input():
    reasoner=HonestPrototypeReasoner(RECORDS)
    honest=[[0,3,3],[1,2,3],[2,1,3],[3,0,3],[4,0,2]]
    score=reasoner.score(honest);assert score['honest_likelihood']==1 and score['deception_score']==0
    reversal=[[0,3,3],[1,4,3],[2,3,3],[3,2,3]]
    assert reasoner.score(reversal)['deception_score']>0
    assert reasoner.score(honest)==reasoner.score(copy.deepcopy(honest))
    with pytest.raises(ValueError):HonestPrototypeReasoner([dict(family='scripted_1',goal_label=0)],temperature=.2)

def test_belief_model_rejects_hidden_future_information():
    torch.manual_seed(0);belief=DeceptionAwareBelief(AntiDeceptionGRU(8))
    a=observation([(0,3,3),(1,2,3)]);b=copy.deepcopy(a);b['trajectory'][2:]=6;b['trajectory_mask'][20]=1;b['true_goal']=1;b['family']='scripted_5'
    assert belief.predict(a,1)==belief.predict(b,1)

class StubBelief:
    def __init__(self,entropy,prob):self.entropy=entropy;self.prob=prob
    def predict(self,o,t):
        mask=o['trajectory_mask'].astype(bool);times=np.flatnonzero(mask)
        return dict(goal_belief=[.95,.05],deception_probability=self.prob,entropy=self.entropy,trajectory=np.column_stack((times,o['trajectory'][times])).tolist())

def test_rule_controller_uncertainty_sensing_and_irreversible_commit():
    o=observation([(0,3,3),(1,2,3),(2,1,3),(4,0,2)])
    controller=TwoLevelAntiDeception(StubBelief(1.,.1),HonestPrototypeReasoner(RECORDS))
    assert controller.act(o,5)==3 and controller.goal is None
    controller.belief=StubBelief(.1,.1);assert controller.act(o,5)==0
    controller.belief=StubBelief(1.,.9);assert controller.act(o,6)==0

def test_dataset_uses_only_seen_families_balanced_and_disjoint_ids(tmp_path):
    rows=[]
    for family in ('honest',*SEEN,*UNSEEN):
        for goal in (0,1):
            for i in range(200):rows.append(dict(sample_id=f'{family}:{goal}:{i}',family=family,goal_label=goal,deception_label=int(family!='honest'),trajectory=[[0,3,3]]))
    path=tmp_path/'data.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows))
    train,val=build_data(path)
    assert len(train)==512 and len(val)==128
    assert sum(r['deception_label'] for r in train)==256 and sum(r['deception_label'] for r in val)==64
    assert {r['family'] for r in train}=={'honest',*SEEN}
    assert not {r['sample_id'] for r in train}&{r['sample_id'] for r in val}
