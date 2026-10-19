import copy
import pytest
import torch
from experiments.analyze_identifiability import features,identifiability,generate_dataset,diagnostic_split,FAMILIES
from models.anti_deception_gru import AntiDeceptionGRU

def example(label=0):
    return dict(trajectory=[[0,3,3],[2,2,3]],belief_history=[[.5,.5],[.7,.3]],deception_label=label,goal_label=label,family='honest' if not label else 'scripted_5')

def test_three_modes_use_only_permitted_channels():
    a=example();b=copy.deepcopy(a);b['trajectory'][1][0]=3;b['deception_label']=1
    torch.testing.assert_close(features(a,'A'),features(b,'A'))
    assert not torch.equal(features(a,'B'),features(b,'B'))
    c=copy.deepcopy(a);c['belief_history'][1]=[.6,.4]
    assert not torch.equal(features(a,'C'),features(c,'C'))
    for mode in ('A','B','C'):
        d=copy.deepcopy(a);d.update(deception_label=1,goal_label=1,family='different',seed=999)
        torch.testing.assert_close(features(a,mode),features(d,mode))

def test_identical_observation_conflicts_have_irreducible_error():
    a=example(0);b=example(1);model=AntiDeceptionGRU(8,input_size=8).eval()
    for mode in ('A','B','C'):
        x=torch.stack([features(a,mode),features(b,mode)])
        _,logits=model(x,torch.tensor([2,2]));torch.testing.assert_close(logits[0],logits[1])
        report=identifiability([a,b],mode)
        assert report['minimum_errors']==1
        assert report['empirical_accuracy_ceiling']==.5
        assert report['empirical_f1_ceiling']==pytest.approx(2/3)
        assert report['empirical_auroc_ceiling']==.5

class ConstantEstimator:
    def estimate(self,trajectory,trajectory_mask,target_positions):return {'A':.5,'B':.5}

def test_generation_and_split_are_label_balanced_without_pseudo_negatives():
    rows=generate_dataset(ConstantEstimator(),10)
    assert len(rows)==160
    assert sum(r['deception_label'] for r in rows)==80
    assert {r['family'] for r in rows}==set(FAMILIES)
    splits=diagnostic_split(rows)
    seen=set()
    for records in splits.values():
        assert records and len(records)==2*sum(r['deception_label'] for r in records)
        ids={r['sample_id'] for r in records};assert not ids & seen;seen|=ids
        assert all(len(r['trajectory'])==len(r['belief_history']) for r in records)
