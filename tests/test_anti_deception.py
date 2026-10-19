import copy
import pytest
import torch
from deception_rl.data.anti_deception import DeceptionDataset, family_split, collate, audit
from experiments.train_anti_deception import binary_metrics
from models.anti_deception_gru import AntiDeceptionGRU

def record(family,goal=0,trajectory=None):
    return dict(family=family,goal_label=goal,deception_label=int(family!='honest'),trajectory=trajectory or [[0,3,3],[2,2,3]])

def test_family_holdout_and_duplicate_leakage():
    rows=[record('honest'),record('scripted_1',trajectory=[[0,3,3],[1,4,3]]),record('scripted_4',trajectory=[[0,3,3],[4,5,3]])]
    train,val=family_split(rows)
    assert {r['family'] for r in train}=={'honest','scripted_1'}
    assert {r['family'] for r in val}=={'scripted_4'}
    rows[-1]['trajectory']=rows[0]['trajectory']
    with pytest.raises(ValueError,match='cross train/validation'):family_split(rows)

def test_only_visible_inputs_and_padding_invariance():
    r=record('honest');altered=copy.deepcopy(r)
    altered.update(goal_label=1,deception_label=1,source='secret label metadata')
    assert torch.equal(DeceptionDataset([r])[0][0],DeceptionDataset([altered])[0][0])
    short=DeceptionDataset([r])[0];long=DeceptionDataset([record('scripted_1',trajectory=[[0,3,3],[1,4,3],[2,5,3]])])[0]
    x,lengths,_,_=collate([short,long]);model=AntiDeceptionGRU(8).eval()
    a=model(x,lengths);x[0,2]=1000;b=model(x,lengths)
    for first,second in zip(a,b):torch.testing.assert_close(first,second)

def test_binary_metrics_ties_and_single_class():
    assert binary_metrics([0,1],[.5,.5])['auroc']==.5
    assert binary_metrics([0,1],[.1,.9])['auroc']==1
    assert binary_metrics([0,1],[.9,.1])['auroc']==0
    result=binary_metrics([1,1],[.8,.9]);assert result['auroc'] is None and result['f1']==1

def test_test_alias_audit_retains_unseen_family():
    honest=record('honest');test=record('scripted_5',goal=1)
    report=audit({'train':[honest],'val':[record('scripted_4',trajectory=[[0,3,3],[1,6,3]])],'test':[test]})
    alias=report['test_observation_aliases'][0]
    assert alias['observations_seen_in_train']==1 and alias['conflicting_goal_labels']==1 and alias['conflicting_deception_labels']==1

def test_stage6_loader_filters_policies_observers_and_unseen_steps(tmp_path):
    import csv
    import json
    from deception_rl.data.anti_deception import load_stage6
    def write_csv(name, rows):
        path=tmp_path/name
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return path
    common={'trajectory':json.dumps([[0,3,3],[1,2,3]]),'true_goal':'A'}
    training=[dict(common,deceiver_policy='honest_shortest_path',deception_steps='')]
    training += [dict(common,deceiver_policy='scripted_deception',deception_steps=str(i)) for i in range(1,5)]
    training += [dict(common,deceiver_policy='random_policy',deception_steps='')]
    scaling=[dict(common,model=m,deception_steps=str(s)) for m in ['ModelA','ModelB'] for s in [4,5,6]]
    families=[dict(common,model=m,deception_family=f) for m in ['ModelA','ModelB'] for f in ['delayed_deception','detour_deception','multi_switch_deception','scripted_extrapolation']]
    pool,test=load_stage6(write_csv('train.csv',training),write_csv('scaling.csv',scaling),write_csv('families.csv',families))
    assert len(pool)==5 and len(test)==4
    assert {r['family'] for r in test}=={'scripted_5','delayed_deception','detour_deception','multi_switch_deception'}
    assert all(r['observer']=='ModelA' for r in test)
    assert all(r['deception_label']==1 for r in test)

def test_family_split_cannot_remove_all_honest_training_data():
    rows=[record('honest'),record('scripted_1',trajectory=[[0,3,3],[1,4,3]])]
    with pytest.raises(ValueError,match='both Honest and deceptive'):
        family_split(rows,('honest',))
