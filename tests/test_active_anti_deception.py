import copy
import numpy as np
import pytest
from deception_rl.agents.active_anti_deception import trajectory_evidence,ActiveAntiDeception,PassiveAntiDeception,DefenderReward
from deception_rl.envs.gridworld import OBSERVE,TRACK
from experiments.run_active_anti_deception import episode,run

def observation(points):
    trajectory=np.full((31,2),-1,dtype=np.int64);mask=np.zeros(31,dtype=np.int8)
    for t,r,c in points:trajectory[t]=[r,c];mask[t]=1
    return {'trajectory':trajectory,'trajectory_mask':mask,'target_positions':np.array([[0,0],[6,6]])}

def test_detector_uses_visible_points_and_distinguishes_turn_from_reversal():
    obs=observation([(0,3,3),(1,2,3),(2,2,2)])
    assert trajectory_evidence(obs,2).deception_probability==.1
    reversal=observation([(0,3,3),(1,2,3),(2,3,3)])
    assert trajectory_evidence(reversal,2).deception_probability==.9
    hidden=copy.deepcopy(reversal);hidden['trajectory'][3:]=6;hidden['true_goal']=1
    assert trajectory_evidence(hidden,2)==trajectory_evidence(reversal,2)

def test_active_actions_and_irreversible_commitment():
    active=ActiveAntiDeception()
    assert active.act(observation([(0,3,3)]),0)==TRACK
    confident=observation([(0,3,3),(1,2,3),(2,1,3)])
    assert active.act(confident,2)==OBSERVE
    assert active.act(confident,5)==TRACK  # stale evidence does not trigger early commitment
    fresh=observation([(0,3,3),(3,2,3),(4,1,3)])
    assert active.act(fresh,5)==0
    opposite=observation([(0,3,3),(4,4,3),(5,5,3)])
    assert active.act(opposite,6)==0
    active.reset();assert active.goal is None

def test_score_applies_false_commitment_once_and_sensing_costs():
    reward=DefenderReward()
    assert reward.components(TRACK,False,False,None,1)['observation_cost']==.03
    first=reward.components(0,False,True,0,1);later=reward.components(0,False,False,0,1)
    assert first['false_commitment_penalty']==.5
    assert later['false_commitment_penalty']==0
    assert reward.components(1,True,True,1,1)['interception_reward']==1

@pytest.mark.parametrize('goal',['A','B'])
def test_passive_honest_interception_and_return(goal):
    r=episode('honest',goal,0,'passive',0,DefenderReward())
    assert r['defender_task_success']==1 and r['false_commitment']==0
    assert r['observation_cost']==pytest.approx(.05) and r['episode_return']==pytest.approx(.95)
    assert r['commit_timestep']==5 and r['detection_probability_t5']==.1

def test_paired_runner_has_balanced_labels_and_consistent_costs(tmp_path):
    import csv,json
    summary=run(tmp_path,per_family=2)
    rows=list(csv.DictReader((tmp_path/'evaluation_episodes.csv').open()))
    assert len(rows)==64
    for name in ['passive','active']:
        g=[r for r in rows if r['controller']==name]
        assert len(g)==32 and sum(int(r['deception_label']) for r in g)==16
        assert summary[name]['commitment_coverage']==1
        for r in g:
            trace=json.loads(r['trace'])
            expected=sum(.01 if t['action']=='observe' else .03 if t['action']=='track' else 0 for t in trace)
            assert float(r['observation_cost'])==pytest.approx(expected)
            assert float(r['false_commitment_penalty'])==pytest.approx(.5*int(r['false_commitment']))

def test_active_tracks_near_goal_before_commit_and_observes_reversal():
    active=ActiveAntiDeception()
    near=observation([(0,3,3),(1,4,3),(2,5,3),(4,6,4)])
    assert active.act(near,4)==TRACK
    reversed_move=observation([(0,3,3),(1,4,3),(2,5,3),(4,6,4),(5,6,3)])
    assert active.act(reversed_move,5)==TRACK
    assert active.goal is None and active.last_evidence.deception_probability==.9
