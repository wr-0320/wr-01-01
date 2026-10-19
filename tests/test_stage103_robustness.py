import copy
from collections import Counter
import numpy as np
import pytest
from experiments.stage103_deception_families import policy,scenario_schedule,MOVES
from experiments.run_stage10_3_robustness import paired_differences


def rollout(family,goal=0,seed=0):
    agent=policy(family);agent.reset(seed=seed);position=np.array([3,3]);targets=np.array([[0,0],[6,6]])
    actions=[];positions=[position.tolist()]
    for _ in range(30):
        action=agent.act(dict(position=position.copy(),goal=goal,target_positions=targets.copy()))
        actions.append(action);position=np.clip(position+MOVES[action],0,6);positions.append(position.tolist())
        if np.array_equal(position,targets[goal]):break
    return actions,positions


def test_long_scripted_preserves_false_target_wait_then_reversal():
    for steps in (8,10):
        actions,positions=rollout(f'scripted_{steps}')
        assert positions[6]==[6,6]
        assert actions[6:steps]==[4]*(steps-6)
        assert actions[steps]==0
        assert positions[-1]==[0,0]
        assert len(actions)==steps+12


def test_mixed_strategy_composition_and_eventual_goal():
    a,p=rollout('delay_detour');assert a[:7]==[0,0,1,1,3,3,0];assert p[-1]==[0,0]
    a,p=rollout('detour_switch');assert a[:7]==[3,3,1,0,1,0,0];assert p[-1]==[0,0]
    for family in ('delay_detour','detour_switch'):
        a,pa=rollout(family,0);b,pb=rollout(family,1)
        assert np.array_equal(6-np.asarray(pa),np.asarray(pb))


def test_stochastic_reproducible_and_has_multiple_observable_routes():
    assert rollout('stochastic_deception',seed=17)==rollout('stochastic_deception',seed=17)
    routes=set()
    for seed in range(16):
        for goal in (0,1):
            actions,positions=rollout('stochastic_deception',goal,seed)
            assert positions[-1]==([0,0] if goal==0 else [6,6])
            assert len(actions)<=20
            routes.add(tuple(actions))
    assert len(routes)>2


def test_fixed_smoke_budget_and_goal_balance():
    paired,extras=scenario_schedule()
    assert len(paired)*3+len(extras)==512
    assert Counter(s['goal'] for s in paired)=={'A':84,'B':84}
    for family in {r['family'] for r in paired}:
        counts=Counter(r['goal'] for r in paired if r['family']==family)
        assert counts['A']==counts['B']
    assert len({r['seed'] for r in paired+extras})==176
    assert len(extras)==8 and all(r['family']=='stochastic_deception' for r in extras)


def test_paired_check_rejects_different_actual_random_trajectories():
    row=dict(episode=0,family='stochastic_deception',true_goal='A',seed=12,deceiver_trace=[dict(action=0)],
             defender_task_success=1,false_commitment=0,observation_cost=.1,decision_regret=.1,decision_latency=4)
    rows=[dict(copy.deepcopy(row),controller=name) for name in ('rule_active','stage10_1','adaptive')]
    assert paired_differences(rows)[0]['cost_delta']==0
    altered=copy.deepcopy(rows);altered[2]['deceiver_trace'][0]['action']=1
    with pytest.raises(AssertionError):paired_differences(altered)
