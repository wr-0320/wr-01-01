from copy import deepcopy
import numpy as np
import pytest
from deception_rl.envs.deceptive_intelligence import DeceptiveIntelligenceEnv,NUM_HYPOTHESES
from deception_rl.intelligence.ride_v2.fixed_reliability import FixedCellReliability
from deception_rl.intelligence.ride_v2.reliability_conditioned import ReliabilityConditionedLikelihood
from deception_rl.intelligence.ride_v2.intent_coupled import IntentCoupledLikelihood


def fixture():
    mean=np.zeros((NUM_HYPOTHESES,3,3,2))
    for h in range(NUM_HYPOTHESES):mean[h]=h/NUM_HYPOTHESES
    base=ReliabilityConditionedLikelihood(mean,np.ones_like(mean)*.1,np.ones((3,3,2))*.6,np.ones((3,3,2))*.2,.5)
    fixed=FixedCellReliability(np.ones((3,3))*.8)
    rows=[]
    for h in range(NUM_HYPOTHESES):
        for f in range(4):
            for _ in range(4):rows.append({'hypothesis_label':h,'surface_label':f,'reported_cells':np.ones((3,3),dtype=bool),'reported_values':np.ones((3,3,4))*.4})
    return base,fixed,rows


def test_identical_surface_components_collapse_to_ordinary_rcl():
    base,fixed,_=fixture()
    matrix=np.ones((NUM_HYPOTHESES,4))
    for h in range(NUM_HYPOTHESES):matrix[h,h%4]=10
    matrix/=matrix.sum(axis=1,keepdims=True)
    model=IntentCoupledLikelihood(base,fixed.reliable,np.broadcast_to(base.compromised_mean,(4,3,3,2)),np.broadcast_to(base.compromised_scale,(4,3,3,2)),matrix)
    env=DeceptiveIntelligenceEnv();obs,_=env.reset(seed=88)
    for _ in range(3):obs,*_=env.step({'red':0,'blue':0})
    before=deepcopy(obs['blue'])
    np.testing.assert_allclose(model.logits(obs['blue']),base.logits(obs['blue'],fixed,None),atol=1e-9,rtol=0)
    for key in before:np.testing.assert_array_equal(obs['blue'][key],before[key])
    with pytest.raises(ValueError):model.logits(dict(obs['blue'],surface_label=0))


def test_equal_conditional_rows_equal_unconditional_surface_control():
    base,fixed,rows=fixture()
    coupled=IntentCoupledLikelihood.fit(rows,base,fixed)
    control=IntentCoupledLikelihood(base,fixed.reliable,coupled.surface_mean,coupled.surface_scale,coupled.transition,coupled=False)
    env=DeceptiveIntelligenceEnv();obs,_=env.reset(seed=91)
    obs,*_=env.step({'red':0,'blue':0})
    np.testing.assert_allclose(coupled.logits(obs['blue']),control.logits(obs['blue']),atol=1e-9)


def test_intent_runner_smoke(tmp_path):
    import json
    from experiments.run_intelligence_intent_coupled import run
    destination=tmp_path/'intent'
    result=run(destination,seeds=(6901,),episodes=1,train_episodes=1,smoke=True)
    assert not result['policy_independent_strategy_claim']
    audit=json.loads((destination/'audit.json').read_text())
    assert audit['test_episodes']==45
    assert audit['latent_component_parameters_shared_with_unconditional_control']
    assert not audit['runtime_H_F_truth_used']
