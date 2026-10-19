from copy import deepcopy
import numpy as np
import pytest
from deception_rl.envs.deceptive_intelligence import DeceptiveIntelligenceEnv,NUM_HYPOTHESES
from deception_rl.intelligence.ride_v2.fixed_reliability import FixedCellReliability
from deception_rl.intelligence.ride_v2.reliability_conditioned import ReliabilityConditionedLikelihood
from deception_rl.intelligence.ride_v2.correlated_likelihood import CorrelationAwareLikelihood
from deception_rl.intelligence.ride_v2.source_scope import restrict_observation


def fixture_models(strength=1.):
    samples=[]
    for step in range(100):
        cells=np.zeros((3,3),dtype=int)
        if step%10==0:cells[:,step%3]=1
        elif step%4==0:cells[step%3,(step+1)%3]=1
        samples.append({'compromised_cells':cells,'reported_cells':np.ones((3,3),dtype=bool)})
    mean=np.zeros((NUM_HYPOTHESES,3,3,2))
    for h in range(NUM_HYPOTHESES):mean[h]=h/NUM_HYPOTHESES
    base=ReliabilityConditionedLikelihood(mean,np.full_like(mean,.1),np.full((3,3,2),.7),np.full((3,3,2),.2),.5)
    fixed=FixedCellReliability.fit(samples)
    return base,fixed,CorrelationAwareLikelihood.fit_dependency(samples,base,fixed,strength=strength)


def test_dependency_preserves_marginals_and_collapses_to_independent():
    base,fixed,model=fixture_models(0.)
    np.testing.assert_allclose(model.marginal_compromise(),1.-fixed.reliable,atol=1e-12)
    env=DeceptiveIntelligenceEnv()
    obs,_=env.reset(seed=12)
    for _ in range(3):obs,*_=env.step({'red':0,'blue':0})
    blue=obs['blue'];before=deepcopy(blue)
    np.testing.assert_allclose(model.logits(blue),base.logits(blue,fixed,None),atol=1e-9,rtol=0)
    for key in blue:np.testing.assert_array_equal(blue[key],before[key])
    with pytest.raises(ValueError):model.logits(dict(blue,true_hypothesis=0))


def test_one_observed_source_cannot_benefit_from_cross_source_dependence():
    base,fixed,model=fixture_models()
    env=DeceptiveIntelligenceEnv()
    obs,_=env.reset(seed=19)
    for _ in range(3):obs,*_=env.step({'red':0,'blue':0})
    single=restrict_observation(obs['blue'],1)
    np.testing.assert_allclose(model.logits(single),base.logits(single,fixed,None),atol=1e-9,rtol=0)
    r=model.latest_coordination_scores(single,np.ones(NUM_HYPOTHESES)/NUM_HYPOTHESES)
    assert np.all((r>=0)&(r<=1))
    assert np.any(model.conditional_compromise[1]>model.conditional_compromise[0])


def test_correlated_runner_smoke(tmp_path):
    import json
    from experiments.run_intelligence_correlated_rcl import run
    destination=tmp_path/'correlated'
    result=run(destination,seeds=(5901,),episodes=1,train_episodes=1,smoke=True)
    audit=json.loads((destination/'audit.json').read_text())
    assert audit['total_test_episodes']==28
    assert audit['cell_marginals_preserved']
    assert audit['zero_dependency_equals_independent_verified']
    assert not result['novel_method_claim_established']
