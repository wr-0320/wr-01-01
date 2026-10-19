import copy
from collections import Counter
import numpy as np
from deception_rl.agents.adaptive_defender import AdaptiveObservationScorer, AdaptiveActiveDefender
from experiments.run_stage10_2_adaptive import schedule, add_metrics


def observation():
    return dict(local_map=np.zeros((7,7)),sensor_origin=np.array([3,3]),target_positions=np.array([[0,0],[6,6]]))


def belief(position=(3,3),lr=1.):
    return dict(goal_belief=[.5,.5],entropy=1.,deception_probability=0.,likelihood_ratio=lr,trajectory=[[0,*position]])


def test_equal_coverage_prefers_cheaper_observe_and_uses_cost():
    scorer=AdaptiveObservationScorer()
    scores,diagnostics=scorer.evaluate(belief(),observation(),0)
    assert diagnostics['sensor_coverage']=={'observe':1.,'track':1.}
    assert max(scores,key=scores.get)=='observe'
    expensive=AdaptiveObservationScorer(observe_cost=.3)
    scores,_=expensive.evaluate(belief(),observation(),0)
    assert max(scores,key=scores.get)=='track'


def test_likelihood_ratio_changes_sensing_choice_when_extra_coverage_matters():
    scorer=AdaptiveObservationScorer(information_gain=.022)
    low,_=scorer.evaluate(belief((1,3)),observation(),0)
    high,detail=scorer.evaluate(belief((1,3),1000.),observation(),0)
    assert max(low,key=low.get)=='observe'
    assert max(high,key=high.get)=='track'
    assert detail['sensor_coverage']['track']>detail['sensor_coverage']['observe']


def test_deadline_risk_commits_and_commit_is_irreversible():
    class Inference:
        def predict(self,o,t):
            result=belief((0,1));result.update(goal_belief=[.99,.01] if t==0 else [.01,.99],entropy=.08)
            return result
    controller=AdaptiveActiveDefender(Inference())
    assert controller.act(observation(),0)==0
    assert controller.act(observation(),1)==0
    assert controller.commit_timestep==0


def test_adaptive_scorer_ignores_hidden_goal_and_family():
    o=observation();other=copy.deepcopy(o)
    other.update(true_goal=1,family='multi_switch_deception',deceiver_position=[6,6])
    scorer=AdaptiveObservationScorer()
    assert scorer.evaluate(belief(),o,0)==scorer.evaluate(belief(),other,0)


def test_fixed_budget_and_paired_label_balance():
    paired,audits=schedule(0)
    assert len(paired)*3+len(audits)==512
    assert Counter(r['goal'] for r in paired)=={'A':84,'B':84}
    assert sum(r['family']=='honest' for r in paired)==84
    assert len({r['episode_id'] for r in paired+audits})==176
    assert Counter(r['family'] for r in audits)=={'honest':2,'delayed_deception':2,'detour_deception':2,'multi_switch_deception':2}


def test_regret_includes_failure_sensing_and_false_commitment():
    row=add_metrics(dict(episode_return=-.65,trace=[dict(action='observe'),dict(action='track'),dict(action='commit_A')]))
    assert np.isclose(row['decision_regret'],1.65)
    assert row['observe_actions']==row['track_actions']==1
