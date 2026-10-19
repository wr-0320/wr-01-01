import copy
import numpy as np
import torch
from deception_rl.beliefs.decoupled import LikelihoodRatioReasoner, EvidenceCalibrator, CalibratedGoalInference
from deception_rl.beliefs.two_level import DeceptionAwareBelief
from deception_rl.agents.decoupled_defender import RiskAwareActionScorer, DecoupledAntiDeception
from models.anti_deception_gru import AntiDeceptionGRU
from tests.test_two_level_framework import RECORDS, observation


def test_likelihood_ratio_indistinguishable_prefix():
    reasoner = LikelihoodRatioReasoner(RECORDS)
    score = reasoner.likelihoods([[0,3,3]])
    assert score['honest_likelihood'] == score['deceptive_likelihood'] == 1
    assert score['likelihood_ratio'] == 1


def test_inference_confidence_and_hidden_information_boundary():
    torch.manual_seed(0)
    inference = CalibratedGoalInference(DeceptionAwareBelief(AntiDeceptionGRU(8)), LikelihoodRatioReasoner(RECORDS), EvidenceCalibrator())
    a = observation([(0,3,3),(1,2,3)])
    b = copy.deepcopy(a)
    b['true_goal'] = 1
    b['trajectory'][20] = [6,6]
    b['trajectory_mask'][20] = 1
    result = inference.predict(a,1)
    assert result == inference.predict(b,1)
    assert np.isclose(sum(result['goal_belief']),1)
    assert result['confidence'] == max(result['goal_belief'])


def test_utility_waits_when_uncertain_and_commits_at_arrival_risk():
    scorer = RiskAwareActionScorer()
    belief = dict(goal_belief=[.5,.5],entropy=1.,deception_probability=.9,trajectory=[[0,3,3]])
    targets = np.array([[0,0],[6,6]])
    scores = scorer.scores(belief,targets,0)
    assert max(scores,key=scores.get) == 'track'
    belief.update(goal_belief=[.99,.01],entropy=.08,trajectory=[[5,0,1]])
    scores = scorer.scores(belief,targets,5)
    assert max(scores,key=scores.get) == 'commit_A'


def test_commitment_remains_irreversible():
    class Inference:
        def predict(self,o,t):
            return dict(goal_belief=[.99,.01] if t==5 else [.01,.99],entropy=.08,deception_probability=.1,trajectory=[[5,0,1]])
    controller = DecoupledAntiDeception(Inference())
    obs = observation([(5,0,1)])
    assert controller.act(obs,5) == 0
    assert controller.act(obs,6) == 0
    assert controller.commit_timestep == 5
