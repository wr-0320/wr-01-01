from copy import deepcopy
import numpy as np
import torch
import pytest
from deception_rl.envs.deceptive_intelligence import DeceptiveIntelligenceEnv
from deception_rl.intelligence.adapter import encode_intelligence_sequence
from deception_rl.intelligence.ride_v2.calibrated_reliability import CalibratedCellReliabilityFilter, LogisticCellModel
from deception_rl.intelligence.ride_v2.reliability_conditioned import encode_reliability_conditioned_sequence, ReliabilityConditionedModel
from deception_rl.intelligence.ride import HonestIntelligenceReasoner
from experiments.run_intelligence_ride import collect_prefix_records


def test_conditioning_preserves_observation_and_original_features():
    records = collect_prefix_records(('honest',), 1, seed=817, split='condition_test', max_cycles=5)
    reasoner = HonestIntelligenceReasoner(records)
    model = LogisticCellModel(np.zeros(10), np.ones(10), np.zeros(17), True)
    predictor = CalibratedCellReliabilityFilter(model)
    env = DeceptiveIntelligenceEnv(max_cycles=5)
    obs, _ = env.reset(seed=17)
    obs, *_ = env.step({'red': 0, 'blue': 0})
    blue = obs['blue']
    before = deepcopy(blue)
    original = encode_intelligence_sequence(blue)
    conditioned = encode_reliability_conditioned_sequence(blue, predictor, reasoner)
    assert conditioned.shape[-1] == 67
    assert torch.equal(conditioned[:, :49], original)
    for key in blue:
        np.testing.assert_array_equal(blue[key], before[key])
    logits, _ = ReliabilityConditionedModel(8)(conditioned[None], torch.tensor([len(conditioned)]))
    assert torch.isfinite(logits).all()
    contaminated = dict(blue, hypothesis_label=0)
    with pytest.raises(ValueError, match='Unsafe'):
        encode_reliability_conditioned_sequence(contaminated, predictor, reasoner)


def test_mixture_likelihood_uses_two_mechanisms_without_mutating_reports():
    from deception_rl.intelligence.ride_v2.reliability_conditioned import ReliabilityConditionedLikelihood
    from deception_rl.intelligence.ride_v2.source_region_reliability import SourceRegionReliabilityEstimate
    from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES
    env = DeceptiveIntelligenceEnv(max_cycles=5)
    obs, _ = env.reset(seed=91)
    obs, *_ = env.step({'red': 0, 'blue': 0})
    blue = obs['blue']
    before = deepcopy(blue)
    mean = np.zeros((NUM_HYPOTHESES, 3, 3, 2))
    for hypothesis in range(NUM_HYPOTHESES):
        mean[hypothesis] = hypothesis / NUM_HYPOTHESES
    likelihood = ReliabilityConditionedLikelihood(mean, np.ones_like(mean)*0.15, np.ones((3,3,2))*0.5, np.ones((3,3,2))*0.3)
    class Predictor:
        def __init__(self, reliable): self.reliable = reliable
        def estimate(self, observation, reasoner):
            history = np.full((1,3,3), self.reliable)
            return SourceRegionReliabilityEstimate(history[-1], history, history.mean(2), np.array([0]), np.zeros((1,3,3,5)))
    compromised_logits = likelihood.logits(blue, Predictor(0), None)
    reliable_logits = likelihood.logits(blue, Predictor(1), None)
    assert np.ptp(compromised_logits) < np.ptp(reliable_logits)
    assert np.isfinite(compromised_logits).all()
    for key in blue:
        np.testing.assert_array_equal(blue[key], before[key])


def test_fixed_cell_priors_use_training_visible_cells_only():
    from deception_rl.intelligence.ride_v2.fixed_reliability import FixedCellReliability
    labels=np.ones((3,3),dtype=int)
    visible=np.zeros((3,3),dtype=bool)
    visible[1,2]=True
    fixed=FixedCellReliability.fit([{'reported_cells':visible,'compromised_cells':labels}])
    assert fixed.reliable[1,2] == pytest.approx(1/3)
    assert fixed.reliable[0,0] == 0.5
    env=DeceptiveIntelligenceEnv(max_cycles=5)
    obs,_=env.reset(seed=12)
    estimate=fixed.estimate(obs['blue'])
    np.testing.assert_array_equal(estimate.current,fixed.reliable)


def test_formal_rcl_runner_smoke_and_overwrite_protection(tmp_path):
    from experiments.run_intelligence_rcl_formal import run
    import json
    destination=tmp_path/'formal_smoke'
    run(destination,seeds=(3901,),episodes=1,epochs=1,hidden_size=8,train_episodes=1,validation_episodes=1,max_cycles=5,smoke=True)
    audit=json.loads((destination/'information_boundary_audit.json').read_text())
    assert audit['total_episodes']==56
    assert audit['source_files_unchanged']
    assert not audit['terminal_truth_used_by_policy']
    assert (destination/'neural_seed3901.pt').exists()
    assert (destination/'rcl_seed3901.json').exists()
    with pytest.raises(ValueError,match='overwrite'):
        run(destination)
