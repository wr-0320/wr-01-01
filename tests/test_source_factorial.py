from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from deception_rl.envs.deceptive_intelligence import DeceptiveIntelligenceEnv
from deception_rl.intelligence.adapter import encode_intelligence_sequence
from deception_rl.intelligence.ride_v2.source_scope import restrict_observation,restrict_sequence,restrict_samples


def test_source_scope_matches_train_encoding_and_preserves_original():
    env=DeceptiveIntelligenceEnv(max_cycles=5)
    observations,_=env.reset(seed=37)
    for _ in range(2): observations,*_=env.step({'red':0,'blue':0})
    blue=observations['blue'];before=deepcopy(blue)
    for source in range(3):
        scoped=restrict_observation(blue,source)
        assert torch.equal(encode_intelligence_sequence(scoped),restrict_sequence(encode_intelligence_sequence(blue),source))
        others=[s for s in range(3) if s!=source]
        assert not scoped['reports'][:,others].any()
        assert not scoped['report_mask'][:,others].any()
        assert not scoped['source_reliability'][others].any()
        np.testing.assert_array_equal(scoped['reports'][:,source],blue['reports'][:,source])
        for key in blue:np.testing.assert_array_equal(blue[key],before[key])
    with pytest.raises(ValueError):restrict_observation(blue,3)


def test_source_restriction_excludes_unavailable_training_cells():
    rows=[{'reported_cells':np.ones((3,3),dtype=bool),'reported_values':np.ones((3,3,4)),'compromised_cells':np.ones((3,3),dtype=int)}]
    visible=restrict_samples(rows,1)
    assert visible[0]['reported_cells'].sum()==3
    assert not visible[0]['reported_values'][[0,2]].any()
    assert rows[0]['reported_cells'].sum()==9


def test_source_factorial_smoke_and_no_overwrite(tmp_path):
    from experiments.run_intelligence_source_factorial import run
    destination=tmp_path/'factorial'
    result=run(destination,seeds=(3991,),episodes=1,epochs=1,train_episodes=1,validation_episodes=1,hidden_size=8,max_cycles=5,smoke=True)
    audit=json.loads((destination/'audit.json').read_text())
    assert audit['total_episodes']==84
    assert audit['source_files_unchanged']
    assert not result['universal_ordering_claim']
    with pytest.raises(ValueError,match='overwrite'):run(destination)
