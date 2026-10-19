from copy import deepcopy
import numpy as np
from deception_rl.envs.deceptive_intelligence import DeceptiveIntelligenceEnv,encode_commit_action
from deception_rl.intelligence.ride_v2.commitment import FrozenRCLCommitment
from experiments.run_intelligence_rcl_commitment_ablation import load_inference


def test_control_inference_is_identical_and_input_immutable():
    env=DeceptiveIntelligenceEnv()
    obs,_=env.reset(seed=17)
    for _ in range(4):obs,*_=env.step({'red':0,'blue':0})
    blue=obs['blue'];before=deepcopy(blue)
    controls=[FrozenRCLCommitment(load_inference(3001),mode) for mode in ['fixed','confidence','entropy','entropy_verify','risk','risk_passive']]
    beliefs=[c.inference.infer(blue)['hypothesis_belief'] for c in controls]
    for b in beliefs:np.testing.assert_array_equal(b,beliefs[0])
    for c in controls:
        action=c.act(blue)
        assert blue['action_mask'][action]
    for key in blue:np.testing.assert_array_equal(blue[key],before[key])


def test_deadline_forces_commit_and_rollouts_repeat_without_simulator_access():
    env=DeceptiveIntelligenceEnv()
    obs,_=env.reset(seed=22)
    for _ in range(9):obs,*_=env.step({'red':0,'blue':0})
    control=FrozenRCLCommitment(load_inference(3001),'risk',seed=37)
    action=control.act(obs['blue'])
    assert action>=encode_commit_action(0)
    b=np.asarray(control.info['hypothesis_belief'])
    a=control._forecast_risks(obs['blue'],b,8)
    assert a==control._forecast_risks(obs['blue'],b,8)
    assert all(np.isfinite(x) for x in a.values())


def test_commitment_runner_separates_validation_and_test(tmp_path):
    import json
    from experiments.run_intelligence_rcl_commitment_ablation import run
    destination=tmp_path/'control'
    run(destination,seeds=(4901,),validation_per_family=1,test_per_family=1,rollouts=4,smoke=True)
    protocol=json.loads((destination/'protocol.json').read_text())
    audit=json.loads((destination/'audit.json').read_text())
    assert protocol['all_cells_completed_before_test_access']
    assert audit['test_episodes']==49
    assert audit['same_rcl_parameters_all_controllers']


def test_clean_head_ablation_runner_marks_untrained_head_unavailable(tmp_path):
    import csv,json
    from experiments.run_intelligence_clean_dual_head_ablation import run
    destination=tmp_path/'heads'
    result=run(destination,seeds=(4902,),episodes=1,epochs=1,smoke=True)
    assert result['episodes']==14
    rows=list(csv.DictReader((destination/'summary.csv').open()))
    assert all(r['deception_brier']=='' for r in rows if r['method']=='goal_only')
    assert json.loads((destination/'audit.json').read_text())['source_unchanged']
