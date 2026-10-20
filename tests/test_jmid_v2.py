import csv

import numpy as np

from experiments.jmid_v2.generator import ROTATED_GROUPS, generate_v2, v2_diagnostics
from experiments.jmid_v2.models import fit_v2_models
from experiments.jmid_v2.run import run


def test_v2_controls_are_distinct_and_content_triggered():
    base=generate_v2(25_000,10,censor_strength=0.0,copy_strength=0.0)
    hard=generate_v2(25_000,10,censor_strength=1.0,copy_strength=.85)
    a,b=v2_diagnostics(base),v2_diagnostics(hard)
    assert b["missing_given_raw1"] > b["missing_given_raw0"] + .25
    assert b["within_state_report_correlation"] > a["within_state_report_correlation"] + .45
    assert abs(b["copy_rate"]-.85) < .02


def test_v2_generator_supports_unseen_rules_and_rotated_graph():
    for index,mode in enumerate(("negative","conflict")):
        data=generate_v2(500,30+index,censor_strength=1.0,copy_strength=.85,censor_mode=mode)
        assert data.y.shape==(500,6)
    rotated=generate_v2(500,35,censor_strength=1.0,copy_strength=.85,groups=ROTATED_GROUPS)
    assert rotated.groups==ROTATED_GROUPS


def test_fixed_composition_is_deterministic_and_probabilistic():
    train=generate_v2(900,40,censor_strength=1.0,copy_strength=.85)
    test=generate_v2(120,41,censor_strength=1.0,copy_strength=.85)
    models=fit_v2_models(train)
    assert set(models)=={"M2_missing","M3_dependence","M23_fixed","M4_joint"}
    first=models["M23_fixed"].predict_proba(test.y)
    second=models["M23_fixed"].predict_proba(test.y)
    assert np.array_equal(first,second)
    assert np.isfinite(first).all() and ((first>0)&(first<1)).all()


def test_v2_smoke_is_paired_and_refuses_overwrite(tmp_path):
    output=tmp_path/"v2"
    result=run(output,seeds=(88,),train_episodes=200,test_episodes=250,smoke=True)
    assert result=={"primary_rows":16,"transfer_rows":32}
    with (output/"primary_per_seed.csv").open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==16
    try: run(output,seeds=(88,),smoke=True)
    except FileExistsError: pass
    else: raise AssertionError("runner overwrote existing evidence")

