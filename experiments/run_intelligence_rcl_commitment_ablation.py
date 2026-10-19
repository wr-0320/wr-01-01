"""Select control settings on seen validation scenes, then evaluate frozen RCL."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES
from deception_rl.intelligence.ride_v2.fixed_reliability import FixedCellReliability
from deception_rl.intelligence.ride_v2.reliability_conditioned import ReliabilityConditionedLikelihood,ReliabilityLikelihoodController
from deception_rl.intelligence.ride_v2.commitment import FrozenRCLCommitment
from experiments.run_intelligence_ride import RED_FAMILIES,evaluate_episode
from experiments.run_intelligence_formal_evaluation import summarize_group
from experiments.run_intelligence_rcl_formal import source_manifest,METRICS
from experiments.run_intelligence_ride_v2_headroom import directory_manifest,write_csv

ROOT=Path(__file__).resolve().parents[1]
PARAMETERS=ROOT/'outputs/intelligence_rcl_formal_evaluation'
DEFAULT_OUTPUT=ROOT/'outputs/intelligence_rcl_commitment_ablation'
SEEDS=(4101,4102,4103,4104,4105)
METHODS=('fixed_4','fixed_validated','confidence_threshold','entropy_gate','entropy_verify','risk_passive','risk_aware')
CANDIDATES={
    'fixed_4':[{'mode':'fixed','commit_step':4}],
    'fixed_validated':[{'mode':'fixed','commit_step':t} for t in range(2,9)],
    'confidence_threshold':[{'mode':'confidence','threshold':x} for x in (.75,.85,.90,.95)],
    'entropy_gate':[{'mode':'entropy','threshold':x} for x in (.15,.25,.35,.45)],
    'entropy_verify':[{'mode':'entropy_verify','threshold':x} for x in (.15,.25,.35,.45)],
    'risk_passive':[{'mode':'risk_passive','cost_weight':x} for x in (.5,1.,2.)],
    'risk_aware':[{'mode':'risk','cost_weight':x} for x in (.5,1.,2.)],
}


def load_inference(model_seed):
    payload=json.loads((PARAMETERS/f'rcl_seed{model_seed}.json').read_text())
    model=ReliabilityConditionedLikelihood(*(np.asarray(payload[key]) for key in ('reliable_mean','reliable_scale','compromised_mean','compromised_scale')),temperature=payload['mixture_temperature'])
    prior=FixedCellReliability(np.asarray(payload['fixed_reliability']))
    return ReliabilityLikelihoodController(model,prior,None)


def controller(model_seed,name,parameters,rollouts=32):
    value=FrozenRCLCommitment(load_inference(model_seed),**parameters,rollouts=rollouts)
    value.name=name
    return value


def evaluate(control,family,hypothesis,seed):
    control.seed=int(seed)
    return evaluate_episode(control,family,hypothesis,seed=seed,max_cycles=10)


def summarize(rows):
    seeds=sorted({r['model_seed'] for r in rows})
    scopes=('overall','seen','unseen',*RED_FAMILIES)
    perseed=[]
    for seed in seeds:
        for method in METHODS:
            group=[r for r in rows if r['model_seed']==seed and r['controller']==method]
            for scope in scopes:
                selected=group if scope=='overall' else [r for r in group if (r['family_scope']==scope if scope in {'seen','unseen'} else r['deception_family']==scope)]
                perseed.append({'model_seed':seed,'method':method,'scope':scope,**{k:v for k,v in summarize_group(selected).items() if k in METRICS}})
    lookup={(r['model_seed'],r['method'],r['scope']):r for r in perseed}
    summary=[]
    for method in METHODS:
        for scope in scopes:
            result={'method':method,'scope':scope,'seeds':len(seeds)}
            for metric in METRICS:
                values=np.asarray([lookup[s,method,scope][metric] for s in seeds])
                half=2.776*values.std(ddof=1)/math.sqrt(5) if len(seeds)==5 else float('nan')
                result[metric]=float(values.mean());result[metric+'_ci95_low']=float(values.mean()-half);result[metric+'_ci95_high']=float(values.mean()+half)
            summary.append(result)
    paired=[]
    for comparator in METHODS:
        if comparator=='risk_aware':continue
        for scope in scopes:
            for metric in METRICS:
                values=np.asarray([lookup[s,'risk_aware',scope][metric]-lookup[s,comparator,scope][metric] for s in seeds])
                half=2.776*values.std(ddof=1)/math.sqrt(5) if len(seeds)==5 else float('nan')
                paired.append({'comparator':comparator,'scope':scope,'metric':metric,'difference':float(values.mean()),'ci95_low':float(values.mean()-half),'ci95_high':float(values.mean()+half)})
    return perseed,summary,paired


def run(output_dir=DEFAULT_OUTPUT,*,seeds=SEEDS,validation_per_family=12,test_per_family=100,rollouts=32,smoke=False):
    destination=Path(output_dir)
    if destination.exists():raise ValueError('Refusing to overwrite existing outputs')
    if not smoke and (tuple(seeds)!=SEEDS or validation_per_family!=12 or test_per_family!=100 or rollouts!=32):raise ValueError('Frozen protocol')
    frozen={p.name:p for p in (ROOT/'outputs').iterdir() if p.is_dir() and (p.name.startswith('intelligence_') or p.name=='thesis_rcl_revision')}
    before={name:directory_manifest(path) for name,path in frozen.items()}
    sources=source_manifest()
    destination.mkdir(parents=True)
    config={'protocol':'Frozen-RCL Controller Ablation v1','model_seeds':[3001+i for i in range(len(seeds))],'test_seeds':list(seeds),'validation_families':['honest','concealment','decoy','delayed'],'validation_episodes_per_family':validation_per_family,'test_episodes_per_family':test_per_family,'candidate_grids':CANDIDATES,'selection_objective':'minimum mean simulator regret on separate seen validation cases; deterministic grid-order tie break','model_refitted':False,'rollouts':rollouts,'primary_comparator':'fixed_validated','support_rule':'risk-aware regret paired 95% upper < 0 vs fixed_validated; accuracy difference >= -0.02; all other comparisons secondary','proxy_model':'one-step independent Gaussian mixture; fixed coverage [0.82,0.90,0.70]; verification residual contamination .30, activity-noise factor .35, confidence offset .10; tracking residual contamination .65, activity-noise factor .65, confidence offset .04, coverage +.16; track persistence beyond one step not modeled; actual RCL inference unchanged','costs':{'wait':.15,'verify':.45,'track':.30},'minimum_decision_step':2,'deadline':9,'all_cells_completed_before_test_access':True,'expected_test_episodes':len(seeds)*len(METHODS)*7*test_per_family,'smoke_only':smoke}
    (destination/'protocol.json').write_text(json.dumps(config,indent=2))
    (destination/'source_manifest.json').write_text(json.dumps(sources,indent=2))
    selected=[];validation_rows=[]
    # Select every control setting before reading any test-family outcome.
    for index,seed in enumerate(seeds):
        model_seed=3001+index
        for name,grid in CANDIDATES.items():
            candidates=[]
            for candidate_index,settings in enumerate(grid):
                control=controller(model_seed,name,settings,rollouts)
                scores=[]
                for fi,family in enumerate(config['validation_families']):
                    for episode in range(validation_per_family):
                        hypothesis=(episode+fi)%NUM_HYPOTHESES
                        row=evaluate(control,family,hypothesis,2_000_000_000+seed*1009+fi*100003+episode)
                        scores.append(row['regret'])
                mean=float(np.mean(scores))
                candidates.append((mean,candidate_index,settings))
                validation_rows.append({'model_seed':model_seed,'controller':name,'candidate_index':candidate_index,'validation_regret':mean,'parameters':json.dumps(settings,sort_keys=True)})
            score,_,parameters=min(candidates,key=lambda x:(x[0],x[1]))
            selected.append({'model_seed':model_seed,'test_seed':seed,'controller':name,'parameters':parameters,'validation_regret':score})
        print(f'Validation complete model={model_seed}',flush=True)
    (destination/'selected_controllers.json').write_text(json.dumps(selected,indent=2))
    write_csv(destination/'validation_grid.csv',validation_rows)
    selections={(r['model_seed'],r['controller']):r['parameters'] for r in selected}
    rows=[]
    for index,seed in enumerate(seeds):
        model_seed=3001+index
        controls={name:controller(model_seed,name,selections[model_seed,name],rollouts) for name in METHODS}
        seed_rows=[]
        for fi,family in enumerate(RED_FAMILIES):
            for episode in range(test_per_family):
                hypothesis=(episode+fi+seed)%NUM_HYPOTHESES
                scenario_seed=2_200_000_000+seed*1009+fi*100003+episode
                for name,control in controls.items():
                    row=evaluate(control,family,hypothesis,scenario_seed)
                    row.update({'model_seed':model_seed,'test_seed':seed})
                    seed_rows.append(row)
            print(f'Test complete model={model_seed} family={family}',flush=True)
        write_csv(destination/f'episodes_seed{model_seed}.csv',seed_rows)
        rows.extend(seed_rows)
    assert len(rows)==config['expected_test_episodes']
    perseed,summary,paired=summarize(rows)
    for filename,values in [('episodes.csv',rows),('per_seed_summary.csv',perseed),('summary.csv',summary),('paired_differences.csv',paired)]:write_csv(destination/filename,values)
    lookup={(r['comparator'],r['scope'],r['metric']):r for r in paired}
    regret=lookup['fixed_validated','overall','regret'];accuracy=lookup['fixed_validated','overall','correct_commitment']
    support=bool(regret['ci95_high']<0 and accuracy['difference']>=-.02)
    claim={'risk_aware_primary_supported':support,'primary_regret_difference':regret,'primary_accuracy_difference':accuracy,'accuracy_tolerance_is_mean_only_not_formal_noninferiority':True,'no_parameter_tuning_after_test':True}
    (destination/'claim_audit.json').write_text(json.dumps(claim,indent=2))
    audit={f'{name}_unchanged':before[name]==directory_manifest(path) for name,path in frozen.items()}
    audit.update({'source_unchanged':sources==source_manifest(),'same_rcl_parameters_all_controllers':True,'runtime_simulator_access':False,'runtime_truth_used':False,'policy_inputs_audited':sum(r['boundary_checks'] for r in rows),'test_episodes':len(rows)})
    assert all(v for k,v in audit.items() if k.endswith('_unchanged'))
    (destination/'audit.json').write_text(json.dumps(audit,indent=2))
    lines=['# Frozen-RCL control ablation','', 'Every controller loads the same RCL parameters within a model seed; no inference refitting. All settings are chosen on separate seen-family validation scenes before test access. Test families include established designs previously used in development. The risk controller uses an approximate one-step predictive mixture model; it is not exact VOI, simulator lookahead, or an optimal policy. The stronger fixed baseline is selected on validation, avoiding an arbitrary fixed-step comparator.','', '| Controller | Correct commitment | False commitment | Regret | Latency | Cost | Brier |','|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['scope']=='overall':lines.append(f"| {r['method']} | {100*r['correct_commitment']:.1f}% | {100*r['false_commitment']:.1f}% | {r['regret']:.3f} | {r['decision_latency']:.2f} | {r['information_cost']:.2f} | {r['hypothesis_brier']:.3f} |")
    lines.extend(['',f'Primary support rule satisfied: **{support}**.',f"Risk-aware minus validation-selected fixed regret: {regret['difference']:+.3f}, 95% seed CI [{regret['ci95_low']:+.3f}, {regret['ci95_high']:+.3f}].",'Accuracy guard uses mean difference only, not a formal noninferiority test. Secondary controller, source-family, and trade-off comparisons are descriptive and intervals are not multiplicity corrected. Calibration of frozen RCL may change with action/stop-policy distribution; equal inference does not imply equal observed trajectories. Raw false-commitment and no-commit rates must be checked along with regret. A negative primary result must remain negative; do not select a different comparator after inspecting results.'])
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(claim,indent=2),flush=True)
    return claim

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    run(parser.parse_args().output_dir)
