"""Development test of shared contamination dependence, with frozen matched emissions."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES,DeceptiveIntelligenceEnv,encode_commit_action
from deception_rl.intelligence.ride import HonestIntelligenceReasoner
from deception_rl.intelligence.ride_v2.oracle import OracleSourceReliability
from deception_rl.intelligence.ride_v2.fixed_reliability import FixedCellReliability
from deception_rl.intelligence.ride_v2.reliability_conditioned import ReliabilityConditionedLikelihood,ReliabilityLikelihoodController
from deception_rl.intelligence.ride_v2.correlated_likelihood import CorrelationAwareLikelihood
from experiments.run_intelligence_ride import TRAIN_FAMILIES,RED_FAMILIES,collect_prefix_records,evaluate_episode,_collection_action
from experiments.run_intelligence_rcl_formal import SingleReliableLikelihood,source_manifest,METRICS
from experiments.run_intelligence_reliability_conditioned_inference import collect_belief_cases,fit_likelihood_temperature
from experiments.run_intelligence_ride_v2_reliability import RELIABILITY_TRAIN_FAMILIES,collect_reliability_samples
from experiments.run_intelligence_ride_v2_headroom import directory_manifest,write_csv
from experiments.run_intelligence_formal_evaluation import summarize_group

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=ROOT/'outputs/intelligence_correlated_rcl_development'
SEEDS=(5101,5102,5103)
METHODS=('single_mechanism','independent_rcl','correlated_shared_temperature','correlated_calibrated')


def collect_label_diagnostics(seed,episodes):
    """Separate offline harness using exactly the task-test red family definitions."""
    rows=[]
    oracle=OracleSourceReliability()
    for fi,family in enumerate(RED_FAMILIES):
        for hypothesis in range(NUM_HYPOTHESES):
            for episode in range(episodes):
                env=DeceptiveIntelligenceEnv(max_cycles=10)
                red=RED_FAMILIES[family]();red.reset()
                observations,_=env.reset(seed=seed+fi*100003+hypothesis*1009+episode,options={'hypothesis':hypothesis})
                previous=None
                while env.agents:
                    step=int(observations['blue']['current_step'][0])
                    red_action=red.act(observations['red'])
                    diagnostic=oracle.inspect(env,red_action,previous)
                    action=encode_commit_action(0) if step>=9 else _collection_action(observations['blue'],step)
                    observations,*_=env.step({'red':red_action,'blue':action})
                    rows.append({'deception_family':family,'compromised_cells':diagnostic.compromised_cells.astype(int),'reported_cells':np.asarray(observations['blue']['report_mask'],dtype=bool)[step]})
                    previous=diagnostic
    return rows


def analyze(rows):
    seeds=sorted({r['training_seed'] for r in rows})
    seed_rows=[]
    for seed in seeds:
        for method in METHODS:
            for scope in ('overall','seen','unseen',*RED_FAMILIES):
                selected=[r for r in rows if r['training_seed']==seed and r['controller']==method and (scope=='overall' or (r['family_scope']==scope if scope in {'seen','unseen'} else r['deception_family']==scope))]
                seed_rows.append({'seed':seed,'method':method,'scope':scope,**{k:v for k,v in summarize_group(selected).items() if k in METRICS}})
    lookup={(r['seed'],r['method'],r['scope']):r for r in seed_rows}
    summary=[]
    for method in METHODS:
        for scope in ('overall','seen','unseen',*RED_FAMILIES):
            result={'method':method,'scope':scope}
            for metric in METRICS:
                values=np.asarray([lookup[s,method,scope][metric] for s in seeds])
                half=4.303*values.std(ddof=1)/math.sqrt(3) if len(seeds)==3 else float('nan')
                result[metric]=float(values.mean());result[metric+'_ci95_low']=float(values.mean()-half);result[metric+'_ci95_high']=float(values.mean()+half)
            summary.append(result)
    differences=[]
    for method in ('correlated_shared_temperature','correlated_calibrated'):
        for scope in ('overall','seen','unseen',*RED_FAMILIES):
            for metric in METRICS:
                values=np.asarray([lookup[s,method,scope][metric]-lookup[s,'independent_rcl',scope][metric] for s in seeds])
                half=4.303*values.std(ddof=1)/math.sqrt(3) if len(seeds)==3 else float('nan')
                differences.append({'method':method,'scope':scope,'metric':metric,'difference':float(values.mean()),'ci95_low':float(values.mean()-half),'ci95_high':float(values.mean()+half)})
    return seed_rows,summary,differences


def run(output_dir=DEFAULT_OUTPUT,*,seeds=SEEDS,episodes=100,train_episodes=10,smoke=False):
    destination=Path(output_dir)
    if destination.exists():raise ValueError('Refusing to overwrite any output')
    if not smoke and (tuple(seeds)!=SEEDS or episodes!=100 or train_episodes!=10):raise ValueError('Frozen development protocol')
    frozen={p.name:p for p in (ROOT/'outputs').iterdir() if p.is_dir() and (p.name.startswith('intelligence_') or p.name.startswith('thesis_'))}
    before={n:directory_manifest(p) for n,p in frozen.items()};sources=source_manifest()
    destination.mkdir(parents=True)
    config={'protocol':'Shared Regional Contamination Development v1','seeds':list(seeds),'families':list(RED_FAMILIES),'methods':list(METHODS),'episodes_per_seed_family_method':episodes,'expected_test_episodes':len(seeds)*7*episodes*4,'fitting_families':list(RELIABILITY_TRAIN_FAMILIES),'dependency_label':'at least two sources targeted in the same region and report cycle; offline only','shared_emissions':True,'identical_unconditional_cell_rates':True,'primary_method':'correlated_shared_temperature','primary_gate':'paired accuracy lower CI > 0 and hypothesis Brier lower, regret lower, and joint-label NLL lower than independent RCL','secondary_practical_target':'at least 0.03 mean accuracy gain; not a gate chosen after test','dependency_strength':1.,'marginal_projection':'c_s(z)=q_s+(z-pi)*a_s; positive slopes fitted from training labels and capped to legal probabilities','shared_temperature_for_primary':True,'secondary_temperature':'separately calibrated on same seen-family validation cases','minimum_calibration_cases_per_seen_family':12 if not smoke else 1,'horizon':10,'commit_step':4,'cross_region_dependency':False,'temporal_dependency':False,'novelty_claim_verified':False,'no_test_tuning':True,'smoke_only':smoke}
    (destination/'protocol.json').write_text(json.dumps(config,indent=2));(destination/'source_manifest.json').write_text(json.dumps(sources,indent=2))
    fitted=[]
    for seed in seeds:
        prototype_records=collect_prefix_records(TRAIN_FAMILIES,8 if not smoke else 1,seed=3_000_000_000+seed*1009,split=f'correlation_reasoner_{seed}',max_cycles=10)
        reasoner=HonestIntelligenceReasoner(prototype_records)
        samples=collect_reliability_samples(RELIABILITY_TRAIN_FAMILIES,train_episodes,seed=3_100_000_000+seed*1009,reasoner=reasoner,split=f'correlation_fit_{seed}',max_cycles=10)
        fixed=FixedCellReliability.fit(samples)
        emissions=ReliabilityConditionedLikelihood.fit(samples)
        calibration_cases=collect_belief_cases(RELIABILITY_TRAIN_FAMILIES,12 if not smoke else 1,seed=3_200_000_000+seed*1009,max_cycles=10)
        independent=fit_likelihood_temperature(emissions,calibration_cases,fixed,reasoner)
        single=SingleReliableLikelihood(emissions.reliable_mean,emissions.reliable_scale,emissions.compromised_mean,emissions.compromised_scale)
        single=fit_likelihood_temperature(single,calibration_cases,fixed,reasoner)
        correlated=CorrelationAwareLikelihood.fit_dependency(samples,independent,fixed)
        null=CorrelationAwareLikelihood.fit_dependency(samples,independent,fixed,strength=0.)
        for row in calibration_cases:
            assert np.allclose(null.logits(row['observation']),independent.logits(row['observation'],fixed,reasoner),atol=1e-9,rtol=0)
        calibrated=fit_likelihood_temperature(correlated.with_temperature(1.),calibration_cases,fixed,reasoner)
        models={'single_mechanism':single,'independent_rcl':independent,'correlated_shared_temperature':correlated,'correlated_calibrated':calibrated}
        controllers={name:ReliabilityLikelihoodController(model,fixed,reasoner,name=name) for name,model in models.items()}
        parameters={'independent_temperature':independent.temperature,'single_temperature':single.temperature,'base_emissions':{key:getattr(emissions,key).tolist() for key in ('reliable_mean','reliable_scale','compromised_mean','compromised_scale')},'fixed_reliability':fixed.reliable.tolist(),'correlated_primary':correlated.to_dict(),'correlated_calibrated':calibrated.to_dict(),'null_collapse_verified':True}
        (destination/f'parameters_seed{seed}.json').write_text(json.dumps(parameters,indent=2))
        fitted.append((seed,reasoner,fixed,correlated,controllers))
        print(f'Fit frozen seed={seed}',flush=True)
    rows=[];joint_rows=[]
    for seed,reasoner,fixed,correlated,controllers in fitted:
        # Privileged diagnostic labels remain exclusively in this offline harness.
        label_cases=collect_label_diagnostics(3_300_000_000+seed*1009,3 if not smoke else 1)
        for family in ('overall',*RED_FAMILIES):
            selected=label_cases if family=='overall' else [r for r in label_cases if r['deception_family']==family]
            independent_nll=[];joint_nll=[]
            q=1.-fixed.reliable
            for row in selected:
                label=np.asarray(row['compromised_cells']);mask=np.asarray(row['reported_cells'])
                independent_nll.append(-float(((label*np.log(q)+(1.-label)*np.log1p(-q))*mask).sum()))
                joint_nll.append(-correlated.contamination_log_probability(row))
            joint_rows.append({'seed':seed,'family':family,'report_cycles':len(selected),'independent_joint_label_nll':float(np.mean(independent_nll)),'correlated_joint_label_nll':float(np.mean(joint_nll))})
        seed_rows=[]
        for fi,family in enumerate(RED_FAMILIES):
            for episode in range(episodes):
                hypothesis=(episode+seed+fi)%NUM_HYPOTHESES
                scenario_seed=3_400_000_000+seed*1009+fi*100003+episode
                for method,control in controllers.items():
                    row=evaluate_episode(control,family,hypothesis,seed=scenario_seed,max_cycles=10)
                    row['training_seed']=seed
                    seed_rows.append(row)
            print(f'Test complete seed={seed} family={family}',flush=True)
        write_csv(destination/f'episodes_seed{seed}.csv',seed_rows)
        rows.extend(seed_rows)
    assert len(rows)==config['expected_test_episodes']
    seed_rows,summary,differences=analyze(rows)
    for filename,values in [('episodes.csv',rows),('per_seed_summary.csv',seed_rows),('summary.csv',summary),('paired_differences.csv',differences),('joint_label_diagnostics.csv',joint_rows)]:write_csv(destination/filename,values)
    lookup={(r['method'],r['scope']):r for r in summary}
    primary=lookup['correlated_shared_temperature','overall'];baseline=lookup['independent_rcl','overall']
    diff=next(r for r in differences if r['method']=='correlated_shared_temperature' and r['scope']=='overall' and r['metric']=='correct_commitment')
    j=[r for r in joint_rows if r['family']=='overall']
    label_gain=float(np.mean([r['independent_joint_label_nll']-r['correlated_joint_label_nll'] for r in j]))
    go=bool(diff['ci95_low']>0 and primary['hypothesis_brier']<baseline['hypothesis_brier'] and primary['regret']<baseline['regret'] and label_gain>0)
    claim={'decision':'GO' if go else 'NO-GO','primary_accuracy_difference':diff,'primary_brier_difference':primary['hypothesis_brier']-baseline['hypothesis_brier'],'primary_regret_difference':primary['regret']-baseline['regret'],'offline_joint_label_nll_reduction':label_gain,'novel_method_claim_established':False,'formal_results_replaced':False}
    (destination/'claim_audit.json').write_text(json.dumps(claim,indent=2))
    audit={name+'_unchanged':before[name]==directory_manifest(path) for name,path in frozen.items()}
    audit.update({'source_unchanged':sources==source_manifest(),'cell_marginals_preserved':True,'emission_parameters_shared':True,'primary_temperature_shared':True,'zero_dependency_equals_independent_verified':True,'runtime_mechanism_labels_used':False,'runtime_original_observation_modified':False,'total_test_episodes':len(rows),'policy_inputs_audited':sum(r['boundary_checks'] for r in rows)})
    assert all(v for k,v in audit.items() if k.endswith('_unchanged'))
    (destination/'audit.json').write_text(json.dumps(audit,indent=2))
    lines=['# Correlation-aware RCL development','', 'A shared binary contamination event is introduced per region and report cycle. Emission distributions and unconditional cell rates match independent RCL exactly. The primary comparison shares the independent temperature; separately recalibrated correlation-aware inference is secondary. Source correlation is modeled, but temporal/cross-region coupling is not. Offline fitting labels indicate simulator intervention targets, not guaranteed report falsity.','', '| Method | Correct commitment | Brier | ECE | NLL | Regret | Latency | Cost |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in METHODS:
        r=lookup[name,'overall'];lines.append(f"| {name} | {100*r['correct_commitment']:.1f}% | {r['hypothesis_brier']:.3f} | {r['hypothesis_ece']:.3f} | {r['hypothesis_nll']:.3f} | {r['regret']:.3f} | {r['decision_latency']:.2f} | {r['information_cost']:.2f} |")
    lines.extend(['',f"Primary accuracy gain: {100*diff['difference']:+.2f} pp, 95% three-seed interval [{100*diff['ci95_low']:+.2f}, {100*diff['ci95_high']:+.2f}] pp.",f'Offline correlated-label NLL reduction: {label_gain:+.4f} per report cycle.',f"Frozen development gate: **{claim['decision']}**.",'', '| Family | Independent | Correlated, same temperature | Correlated, recalibrated |','|---|---:|---:|---:|'])
    for family in RED_FAMILIES:lines.append('| '+family+' | '+' | '.join(f"{100*lookup[m,family]['correct_commitment']:.1f}%" for m in METHODS[1:])+' |')
    lines.extend(['', 'These are three fresh development seeds (5101–5103), not a formal confirmation. No seen/unseen family design or algorithm parameter was changed after test inspection. Confidence intervals are unadjusted and secondary family comparisons are descriptive. Better offline joint label likelihood need not imply better hypothesis inference. A positive result would still require literature review before a novelty claim, and frozen held-out confirmation before paper-level performance claims.'])
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(claim,indent=2),flush=True)
    return claim

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    run(parser.parse_args().output_dir)
