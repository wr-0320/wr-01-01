"""Matched fresh-seed development test of true/induced-region coupling."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from deception_rl.agents.intelligence_policies import DecoyRedPolicy
from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES,DeceptiveIntelligenceEnv,encode_commit_action,decode_red_action
from deception_rl.intelligence.ride import HonestIntelligenceReasoner
from deception_rl.intelligence.ride_v2.fixed_reliability import FixedCellReliability
from deception_rl.intelligence.ride_v2.oracle import OracleSourceReliability
from deception_rl.intelligence.ride_v2.reliability_conditioned import ReliabilityConditionedLikelihood,ReliabilityLikelihoodController
from deception_rl.intelligence.ride_v2.intent_coupled import IntentCoupledLikelihood
from experiments.run_intelligence_ride import TRAIN_FAMILIES,RED_FAMILIES,collect_prefix_records,_collection_action,_controller_info
from experiments.run_intelligence_ride_v2_headroom import DEVELOPMENT_RED_FAMILIES,directory_manifest,write_csv
from experiments.run_intelligence_reliability_conditioned_inference import collect_belief_cases,fit_likelihood_temperature
from experiments.run_intelligence_rcl_formal import SingleReliableLikelihood,source_manifest,METRICS
from experiments.run_intelligence_formal_evaluation import summarize_group

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/intelligence_intent_coupled_development'
SEEDS=(6101,6102,6103)
METHODS=('single_mechanism','standard_rcl','surface_independent','intent_coupled_shared_temperature','intent_coupled_calibrated')
FIT_FAMILIES=('honest','concealment','decoy','delayed_dev_onset2')


class ReverseDecoy(DecoyRedPolicy):
    def _decoy_region(self,observation):return (int(observation['main_region'])+2)%3

STRESS={'reversed_decoy':lambda:ReverseDecoy(False),'reversed_multi_source':lambda:ReverseDecoy(True)}


def collect_fit(seed,episodes):
    rows=[];oracle=OracleSourceReliability()
    for fi,family in enumerate(FIT_FAMILIES):
        for hypothesis in range(NUM_HYPOTHESES):
            for episode in range(episodes):
                env=DeceptiveIntelligenceEnv(max_cycles=10)
                red=DEVELOPMENT_RED_FAMILIES[family]();red.reset()
                observations,_=env.reset(seed=seed+fi*100003+hypothesis*1009+episode,options={'hypothesis':hypothesis})
                previous=None
                while env.agents:
                    step=int(observations['blue']['current_step'][0])
                    red_action=red.act(observations['red'])
                    diagnostic=oracle.inspect(env,red_action,previous)
                    kind,region=decode_red_action(diagnostic.effective_red_action)
                    surface=3 if kind=='concealment' else int(region) if kind in {'decoy','multi_source_deception'} else None
                    action=encode_commit_action(0) if step>=9 else _collection_action(observations['blue'],step)
                    observations,*_=env.step({'red':red_action,'blue':action})
                    rows.append({'episode_id':f'{family}:{hypothesis}:{episode}','step':step,'hypothesis_label':hypothesis,'surface_label':surface,'reported_values':np.asarray(observations['blue']['reports'])[step].copy(),'reported_cells':np.asarray(observations['blue']['report_mask'],dtype=bool)[step].copy(),'compromised_cells':diagnostic.compromised_cells.astype(int)})
                    previous=diagnostic
    return rows


def evaluate(controller,family,hypothesis,seed):
    from deception_rl.intelligence.adapter import audit_blue_observation
    factory=RED_FAMILIES[family] if family in RED_FAMILIES else STRESS[family]
    env=DeceptiveIntelligenceEnv(max_cycles=10);red=factory();red.reset();controller.reset()
    observations,_=env.reset(seed=seed,options={'hypothesis':hypothesis})
    metrics=None;checks=0
    while env.agents:
        audit_blue_observation(observations['blue']);checks+=1
        action=controller.act(observations['blue'])
        observations,_,_,_,infos=env.step({'red':red.act(observations['red']),'blue':action})
        if 'episode_metrics' in infos['blue']:metrics=infos['blue']['episode_metrics']
    info=_controller_info(controller);belief=np.asarray(info['hypothesis_belief'])
    target=np.eye(NUM_HYPOTHESES)[hypothesis]
    return {'controller':controller.name,'deception_family':family,'family_scope':'stress' if family in STRESS else 'seen' if family in TRAIN_FAMILIES else 'unseen','hypothesis':hypothesis,'seed':seed,'interception':metrics['decision_correct'],'false_commitment':int(not metrics['timed_out'] and not metrics['decision_correct']),'timed_out':metrics['timed_out'],'decision_latency':metrics['decision_step'],'information_cost':metrics['blue_information_cost'],'regret':metrics['decision_regret'],'confidence':float(belief.max()),'hypothesis_brier':float(np.sum((belief-target)**2)),'hypothesis_nll':float(-np.log(max(belief[hypothesis],1e-8))),'deception_label':int(family!='honest'),'deception_probability':0.,'deception_brier':float(family!='honest'),'deception_nll':0.,'boundary_checks':checks}


def analyze(rows):
    seeds=sorted({r['training_seed'] for r in rows})
    scopes=('overall','seen','unseen','stress',*RED_FAMILIES,*STRESS)
    perseed=[]
    for seed in seeds:
        for method in METHODS:
            group=[r for r in rows if r['training_seed']==seed and r['controller']==method]
            for scope in scopes:
                selected=[r for r in group if (r['family_scope']!='stress' if scope=='overall' else r['family_scope']==scope if scope in {'seen','unseen','stress'} else r['deception_family']==scope)]
                perseed.append({'seed':seed,'method':method,'scope':scope,**{k:v for k,v in summarize_group(selected).items() if k in METRICS}})
    lookup={(r['seed'],r['method'],r['scope']):r for r in perseed}
    summary=[]
    for method in METHODS:
        for scope in scopes:
            result={'method':method,'scope':scope}
            for metric in METRICS:
                values=np.array([lookup[s,method,scope][metric] for s in seeds])
                half=4.303*values.std(ddof=1)/math.sqrt(3) if len(seeds)==3 else float('nan')
                result[metric]=float(values.mean());result[metric+'_ci95_low']=float(values.mean()-half);result[metric+'_ci95_high']=float(values.mean()+half)
            summary.append(result)
    contrasts=[]
    for method in METHODS[-2:]:
        for comparator in ('standard_rcl','surface_independent'):
            for scope in scopes:
                for metric in METRICS:
                    values=np.array([lookup[s,method,scope][metric]-lookup[s,comparator,scope][metric] for s in seeds])
                    half=4.303*values.std(ddof=1)/math.sqrt(3) if len(seeds)==3 else float('nan')
                    contrasts.append({'method':method,'comparator':comparator,'scope':scope,'metric':metric,'difference':float(values.mean()),'ci95_low':float(values.mean()-half),'ci95_high':float(values.mean()+half)})
    return perseed,summary,contrasts


def run(output_dir=OUTPUT,*,seeds=SEEDS,episodes=100,train_episodes=10,smoke=False):
    destination=Path(output_dir)
    if destination.exists():raise ValueError('Refusing to overwrite outputs')
    if not smoke and (tuple(seeds)!=SEEDS or episodes!=100 or train_episodes!=10):raise ValueError('Frozen protocol')
    frozen={p.name:p for p in (ROOT/'outputs').iterdir() if p.is_dir() and (p.name.startswith('intelligence_') or p.name.startswith('thesis_'))}
    before={n:directory_manifest(p) for n,p in frozen.items()};sources=source_manifest()
    destination.mkdir(parents=True)
    config={'protocol':'True/Induced Region Coupling Development v1','seeds':list(seeds),'fit_families':list(FIT_FAMILIES),'primary_families':list(RED_FAMILIES),'separate_policy_shift_stress':list(STRESS),'methods':list(METHODS),'episodes_per_method_family_seed':episodes,'expected_test_episodes':len(seeds)*9*episodes*5,'primary_model':'intent_coupled_shared_temperature','primary_controls':['standard_rcl','surface_independent'],'primary_gate':'accuracy paired lower CI > 0 against both controls, and Brier/regret lower against both; task gains required, label fit alone insufficient','reliable_emissions_frozen_across_variants':True,'cell_priors_frozen_across_variants':True,'primary_temperature_shared_with_standard':True,'transition_smoothing':1.,'component_scale_floor':.08,'surface_states':['decoy_north','decoy_central','decoy_south','concealment'],'component_fit':'all observed cells during presentation-event cycles; not only targeted cells; unconditional-surface control uses identical components','hypothesis_coupling':'estimated from offline training H and effective-red-action-derived surface labels; never from test labels','caveat':'surface states encode induced region, not a full fictitious operational hypothesis; relationship can be red-policy specific','event_scope':'shared across cells within cycle; independent across cycles','validation_cases_per_seen_family':12 if not smoke else 1,'no_test_tuning':True,'smoke_only':smoke}
    (destination/'protocol.json').write_text(json.dumps(config,indent=2));(destination/'source_manifest.json').write_text(json.dumps(sources,indent=2))
    fitted=[]
    for seed in seeds:
        samples=collect_fit(4_000_000_000+seed*1009,train_episodes)
        fixed=FixedCellReliability.fit(samples)
        base=ReliabilityConditionedLikelihood.fit(samples)
        cases=collect_belief_cases(FIT_FAMILIES,12 if not smoke else 1,seed=4_100_000_000+seed*1009,max_cycles=10)
        standard=fit_likelihood_temperature(base,cases,fixed,None)
        single=SingleReliableLikelihood(base.reliable_mean,base.reliable_scale,base.compromised_mean,base.compromised_scale)
        single=fit_likelihood_temperature(single,cases,fixed,None)
        coupled=IntentCoupledLikelihood.fit(samples,standard,fixed)
        independent=IntentCoupledLikelihood(standard,coupled.reliable,coupled.surface_mean,coupled.surface_scale,coupled.transition,coupled=False)
        recalibrated=fit_likelihood_temperature(coupled.with_temperature(1.),cases,fixed,None)
        models={'single_mechanism':single,'standard_rcl':standard,'surface_independent':independent,'intent_coupled_shared_temperature':coupled,'intent_coupled_calibrated':recalibrated}
        controls={name:ReliabilityLikelihoodController(model,fixed,None,name=name) for name,model in models.items()}
        params={'reliable_mean':base.reliable_mean.tolist(),'reliable_scale':base.reliable_scale.tolist(),'compromised_mean':base.compromised_mean.tolist(),'compromised_scale':base.compromised_scale.tolist(),'fixed_reliability':fixed.reliable.tolist(),'standard_temperature':standard.temperature,'coupled':coupled.to_dict(),'secondary_calibrated':recalibrated.to_dict()}
        (destination/f'parameters_seed{seed}.json').write_text(json.dumps(params,indent=2))
        fitted.append((seed,controls))
        print(f'Fit frozen seed={seed}',flush=True)
    rows=[]
    for seed,controls in fitted:
        seed_rows=[]
        for fi,family in enumerate((*RED_FAMILIES,*STRESS)):
            for episode in range(episodes):
                hypothesis=(episode+fi+seed)%NUM_HYPOTHESES
                scenario_seed=4_200_000_000+seed*1009+fi*100003+episode
                for name,control in controls.items():
                    row=evaluate(control,family,hypothesis,scenario_seed);row['training_seed']=seed;seed_rows.append(row)
            print(f'Test complete seed={seed} family={family}',flush=True)
        write_csv(destination/f'episodes_seed{seed}.csv',seed_rows);rows.extend(seed_rows)
    assert len(rows)==config['expected_test_episodes']
    perseed,summary,contrasts=analyze(rows)
    for filename,values in [('episodes.csv',rows),('per_seed_summary.csv',perseed),('summary.csv',summary),('paired_differences.csv',contrasts)]:write_csv(destination/filename,values)
    lookup={(r['method'],r['scope']):r for r in summary}
    diffs={(r['method'],r['comparator'],r['scope'],r['metric']):r for r in contrasts}
    primary='intent_coupled_shared_temperature'
    gains={control:diffs[primary,control,'overall','correct_commitment'] for control in ('standard_rcl','surface_independent')}
    go=all(d['ci95_low']>0 for d in gains.values()) and all(lookup[primary,'overall'][metric]<lookup[control,'overall'][metric] for control in gains for metric in ('hypothesis_brier','regret'))
    claim={'decision':'GO' if go else 'NO-GO','primary_accuracy_comparisons':gains,'reverse_policy_shift_accuracy_delta_vs_standard':diffs[primary,'standard_rcl','stress','correct_commitment'],'policy_independent_strategy_claim':False,'new_innovation_claim_established':False,'formal_results_replaced':False}
    (destination/'claim_audit.json').write_text(json.dumps(claim,indent=2))
    audit={n+'_unchanged':before[n]==directory_manifest(p) for n,p in frozen.items()}
    audit.update({'source_unchanged':sources==source_manifest(),'reliable_emissions_shared':True,'cell_contamination_priors_shared':True,'latent_component_parameters_shared_with_unconditional_control':True,'primary_temperature_shared':True,'runtime_H_F_truth_used':False,'test_episodes':len(rows),'policy_inputs_audited':sum(r['boundary_checks'] for r in rows)})
    assert all(v for k,v in audit.items() if k.endswith('_unchanged'))
    (destination/'audit.json').write_text(json.dumps(audit,indent=2))
    lines=['# True/induced-region intent-coupled RCL development','', 'Primary matched controls: standard hypothesis-independent RCL and a latent-surface mixture with the same components but H-independent surface weights. All share reliable emissions, marginal cell contamination rates, primary temperature, fixed commit step and sensing costs. F is an induced region plus a concealment fallback state, not a complete fake operational hypothesis. Surface components use all observed reports during presentation-event cycles; this fitting change is shared by the unconditional control. The conditional relation can encode red-script regularities. Reversed decoy policies are separate stress tests, never included in primary overall figures.','', '| Method | Accuracy | Brier | ECE | NLL | Regret | Latency | Cost |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in METHODS:
        r=lookup[name,'overall'];lines.append(f"| {name} | {100*r['correct_commitment']:.1f}% | {r['hypothesis_brier']:.3f} | {r['hypothesis_ece']:.3f} | {r['hypothesis_nll']:.3f} | {r['regret']:.3f} | {r['decision_latency']:.2f} | {r['information_cost']:.2f} |")
    lines.extend(['',f"Frozen gate: **{claim['decision']}**.",'', '| Comparator | Primary accuracy gain (pp) | Three-seed 95% CI (pp) |','|---|---:|---:|'])
    for control,d in gains.items():lines.append(f"| {control} | {100*d['difference']:+.2f} | [{100*d['ci95_low']:+.2f},{100*d['ci95_high']:+.2f}] |")
    lines.extend(['','| Family | Standard RCL | H-independent surface | Coupled, shared temperature |','|---|---:|---:|---:|'])
    for family in (*RED_FAMILIES,*STRESS):lines.append('| '+family+' | '+' | '.join(f"{100*lookup[m,family]['correct_commitment']:.1f}%" for m in ['standard_rcl','surface_independent',primary])+' |')
    lines.extend(['', 'Three new development seeds only; no formal confirmation or paper novelty claim. Intervals use seed means and are unadjusted; family/stress comparisons are secondary. Concealment is not a false regional story and has its own fallback state. No temporal policy latent is inferred. A benefit that disappears under policy-rule reversal cannot establish policy-independent strategic reasoning. Positive results would require independent validation and literature comparison before a third-innovation claim.'])
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(claim,indent=2),flush=True)
    return claim

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    run(parser.parse_args().output_dir)
