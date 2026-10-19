"""Paired source-count × inference-method study; preserves all older artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES,NUM_SOURCES
from deception_rl.intelligence.ride import HonestIntelligenceReasoner
from deception_rl.intelligence.ride_v2.fixed_reliability import FixedCellReliability
from deception_rl.intelligence.ride_v2.reliability_conditioned import ReliabilityConditionedLikelihood,ReliabilityLikelihoodController
from deception_rl.intelligence.ride_v2.source_scope import restrict_observation,restrict_sequence,restrict_samples,SourceScopedController
from experiments.run_intelligence_ride import TRAIN_FAMILIES,RED_FAMILIES,collect_prefix_records,train_model,evaluate_episode
from experiments.run_intelligence_ride_v2_reliability import RELIABILITY_TRAIN_FAMILIES,collect_reliability_samples,_inference
from experiments.run_intelligence_reliability_conditioned_inference import collect_belief_cases,fit_likelihood_temperature
from experiments.run_intelligence_rcl_formal import SingleReliableLikelihood,source_manifest,METRICS
from experiments.run_intelligence_formal_evaluation import summarize_group
from experiments.run_intelligence_ride_v2_headroom import directory_manifest,write_csv

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=ROOT/'outputs/intelligence_source_factorial_development'
SEEDS=(3101,3102,3103,3104,3105)
FAMILIES=tuple(RED_FAMILIES)
CELLS=('single_plain','single_rcl','multi_plain','multi_rcl','single_gaussian','multi_gaussian')


def summarize(rows):
    seed_rows=[]
    for seed in sorted({r['training_seed'] for r in rows}):
        for cell in CELLS:
            for scope in ('overall',*FAMILIES):
                selected=[r for r in rows if r['training_seed']==seed and r['cell']==cell and (scope=='overall' or r['deception_family']==scope)]
                seed_rows.append({'seed':seed,'cell':cell,'scope':scope,'episodes':len(selected),**{k:v for k,v in summarize_group(selected).items() if k in METRICS}})
    lookup={(r['seed'],r['cell'],r['scope']):r for r in seed_rows}
    seeds=sorted({r['training_seed'] for r in rows})
    summary=[]
    for cell in CELLS:
        for scope in ('overall',*FAMILIES):
            selected=[r for r in seed_rows if r['cell']==cell and r['scope']==scope]
            item={'cell':cell,'scope':scope,'seeds':len(seeds),'episodes':sum(r['episodes'] for r in selected)}
            for metric in METRICS:
                values=np.asarray([r[metric] for r in selected])
                half=2.776*values.std(ddof=1)/math.sqrt(5) if len(values)==5 else float('nan')
                item[metric]=float(values.mean())
                item[metric+'_ci95_low']=float(values.mean()-half)
                item[metric+'_ci95_high']=float(values.mean()+half)
            summary.append(item)
    definitions={
        'rcl_gain_single':{'single_rcl':1,'single_plain':-1},
        'rcl_gain_multi':{'multi_rcl':1,'multi_plain':-1},
        'more_sources_plain':{'multi_plain':1,'single_plain':-1},
        'more_sources_rcl':{'multi_rcl':1,'single_rcl':-1},
        'interaction':{'multi_rcl':1,'multi_plain':-1,'single_rcl':-1,'single_plain':1},
        'mixture_gain_single':{'single_rcl':1,'single_gaussian':-1},
        'mixture_gain_multi':{'multi_rcl':1,'multi_gaussian':-1},
        'multi_plain_minus_single_rcl':{'multi_plain':1,'single_rcl':-1},
    }
    contrasts=[]
    for name,weights in definitions.items():
        for scope in ('overall',*FAMILIES):
            for metric in METRICS:
                values=np.asarray([sum(weight*lookup[seed,cell,scope][metric] for cell,weight in weights.items()) for seed in seeds])
                half=2.776*values.std(ddof=1)/math.sqrt(5) if len(values)==5 else float('nan')
                contrasts.append({'contrast':name,'scope':scope,'metric':metric,'difference':float(values.mean()),'ci95_low':float(values.mean()-half),'ci95_high':float(values.mean()+half)})
    source_rows=[]
    for scope in ('overall',*FAMILIES):
        for source in range(NUM_SOURCES):
            for cell in ('single_plain','single_rcl','single_gaussian'):
                selected=[r for r in rows if r['cell']==cell and r['source']==source and (scope=='overall' or r['deception_family']==scope)]
                source_rows.append({'cell':cell,'source':source,'scope':scope,'episodes':len(selected),**{k:v for k,v in summarize_group(selected).items() if k in METRICS}})
    return seed_rows,summary,contrasts,source_rows


def report(destination,summary,contrasts):
    lookup={(r['cell'],r['scope']):r for r in summary}
    diffs={(r['contrast'],r['scope'],r['metric']):r for r in contrasts}
    rows=['# Source count × RCL factorial study','', 'Single-source means retain all regions and all report times for one source. Three source identities are evaluated separately and averaged equally within each training seed. All four main cells share scenario seeds, passive WAIT actions, commit step 4, and information costs. Single-source neural encoders are retrained under exactly the same source masking used for calibration and testing. Unavailable report values, masks, and public source reliability entries are zeroed.','', 'Five new training seeds (3101–3105). 100 episodes per source variant × family × seed. Primary design: four cells; matched Gaussian single-mechanism controls are secondary diagnostics. Ordinary inference is the existing GRU, so the main RCL comparison includes architecture and offline mechanism-supervision differences. No learned reliability model is used.','', '| Source count | Ordinary GRU | Fixed-prior RCL |','|---|---:|---:|']
    rows.append(f"| Single (three-source macro mean) | {100*lookup['single_plain','overall']['correct_commitment']:.1f}% | {100*lookup['single_rcl','overall']['correct_commitment']:.1f}% |")
    rows.append(f"| Multi | {100*lookup['multi_plain','overall']['correct_commitment']:.1f}% | {100*lookup['multi_rcl','overall']['correct_commitment']:.1f}% |")
    rows.extend(['','| Cell | Accuracy | Brier | ECE | NLL | Regret | Latency | Cost |','|---|---:|---:|---:|---:|---:|---:|---:|'])
    for cell in CELLS:
        r=lookup[cell,'overall']
        rows.append(f"| {cell} | {100*r['correct_commitment']:.1f}% | {r['hypothesis_brier']:.3f} | {r['hypothesis_ece']:.3f} | {r['hypothesis_nll']:.3f} | {r['regret']:.2f} | {r['decision_latency']:.2f} | {r['information_cost']:.2f} |")
    rows.extend(['','| Contrast | Accuracy difference (pp) | Five-seed 95% t CI (pp) |','|---|---:|---:|'])
    for name in dict.fromkeys(r['contrast'] for r in contrasts):
        r=diffs[name,'overall','correct_commitment']
        rows.append(f"| {name} | {100*r['difference']:+.2f} | [{100*r['ci95_low']:+.2f}, {100*r['ci95_high']:+.2f}] |")
    values=[lookup[cell,'overall']['correct_commitment'] for cell in ('single_plain','single_rcl','multi_plain','multi_rcl')]
    ordering=bool(all(a<b for a,b in zip(values,values[1:])))
    single=diffs['rcl_gain_single','overall','correct_commitment'];multi=diffs['rcl_gain_multi','overall','correct_commitment']
    result={'single_rcl_accuracy_gain':single,'multi_rcl_accuracy_gain':multi,'strict_four_cell_mean_ordering':ordering,'single_rcl_ci_excludes_zero':bool(single['ci95_low']>0),'multi_rcl_ci_excludes_zero':bool(multi['ci95_low']>0),'universal_ordering_claim':False}
    rows.extend(['',f'Strict proposed ordering of mean accuracies: **{ordering}**. This is descriptive; no universal ordering is assumed. Main contrasts are reported with unadjusted five-seed t intervals; secondary family/source comparisons are descriptive and no multiplicity correction is applied.','', '| Family | Single GRU | Single RCL | Multi GRU | Multi RCL |','|---|---:|---:|---:|---:|'])
    for family in FAMILIES:
        rows.append('| '+family+' | '+' | '.join(f"{100*lookup[cell,family]['correct_commitment']:.1f}%" for cell in ('single_plain','single_rcl','multi_plain','multi_rcl'))+' |')
    rows.extend(['','The single-source condition restricts visibility in the existing multi-source simulator; it does not change the red budget or report generator into a separate single-sensor adversary. Training trajectories use a shared scripted collection schedule; task evaluation is passive and cost-matched. Multi-source neural fitting uses eight episodes per hypothesis/family, likelihood fitting uses ten plus offline mechanism labels, matching the prior frozen training budgets. Temperature for each RCL and Gaussian view is fitted only on separate seen-family validation cases. Gaussian controls share exactly the emission parameters of the corresponding RCL model. These are new development draws of established family designs, not a new adversary benchmark. Original `single_report` used a last-timestep ablation and must not be equated with this single-source/full-history condition.'])
    (destination/'report.md').write_text('\n'.join(rows)+'\n')
    (destination/'claims.json').write_text(json.dumps(result,indent=2))
    return result


def run(output_dir=DEFAULT_OUTPUT,*,seeds=SEEDS,episodes=100,epochs=8,train_episodes=8,validation_episodes=3,hidden_size=32,max_cycles=10,smoke=False):
    destination=Path(output_dir)
    if destination.exists(): raise ValueError('Refusing to overwrite previous results')
    if not smoke and (tuple(seeds)!=SEEDS or episodes!=100 or epochs!=8 or train_episodes!=8 or validation_episodes!=3 or hidden_size!=32 or max_cycles!=10):
        raise ValueError('Frozen study settings; use smoke=True for tests')
    frozen={p.name:p for p in (ROOT/'outputs').iterdir() if p.is_dir() and (p.name.startswith('intelligence_') or p.name=='thesis_rcl_revision')}
    before={name:directory_manifest(path) for name,path in frozen.items()}
    source_before=source_manifest()
    destination.mkdir(parents=True)
    config={'seeds':list(seeds),'episodes_per_variant_family_seed':episodes,'max_cycles':max_cycles,'epochs':epochs,'hidden_size':hidden_size,'train_episodes_per_hypothesis':train_episodes,'validation_episodes_per_hypothesis':validation_episodes,'main_cells':list(CELLS[:4]),'diagnostic_cells':list(CELLS[4:]),'single_source_aggregation':'equal average across three source identities within each training seed','expected_episode_count':len(seeds)*len(FAMILIES)*episodes*12,'smoke_only':smoke,'training_mask_matches_runtime':True,'prior_results_replaced':False,'primary_contrasts':['rcl_gain_single','rcl_gain_multi'],'formal_benchmark_replaced':False}
    (destination/'config.json').write_text(json.dumps(config,indent=2))
    (destination/'source_manifest.json').write_text(json.dumps(source_before,indent=2))
    all_rows=[]
    for seed in seeds:
        raw_train=collect_prefix_records(TRAIN_FAMILIES,train_episodes,seed=1_300_000_000+seed*10003,split=f'factorial_train_{seed}',max_cycles=max_cycles)
        raw_validation=collect_prefix_records(TRAIN_FAMILIES,validation_episodes,seed=1_350_000_000+seed*10003,split=f'factorial_validation_{seed}',max_cycles=max_cycles)
        reasoner=HonestIntelligenceReasoner(raw_train)
        samples=collect_reliability_samples(RELIABILITY_TRAIN_FAMILIES,10 if not smoke else 1,seed=1_400_000_000+seed*10003,reasoner=reasoner,split=f'factorial_likelihood_{seed}',max_cycles=max_cycles)
        calibration_cases=collect_belief_cases(RELIABILITY_TRAIN_FAMILIES,3 if not smoke else 1,seed=1_450_000_000+seed*10003,max_cycles=max_cycles)
        controllers={}
        variant_meta={}
        for source in (None,*range(NUM_SOURCES)):
            prefix='multi' if source is None else f'source{source}'
            train=raw_train if source is None else [{**r,'sequence':restrict_sequence(r['sequence'],source)} for r in raw_train]
            validation=raw_validation if source is None else [{**r,'sequence':restrict_sequence(r['sequence'],source)} for r in raw_validation]
            model,history=train_model(train,validation,epochs=epochs,hidden_size=hidden_size,batch_size=64,seed=seed)
            plain=_inference(model,reasoner,None,prefix+'_plain',full_ride=False,max_cycles=max_cycles)
            visible_samples=samples if source is None else restrict_samples(samples,source)
            fixed=FixedCellReliability.fit(visible_samples)
            emissions=ReliabilityConditionedLikelihood.fit(visible_samples)
            gaussian=SingleReliableLikelihood(emissions.reliable_mean,emissions.reliable_scale,emissions.compromised_mean,emissions.compromised_scale)
            cases=calibration_cases if source is None else [{**r,'observation':restrict_observation(r['observation'],source)} for r in calibration_cases]
            mixture=fit_likelihood_temperature(emissions,cases,fixed,reasoner)
            gaussian=fit_likelihood_temperature(gaussian,cases,fixed,reasoner)
            candidates={'plain':plain,'rcl':ReliabilityLikelihoodController(mixture,fixed,reasoner,name=prefix+'_rcl'),'gaussian':ReliabilityLikelihoodController(gaussian,fixed,reasoner,name=prefix+'_gaussian')}
            for kind,controller in candidates.items():
                name=prefix+'_'+kind
                controller.name=name
                controllers[name]=controller if source is None else SourceScopedController(controller,source,name)
                variant_meta[name]={'source':-1 if source is None else source,'cell':('multi_' if source is None else 'single_')+kind}
            torch.save({'model_state':model.state_dict(),'source':source,'hidden_size':hidden_size,'seed':seed},destination/f'{prefix}_neural_seed{seed}.pt')
            parameters={field:getattr(emissions,field).tolist() for field in ('reliable_mean','reliable_scale','compromised_mean','compromised_scale')}
            parameters.update({'fixed_reliability':fixed.reliable.tolist(),'mixture_temperature':mixture.temperature,'gaussian_temperature':gaussian.temperature,'source':source})
            (destination/f'{prefix}_likelihood_seed{seed}.json').write_text(json.dumps(parameters,indent=2))
            write_csv(destination/f'{prefix}_training_seed{seed}.csv',history)
        seed_rows=[]
        for family_index,family in enumerate(FAMILIES):
            for episode in range(episodes):
                hypothesis=(episode+seed+family_index)%NUM_HYPOTHESES
                scenario_seed=1_500_000_000+seed*100003+family_index*1009+episode
                for name,controller in controllers.items():
                    row=evaluate_episode(controller,family,hypothesis,seed=scenario_seed,max_cycles=max_cycles)
                    if row['controller']!=name: raise AssertionError('Controller name drift')
                    row.update({'training_seed':seed,**variant_meta[name]})
                    seed_rows.append(row)
            print(f'seed={seed} family={family} complete',flush=True)
        write_csv(destination/f'episodes_seed{seed}.csv',seed_rows)
        all_rows.extend(seed_rows)
    if len(all_rows)!=config['expected_episode_count']: raise AssertionError('Incomplete factorial experiment')
    seed_rows,summary,contrasts,source_rows=summarize(all_rows)
    for filename,rows in [('episodes.csv',all_rows),('per_seed_summary.csv',seed_rows),('summary.csv',summary),('contrasts.csv',contrasts),('source_breakdown.csv',source_rows)]:write_csv(destination/filename,rows)
    audit={f'{name}_unchanged':before[name]==directory_manifest(path) for name,path in frozen.items()}
    audit.update({'source_files_unchanged':source_before==source_manifest(),'single_source_values_and_masks_zeroed':True,'single_source_public_reliability_masked':True,'single_source_temporal_history_preserved':True,'single_source_training_restricted':True,'source_variants_not_counted_as_independent_training_seeds':True,'runtime_truth_used':False,'linear_attenuation_used':False,'policy_inputs_audited':sum(r['boundary_checks'] for r in all_rows),'total_episodes':len(all_rows)})
    if not all(v for k,v in audit.items() if k.endswith('_unchanged')):raise AssertionError('Frozen output/source changed')
    (destination/'audit.json').write_text(json.dumps(audit,indent=2))
    result=report(destination,summary,contrasts)
    manifest={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.rglob('*')) if p.is_file()}
    (destination/'result_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    run(args.output_dir)
