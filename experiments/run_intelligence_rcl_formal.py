"""Frozen fresh-seed formal evaluation of fixed-prior RCL inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deception_rl.agents.intelligence_policies import AdaptiveVerificationBluePolicy
from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES
from deception_rl.intelligence.baselines import BayesianFusionController, POMCPLikeController, ActiveInferenceController
from deception_rl.intelligence.ride import HonestIntelligenceReasoner
from deception_rl.intelligence.ride_v2.fixed_reliability import FixedCellReliability
from deception_rl.intelligence.ride_v2.reliability_conditioned import ReliabilityConditionedLikelihood, ReliabilityLikelihoodController
from experiments.run_intelligence_ride import RED_FAMILIES, TRAIN_FAMILIES, collect_prefix_records, train_model, fit_calibrator, evaluate_episode
from experiments.run_intelligence_ride_v2_reliability import RELIABILITY_TRAIN_FAMILIES, collect_reliability_samples, _inference
from experiments.run_intelligence_reliability_conditioned_inference import collect_belief_cases, fit_likelihood_temperature
from experiments.run_intelligence_ride_v2_headroom import directory_manifest, write_csv
from experiments.run_intelligence_formal_evaluation import summarize_group

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT/'outputs/intelligence_rcl_formal_evaluation'
SEEDS = (3001, 3002, 3003, 3004, 3005)
METHODS = ('multi_source', 'ride_v1', 'rcl_fixed', 'bayesian_fusion', 'pomcp_like', 'active_inference', 'rule_active', 'single_mechanism')
METRICS = ('correct_commitment','false_commitment','no_commit_rate','decision_latency','information_cost','regret','hypothesis_brier','hypothesis_ece','hypothesis_nll')


class SingleReliableLikelihood(ReliabilityConditionedLikelihood):
    """Matched emissions with the compromised mechanism removed exactly."""
    def logits(self, observation, predictor, reasoner):
        from deception_rl.intelligence.adapter import audit_blue_observation
        audit_blue_observation(observation)
        reports=np.asarray(observation['reports'],dtype=float)
        mask=np.asarray(observation['report_mask'],dtype=bool)
        times=np.flatnonzero(np.asarray(observation['time_mask'],dtype=bool))
        result=np.full(NUM_HYPOTHESES,-np.log(NUM_HYPOTHESES))
        for timestep in times:
            for source,region in np.argwhere(mask[timestep]):
                result += self._log_normal(reports[timestep,source,region,:2],self.reliable_mean[:,source,region],self.reliable_scale[:,source,region])
        return result/self.temperature

    def with_temperature(self, temperature):
        return SingleReliableLikelihood(self.reliable_mean,self.reliable_scale,self.compromised_mean,self.compromised_scale,float(temperature))


def source_manifest():
    paths = list((ROOT/'src/deception_rl').rglob('*.py')) + list((ROOT/'experiments').glob('*intelligence*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def protocol(seeds, episodes, epochs, hidden_size, train_episodes, validation_episodes, max_cycles):
    return {
        'protocol':'Frozen Fixed-Prior RCL Formal v1', 'training_seeds':list(seeds),
        'methods':list(METHODS), 'families':list(RED_FAMILIES),
        'episodes_per_method_family_seed':episodes,
        'expected_evaluation_episodes':len(seeds)*len(METHODS)*len(RED_FAMILIES)*episodes,
        'epochs':epochs,'hidden_size':hidden_size,'batch_size':64,
        'neural_train_episodes_per_hypothesis_family':train_episodes,
        'neural_validation_episodes_per_hypothesis_family':validation_episodes,
        'likelihood_train_episodes_per_hypothesis_family':10,
        'likelihood_fit_families':list(RELIABILITY_TRAIN_FAMILIES),
        'temperature_validation_cases_per_seen_family':3,
        'temperature_grid':{'min':0.1,'max':10.0,'count':101},
        'scale_floor':0.08,'fixed_prior_smoothing':'Beta(1,1)',
        'fixed_commit_step':4,'max_cycles':max_cycles,
        'primary_comparison':'rcl_fixed minus multi_source',
        'primary_support_rule':'paired seed 95% CI accuracy lower > 0; Brier lower; regret lower; cost and latency equal',
        'mechanism_comparison':'rcl_fixed minus single_mechanism (diagnostic, no multiplicity correction)',
        'interval_unit':'training seed; 95% t interval; df=4 for five seeds',
        'held_out_meaning':'fresh train/calibration/scenario seeds; multi_source, switch, adaptive excluded from fit; family designs have been explored in development',
        'formal_results_replaced':False,'learned_reliability_used':False,
        'optional_end_to_end_policy':'not included; no frozen trained policy available',
    }


def per_seed(rows):
    output=[]
    for seed in sorted({r['training_seed'] for r in rows}):
        for method in METHODS:
            method_rows=[r for r in rows if r['training_seed']==seed and r['controller']==method]
            for scope in ('overall','seen','unseen',*RED_FAMILIES):
                selected=method_rows if scope=='overall' else [r for r in method_rows if (r['family_scope']==scope if scope in ('seen','unseen') else r['deception_family']==scope)]
                output.append({'seed':seed,'method':method,'scope':scope,'episodes':len(selected),**{k:v for k,v in summarize_group(selected).items() if k in METRICS}})
    return output


def analyze(rows):
    seed_rows=per_seed(rows)
    grouped=defaultdict(list)
    for row in seed_rows: grouped[(row['method'],row['scope'])].append(row)
    summary=[]
    for (method,scope),selected in sorted(grouped.items()):
        item={'method':method,'scope':scope,'seeds':len(selected),'episodes':sum(r['episodes'] for r in selected)}
        for metric in METRICS:
            values=np.asarray([r[metric] for r in selected])
            half=2.776*values.std(ddof=1)/math.sqrt(5) if len(values)==5 else float('nan')
            item[metric]=float(values.mean())
            item[metric+'_ci95_low']=float(values.mean()-half)
            item[metric+'_ci95_high']=float(values.mean()+half)
        summary.append(item)
    paired=[]
    lookup={(r['seed'],r['method'],r['scope']):r for r in seed_rows}
    seeds=sorted({r['training_seed'] for r in rows})
    for comparator in METHODS:
        if comparator=='rcl_fixed': continue
        for scope in ('overall','seen','unseen',*RED_FAMILIES):
            for metric in METRICS:
                values=np.asarray([lookup[(seed,'rcl_fixed',scope)][metric]-lookup[(seed,comparator,scope)][metric] for seed in seeds])
                half=2.776*values.std(ddof=1)/math.sqrt(5) if len(values)==5 else float('nan')
                paired.append({'comparator':comparator,'scope':scope,'metric':metric,'difference':float(values.mean()),'ci95_low':float(values.mean()-half),'ci95_high':float(values.mean()+half)})
    return seed_rows,summary,paired


def write_report(destination, summary, paired):
    lookup={(r['method'],r['scope']):r for r in summary}
    differences={(r['comparator'],r['scope'],r['metric']):r for r in paired}
    main=differences[('multi_source','overall','correct_commitment')]
    rcl=lookup[('rcl_fixed','overall')]
    base=lookup[('multi_source','overall')]
    supported=(main['ci95_low']>0 and rcl['hypothesis_brier']<base['hypothesis_brier'] and rcl['regret']<base['regret'] and abs(rcl['decision_latency']-base['decision_latency'])<1e-9 and abs(rcl['information_cost']-base['information_cost'])<1e-9)
    config=json.loads((destination/'protocol.json').read_text())
    lines=['# Fixed-Prior RCL Formal Evaluation','', f"Training seeds {config['training_seeds']}; {config['episodes_per_method_family_seed']} paired episodes per method × family × seed; {config['expected_evaluation_episodes']:,} total episodes. Intervals use training-seed means, not individual episodes. All fitting and calibration precede evaluation for each seed. Smoke only: {config['smoke_only']}.",'', '| Method | Correct commitment | Brier | ECE | NLL | Regret | Latency | Cost |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        r=lookup[(method,'overall')]
        lines.append(f"| {method} | {100*r['correct_commitment']:.1f}% | {r['hypothesis_brier']:.3f} | {r['hypothesis_ece']:.3f} | {r['hypothesis_nll']:.3f} | {r['regret']:.2f} | {r['decision_latency']:.2f} | {r['information_cost']:.2f} |")
    lines.extend(['', f"Primary paired accuracy difference: **{100*main['difference']:+.2f} pp** (95% CI {100*main['ci95_low']:+.2f} to {100*main['ci95_high']:+.2f} pp).",f"Primary frozen support rule: **{'SUPPORTED' if supported else 'NOT SUPPORTED'}**.",'', '| Family | Multi-source | RCL-fixed | Difference | Single mechanism |','|---|---:|---:|---:|---:|'])
    for family in RED_FAMILIES:
        a=lookup[('multi_source',family)]['correct_commitment'];b=lookup[('rcl_fixed',family)]['correct_commitment'];c=lookup[('single_mechanism',family)]['correct_commitment']
        lines.append(f'| {family} | {100*a:.1f}% | {100*b:.1f}% | {100*(b-a):+.1f} pp | {100*c:.1f}% |')
    lines.extend(['','## Seen and unseen families','', '| Method | Seen | Unseen | Seen minus unseen |','|---|---:|---:|---:|'])
    for method in METHODS:
        seen=lookup[(method,'seen')]['correct_commitment'];unseen=lookup[(method,'unseen')]['correct_commitment']
        lines.append(f'| {method} | {100*seen:.1f}% | {100*unseen:.1f}% | {100*(seen-unseen):+.1f} pp |')
    mechanism=differences[('single_mechanism','overall','correct_commitment')]
    lines.extend(['','## Interpretation and limits','',f"Mixture versus matched single mechanism accuracy: {100*mechanism['difference']:+.2f} pp (95% CI {100*mechanism['ci95_low']:+.2f} to {100*mechanism['ci95_high']:+.2f} pp). This diagnostic separates mixture modeling from Gaussian emission modeling; it is secondary and has no multiplicity correction.", 'RCL-fixed uses training-only simulator mechanism labels to fit reliable/compromised emissions. It has additional mechanism supervision compared with the neural Multi-source baseline, so the experiment does not isolate architecture under equal supervision. Its compromised mechanism is hypothesis independent and report features are modeled as independent Gaussians with a validation-fitted temperature.', 'Only RCL-fixed and Multi-source share the fixed passive decision schedule. Other baselines retain their original active sensing and stopping rules; compare their cost and latency alongside accuracy. POMCP-like is the repository\'s one-step categorical rollout baseline, not a full POMCP solver.', 'Family designs were used during development; only the training, calibration, and episode draws are fresh. This is generalization to held-out scenario draws and families excluded from fitting, not proof of robustness to newly invented deception families.', 'RCL has no deception classification head; no deception-classification superiority is claimed. This result also does not support gains from learned reliability, which is absent.', 'All older formal and development outputs remain frozen. This study concerns an abstract adversarial intelligence simulator.'])
    (destination/'formal_report.md').write_text('\n'.join(lines)+'\n')
    result={'primary_claim_supported':bool(supported),'rcl_vs_multisource_accuracy':main,'rcl_vs_single_mechanism_accuracy':mechanism}
    (destination/'claim_audit.json').write_text(json.dumps(result,indent=2))
    return result


def run(output_dir=DEFAULT_OUTPUT, *, seeds=SEEDS, episodes=100, epochs=8, hidden_size=32, train_episodes=8, validation_episodes=3, max_cycles=10, smoke=False):
    destination=Path(output_dir)
    if destination.exists(): raise ValueError('Refusing to overwrite any existing output directory')
    if not smoke and (tuple(seeds)!=SEEDS or episodes!=100 or epochs!=8 or hidden_size!=32 or train_episodes!=8 or validation_episodes!=3 or max_cycles!=10):
        raise ValueError('Formal protocol is frozen; custom configurations require smoke=True')
    frozen={p.name:p for p in (ROOT/'outputs').iterdir() if p.is_dir() and p.name.startswith('intelligence_')}
    before={name:directory_manifest(path) for name,path in frozen.items()}
    sources=source_manifest()
    destination.mkdir(parents=True)
    config=protocol(seeds,episodes,epochs,hidden_size,train_episodes,validation_episodes,max_cycles)
    config['smoke_only']=smoke
    (destination/'protocol.json').write_text(json.dumps(config,indent=2))
    (destination/'source_manifest.json').write_text(json.dumps(sources,indent=2))
    all_rows=[]
    for seed in seeds:
        train=collect_prefix_records(TRAIN_FAMILIES,train_episodes,seed=600_000_000+seed*10003,split=f'rcl_formal_train_{seed}',max_cycles=max_cycles)
        validation=collect_prefix_records(TRAIN_FAMILIES,validation_episodes,seed=650_000_000+seed*10003,split=f'rcl_formal_validation_{seed}',max_cycles=max_cycles)
        model,history=train_model(train,validation,epochs=epochs,hidden_size=hidden_size,batch_size=64,seed=seed)
        calibrator,_=fit_calibrator(model,validation)
        reasoner=HonestIntelligenceReasoner(train)
        likelihood_samples=collect_reliability_samples(RELIABILITY_TRAIN_FAMILIES,10 if not smoke else 1,seed=700_000_000+seed*10003,reasoner=reasoner,split=f'rcl_formal_likelihood_{seed}',max_cycles=max_cycles)
        fixed=FixedCellReliability.fit(likelihood_samples)
        raw_likelihood=ReliabilityConditionedLikelihood.fit(likelihood_samples)
        calibration_cases=collect_belief_cases(RELIABILITY_TRAIN_FAMILIES,3 if not smoke else 1,seed=750_000_000+seed*10003,max_cycles=max_cycles)
        likelihood=fit_likelihood_temperature(raw_likelihood,calibration_cases,fixed,reasoner)
        single_fixed=FixedCellReliability(np.ones_like(fixed.reliable))
        single_model=SingleReliableLikelihood(raw_likelihood.reliable_mean,raw_likelihood.reliable_scale,raw_likelihood.compromised_mean,raw_likelihood.compromised_scale)
        single_likelihood=fit_likelihood_temperature(single_model,calibration_cases,single_fixed,reasoner)
        controllers={
            'multi_source':_inference(model,reasoner,None,'multi_source',full_ride=False,max_cycles=max_cycles),
            'ride_v1':_inference(model,reasoner,calibrator,'ride_v1',full_ride=True,max_cycles=max_cycles),
            'rcl_fixed':ReliabilityLikelihoodController(likelihood,fixed,reasoner,name='rcl_fixed'),
            'bayesian_fusion':BayesianFusionController(max_cycles=max_cycles),
            'pomcp_like':POMCPLikeController(max_cycles=max_cycles,seed=seed),
            'active_inference':ActiveInferenceController(max_cycles=max_cycles),
            'rule_active':AdaptiveVerificationBluePolicy(max_cycles=max_cycles),
            'single_mechanism':ReliabilityLikelihoodController(single_likelihood,single_fixed,reasoner,name='single_mechanism'),
        }
        controllers['rule_active'].name='rule_active'
        if tuple(controllers)!=METHODS: raise AssertionError('Method order drift')
        torch.save({'model_state':model.state_dict(),'calibrator_state':calibrator.state_dict(),'hidden_size':hidden_size,'seed':seed},destination/f'neural_seed{seed}.pt')
        parameters={field:getattr(likelihood,field).tolist() for field in ('reliable_mean','reliable_scale','compromised_mean','compromised_scale')}
        parameters.update({'fixed_reliability':fixed.reliable.tolist(),'mixture_temperature':likelihood.temperature,'single_temperature':single_likelihood.temperature})
        (destination/f'rcl_seed{seed}.json').write_text(json.dumps(parameters,indent=2))
        write_csv(destination/f'training_history_seed{seed}.csv',history)
        seed_rows=[]
        for family_index,family in enumerate(RED_FAMILIES):
            for episode in range(episodes):
                hypothesis=(episode+seed+family_index)%NUM_HYPOTHESES
                scenario_seed=800_000_000+seed*100003+family_index*1009+episode
                for method,controller in controllers.items():
                    if hasattr(controller,'seed'): controller.seed=scenario_seed
                    row=evaluate_episode(controller,family,hypothesis,seed=scenario_seed,max_cycles=max_cycles)
                    if row['controller']!=method: raise AssertionError('Method name drift')
                    row['training_seed']=seed
                    seed_rows.append(row)
            print(f'seed={seed} family={family} completed',flush=True)
        write_csv(destination/f'episodes_seed{seed}.csv',seed_rows)
        all_rows.extend(seed_rows)
    if len(all_rows)!=config['expected_evaluation_episodes']: raise AssertionError('Episode count drift')
    seed_rows,summary,paired=analyze(all_rows)
    write_csv(destination/'episodes.csv',all_rows)
    write_csv(destination/'per_seed_summary.csv',seed_rows)
    write_csv(destination/'summary.csv',summary)
    write_csv(destination/'paired_differences.csv',paired)
    audit={f'{name}_unchanged':before[name]==directory_manifest(path) for name,path in frozen.items()}
    audit.update({'source_files_unchanged':sources==source_manifest(),'terminal_truth_used_by_policy':False,'runtime_oracle_used':False,'fixed_prior_training_only':True,'blue_inputs_audited':sum(r['boundary_checks'] for r in all_rows),'total_episodes':len(all_rows)})
    if not all(v for k,v in audit.items() if k.endswith('_unchanged')): raise AssertionError('Frozen files changed during evaluation')
    (destination/'information_boundary_audit.json').write_text(json.dumps(audit,indent=2))
    result=write_report(destination,summary,paired)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    run(args.output_dir)
