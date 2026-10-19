"""Fresh-seed fixed versus learned reliability ablation with shared RCL emissions."""
import json
from pathlib import Path
import numpy as np
import torch
from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES
from deception_rl.intelligence.ride import HonestIntelligenceReasoner
from deception_rl.intelligence.ride_v2.calibrated_reliability import CalibratedCellReliabilityFilter, LogisticCellModel, PlattCalibrator
from deception_rl.intelligence.ride_v2.fixed_reliability import FixedCellReliability
from deception_rl.intelligence.ride_v2.reliability_conditioned import ReliabilityConditionedLikelihood, ReliabilityLikelihoodController
from experiments.run_intelligence_calibrated_cell_reliability import DEFAULT_OUTPUT as PREDICTOR_OUTPUT, CALIBRATED_SEEDS
from experiments.run_intelligence_reliability_conditioned_inference import DEFAULT_OUTPUT as PREVIOUS_OUTPUT, collect_belief_cases, fit_likelihood_temperature, evaluate_beliefs
from experiments.run_intelligence_ride import TRAIN_FAMILIES, collect_prefix_records, train_model
from experiments.run_intelligence_ride_v2_reliability import RELIABILITY_TRAIN_FAMILIES, collect_reliability_samples, _inference
from experiments.run_intelligence_ride_v2_headroom import DEVELOPMENT_RED_FAMILIES, directory_manifest, evaluate_development_episode, summarize, write_csv

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'outputs/intelligence_rcl_fixed_ablation_development'
SEEDS = (2525, 2626, 2727)
METHODS = ('multi_source','rcl_fixed','rcl_learned')

def run():
    if OUTPUT.exists():
        raise ValueError('Use a new output directory; existing results must remain frozen')
    frozen = {p.name:p for p in (ROOT/'outputs').iterdir() if p.is_dir() and p.name.startswith('intelligence_')}
    before = {name:directory_manifest(path) for name,path in frozen.items()}
    OUTPUT.mkdir()
    beliefs=[]
    episodes=[]
    for seed,predictor_seed in zip(SEEDS,CALIBRATED_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        train = collect_prefix_records(TRAIN_FAMILIES,8,seed=400_000_000+seed*10003,split=f'ablation_train_{seed}',max_cycles=10)
        validation = collect_prefix_records(TRAIN_FAMILIES,3,seed=410_000_000+seed*10003,split=f'ablation_validation_{seed}',max_cycles=10)
        model,_ = train_model(train,validation,epochs=8,hidden_size=32,batch_size=64,seed=seed)
        payload=json.loads((PREDICTOR_OUTPUT/f'calibrated_model_seed{predictor_seed}.json').read_text())
        params=payload['balanced_logistic']
        learned=CalibratedCellReliabilityFilter(LogisticCellModel(np.asarray(params['mean']),np.asarray(params['scale']),np.asarray(params['coefficient']),params['balanced']),PlattCalibrator(**payload['platt']))
        prototype_records=collect_prefix_records(TRAIN_FAMILIES,8,seed=210_000_000+predictor_seed*10003,split=f'calibrated_reasoner_{predictor_seed}',max_cycles=10)
        reasoner=HonestIntelligenceReasoner(prototype_records)
        samples=collect_reliability_samples(RELIABILITY_TRAIN_FAMILIES,10,seed=420_000_000+seed*10003,reasoner=reasoner,split=f'ablation_likelihood_train_{seed}',max_cycles=10)
        fixed=FixedCellReliability.fit(samples)
        emissions=ReliabilityConditionedLikelihood.fit(samples)
        calibration_cases=collect_belief_cases(RELIABILITY_TRAIN_FAMILIES,3,seed=430_000_000+seed*10003,max_cycles=10)
        controllers={'multi_source':_inference(model,reasoner,None,'multi_source',full_ride=False,max_cycles=10)}
        temperatures={}
        for name,predictor in [('rcl_fixed',fixed),('rcl_learned',learned)]:
            likelihood=fit_likelihood_temperature(emissions,calibration_cases,predictor,reasoner)
            temperatures[name]=likelihood.temperature
            controllers[name]=ReliabilityLikelihoodController(likelihood,predictor,reasoner,name=name)
        (OUTPUT/f'parameters_seed{seed}.json').write_text(json.dumps({'predictor_seed':predictor_seed,'fixed_reliability':fixed.reliable.tolist(),'temperatures':temperatures,'emissions_shared':True},indent=2))
        cases=collect_belief_cases(tuple(DEVELOPMENT_RED_FAMILIES),30,seed=440_000_000+seed*10003,max_cycles=10)
        beliefs.extend(evaluate_beliefs(controllers,cases,seed))
        for family_index,family in enumerate(DEVELOPMENT_RED_FAMILIES):
            for episode in range(30):
                hypothesis=(episode+seed+family_index)%NUM_HYPOTHESES
                for name,controller in controllers.items():
                    row,_=evaluate_development_episode(controller,family,hypothesis,seed=450_000_000+seed*100003+family_index*1009+episode,max_cycles=10)
                    row['development_seed']=seed
                    episodes.append(row)
        print(f'Completed seed {seed}',flush=True)
    summary=summarize(episodes)
    for method in METHODS:
        summary.extend(summarize([{**r,'deception_family':'overall'} for r in episodes if r['controller']==method]))
    write_csv(OUTPUT/'commitment_episodes.csv',episodes)
    write_csv(OUTPUT/'commitment_summary.csv',summary)
    write_csv(OUTPUT/'belief_metrics_by_seed.csv',beliefs)
    lookup={(r['method'],r['deception_family']):r for r in summary}
    lines=['# RCL Fixed Reliability Ablation','', 'Fresh development seeds: 2525 / 2626 / 2727. Fixed cell priors use only reported training cells with Beta(1,1) smoothing. Both RCL methods share identical fitted emission parameters; each temperature uses the same separate seen-family validation cases. Predictor coefficients are loaded unchanged. Commitment is fixed at step 4; no Verify.','', '| Method | Correct commitment | Regret | Latency | Cost | Belief Brier | Belief ECE |','|---|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        r=lookup[(method,'overall')]
        bs=[b for b in beliefs if b['method']==method and b['deception_family']=='overall']
        lines.append(f"| {method} | {100*r['correct_commitment']:.1f}% | {r['regret']:.2f} | {r['decision_latency']:.2f} | {r['information_cost']:.2f} | {np.mean([b['hypothesis_brier'] for b in bs]):.3f} | {np.mean([b['hypothesis_ece'] for b in bs]):.3f} |")
    baseline=lookup[('multi_source','overall')]['correct_commitment']
    fixed_accuracy=lookup[('rcl_fixed','overall')]['correct_commitment']
    learned_accuracy=lookup[('rcl_learned','overall')]['correct_commitment']
    changes=[]
    for seed in SEEDS:
        acc={m:np.mean([r['correct_commitment'] for r in episodes if r['development_seed']==seed and r['controller']==m]) for m in METHODS}
        changes.append(acc['rcl_learned']-acc['rcl_fixed'])
    lines.extend(['',f'Fixed vs Multi-source: {100*(fixed_accuracy-baseline):+.2f} pp.',f'Learned vs fixed: {100*(learned_accuracy-fixed_accuracy):+.2f} pp.', 'Per-seed learned vs fixed: '+', '.join(f'{100*x:+.2f} pp' for x in changes)+'.','', '| Family | Multi-source | RCL fixed | RCL learned |','|---|---:|---:|---:|'])
    for family in DEVELOPMENT_RED_FAMILIES:
        lines.append('| '+family+' | '+' | '.join(f"{100*lookup[(m,family)]['correct_commitment']:.1f}%" for m in METHODS)+' |')
    lines.extend(['','This is development evidence only. Dynamic reliability attribution requires learned RCL to improve over fixed RCL; improvement over Multi-source alone cannot establish that attribution. No formal benchmark was run.'])
    interval_half = 4.303 * np.std(changes, ddof=1) / np.sqrt(len(changes))
    attribution = {
        'decision': 'LEARNED_RELIABILITY_NOT_SUPPORTED' if learned_accuracy <= fixed_accuracy else 'LEARNED_RELIABILITY_PROMISING',
        'learned_vs_fixed': learned_accuracy-fixed_accuracy,
        'seed_differences': [float(x) for x in changes],
        'seed_t_ci95': [float(np.mean(changes)-interval_half),float(np.mean(changes)+interval_half)],
        'formal_benchmark_started': False,
    }
    (OUTPUT/'attribution.json').write_text(json.dumps(attribution,indent=2))
    (OUTPUT/'config.json').write_text(json.dumps({
        'seeds':list(SEEDS), 'predictor_seed_mapping':dict(zip(map(str,SEEDS),CALIBRATED_SEEDS)),
        'train_episodes_per_hypothesis':8, 'likelihood_train_episodes_per_hypothesis':10,
        'calibration_cases_per_family':3, 'evaluation_episodes_per_family_seed':30,
        'methods':list(METHODS), 'epochs':8, 'hidden_size':32,
        'commitment_episode_count':len(episodes), 'belief_cases_per_method':len(SEEDS)*7*30,
        'fixed_prior_smoothing':'Beta(1,1)', 'formal_results_replaced':False,
    },indent=2))
    lines.extend(['', f"Attribution decision: **{attribution['decision']}**.",
                  f"Seed-level 95% t interval for learned minus fixed: {100*attribution['seed_t_ci95'][0]:+.2f} to {100*attribution['seed_t_ci95'][1]:+.2f} pp (3 seeds)."])
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')
    audit={f'{name}_unchanged':before[name]==directory_manifest(path) for name,path in frozen.items()}
    if not all(audit.values()): raise AssertionError('Frozen output changed')
    audit.update({'fixed_priors_training_only':True,'predictor_refitted':False,'shared_emission_parameters':True,'runtime_hidden_labels':False})
    (OUTPUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print('\n'.join(lines))

if __name__=='__main__': run()
