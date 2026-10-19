"""Goal-only versus goal+deception supervision with matched shared GRUs."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from deception_rl.envs.deceptive_intelligence import NUM_HYPOTHESES
from deception_rl.intelligence.ride import IntelligenceRIDeModel,HonestIntelligenceReasoner,RIDeIntelligenceController,RIDeVariant,collate_intelligence_records
from experiments.run_intelligence_ride import TRAIN_FAMILIES,RED_FAMILIES,collect_prefix_records,_batches,evaluate_episode
from experiments.run_intelligence_formal_evaluation import summarize_group
from experiments.run_intelligence_rcl_formal import source_manifest,METRICS
from experiments.run_intelligence_ride_v2_headroom import directory_manifest,write_csv

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=ROOT/'outputs/intelligence_clean_dual_head_ablation'
SEEDS=(4201,4202,4203,4204,4205)


def train(train_records,validation,seed,dual,epochs=8):
    torch.manual_seed(seed)
    model=IntelligenceRIDeModel(32)
    optimizer=torch.optim.Adam(model.parameters(),lr=.004)
    labels=np.asarray([r['deception_label'] for r in train_records])
    counts=np.bincount(labels,minlength=2)
    weights=torch.tensor(len(labels)/(2*np.maximum(counts,1)),dtype=torch.float32)
    best=float('inf');state=deepcopy(model.state_dict());history=[]
    rng=random.Random(seed)
    for epoch in range(epochs):
        model.train()
        for batch in _batches(train_records,64,rng):
            x,lengths,h,d=collate_intelligence_records(batch)
            optimizer.zero_grad()
            hl,dl=model(x,lengths)
            loss=nn.functional.cross_entropy(hl,h)
            if dual:loss+=nn.functional.cross_entropy(dl,d,weight=weights)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model.eval();total=0.
        with torch.inference_mode():
            for start in range(0,len(validation),64):
                batch=validation[start:start+64]
                x,lengths,h,d=collate_intelligence_records(batch)
                hl,_=model(x,lengths)
                total+=float(nn.functional.cross_entropy(hl,h))*len(batch)
        goal_loss=total/len(validation)
        history.append({'epoch':epoch+1,'validation_goal_cross_entropy':goal_loss})
        if goal_loss<best:best=goal_loss;state=deepcopy(model.state_dict())
    model.load_state_dict(state);model.eval()
    return model,history


def run(output_dir=DEFAULT_OUTPUT,*,seeds=SEEDS,episodes=100,epochs=8,smoke=False):
    destination=Path(output_dir)
    if destination.exists():raise ValueError('Refusing to overwrite existing outputs')
    if not smoke and (tuple(seeds)!=SEEDS or episodes!=100 or epochs!=8):raise ValueError('Frozen protocol')
    frozen={p.name:p for p in (ROOT/'outputs').iterdir() if p.is_dir() and (p.name.startswith('intelligence_') or p.name=='thesis_rcl_revision')}
    before={n:directory_manifest(p) for n,p in frozen.items()};sources=source_manifest()
    destination.mkdir(parents=True)
    config={'seeds':list(seeds),'methods':['goal_only','dual_head'],'episodes_per_family_seed':episodes,'epochs':epochs,'hidden_size':32,'training_episodes_per_hypothesis_family':8,'validation_episodes_per_hypothesis_family':3,'identical_initialization_minibatches_and_optimizer':True,'model_selection':'minimum validation goal cross entropy for both methods','dual_head_loss_weight':1.,'temperature_calibration':False,'counterfactual':False,'adaptive_commitment':False,'fixed_commit_step':4,'primary_metrics':['correct_commitment','hypothesis_brier'],'expected_episodes':len(seeds)*7*episodes*2,'smoke_only':smoke}
    (destination/'protocol.json').write_text(json.dumps(config,indent=2))
    (destination/'source_manifest.json').write_text(json.dumps(sources,indent=2))
    rows=[]
    for seed in seeds:
        train_records=collect_prefix_records(TRAIN_FAMILIES,8 if not smoke else 1,seed=2_500_000_000+seed*1009,split=f'clean_dual_train_{seed}',max_cycles=10)
        validation=collect_prefix_records(TRAIN_FAMILIES,3 if not smoke else 1,seed=2_600_000_000+seed*1009,split=f'clean_dual_validation_{seed}',max_cycles=10)
        reasoner=HonestIntelligenceReasoner(train_records)
        controllers={}
        for name,dual in [('goal_only',False),('dual_head',True)]:
            model,history=train(train_records,validation,seed,dual,epochs)
            variant=RIDeVariant(name,use_deception_head=dual,use_counterfactual=False,use_calibration=False,adaptive_commitment=False)
            controllers[name]=RIDeIntelligenceController(model,reasoner,None,variant,max_cycles=10,fixed_commit_step=4)
            torch.save({'model_state':model.state_dict(),'seed':seed,'hidden_size':32,'dual_loss':dual},destination/f'{name}_seed{seed}.pt')
            write_csv(destination/f'{name}_history_seed{seed}.csv',history)
        for fi,family in enumerate(RED_FAMILIES):
            for episode in range(episodes):
                hypothesis=(episode+seed+fi)%NUM_HYPOTHESES
                scenario_seed=2_700_000_000+seed*1009+fi*100003+episode
                for name,controller in controllers.items():
                    row=evaluate_episode(controller,family,hypothesis,seed=scenario_seed,max_cycles=10)
                    row.update({'training_seed':seed,'deception_head_present':name=='dual_head'})
                    rows.append(row)
        print(f'Dual-head test complete seed={seed}',flush=True)
    seed_rows=[]
    for seed in seeds:
        for method in ['goal_only','dual_head']:
            for scope in ('overall','seen','unseen',*RED_FAMILIES):
                group=[r for r in rows if r['training_seed']==seed and r['controller']==method and (scope=='overall' or (r['family_scope']==scope if scope in {'seen','unseen'} else r['deception_family']==scope))]
                metrics=summarize_group(group)
                if method=='goal_only':
                    metrics={k:('' if k.startswith('deception_') else v) for k,v in metrics.items()}
                seed_rows.append({'seed':seed,'method':method,'scope':scope,**metrics})
    summary=[]
    for method in ['goal_only','dual_head']:
        for scope in ('overall','seen','unseen',*RED_FAMILIES):
            group=[r for r in seed_rows if r['method']==method and r['scope']==scope]
            item={'method':method,'scope':scope}
            for metric in summarize_group(rows):
                if method=='goal_only' and metric.startswith('deception_'):item[metric]='';continue
                item[metric]=float(np.mean([r[metric] for r in group]))
            summary.append(item)
    lookup={(r['seed'],r['method'],r['scope']):r for r in seed_rows}
    contrasts=[]
    for metric in METRICS:
        values=np.asarray([lookup[s,'dual_head','overall'][metric]-lookup[s,'goal_only','overall'][metric] for s in seeds])
        half=2.776*values.std(ddof=1)/math.sqrt(5) if len(seeds)==5 else float('nan')
        contrasts.append({'metric':metric,'difference':float(values.mean()),'ci95_low':float(values.mean()-half),'ci95_high':float(values.mean()+half)})
    for filename,values in [('episodes.csv',rows),('per_seed_summary.csv',seed_rows),('summary.csv',summary),('paired_differences.csv',contrasts)]:write_csv(destination/filename,values)
    audit={n+'_unchanged':before[n]==directory_manifest(p) for n,p in frozen.items()}
    audit.update({'source_unchanged':sources==source_manifest(),'runtime_truth_used':False,'test_episodes':len(rows)})
    assert all(v for k,v in audit.items() if k.endswith('_unchanged'))
    (destination/'audit.json').write_text(json.dumps(audit,indent=2))
    assert len(rows)==config['expected_episodes']
    lines=['# Clean single-head versus dual-head ablation','', 'The goal branch, GRU size, initial parameters, minibatch ordering, optimizer, training set, validation set, and commitment time are matched. The only training intervention is addition of weighted deception cross entropy. Both checkpoints are selected using validation goal cross entropy. No temperature, counterfactual module, or active sensing is used at evaluation. Goal-only has no trained deception head; its deception metrics are not meaningful and are left blank in summary tables. This study tests encoder multitask supervision, not a complete joint posterior.','', '| Method | Correct commitment | Brier | ECE | NLL | Deception Brier | Deception AUROC | Regret |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['scope']=='overall':lines.append(f"| {r['method']} | {100*r['correct_commitment']:.1f}% | {r['hypothesis_brier']:.3f} | {r['hypothesis_ece']:.3f} | {r['hypothesis_nll']:.3f} | {r['deception_brier']} | {r['deception_auroc']} | {r['regret']:.3f} |")
    lines.extend(['','Paired differences (dual minus goal-only):','', '| Metric | Difference | Five-seed 95% t CI |','|---|---:|---:|'])
    for r in contrasts:lines.append(f"| {r['metric']} | {r['difference']:+.4f} | [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] |")
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
    return {'episodes':len(rows),'paired':contrasts}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    print(json.dumps(run(parser.parse_args().output_dir),indent=2))
