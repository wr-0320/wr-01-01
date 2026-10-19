"""Build Stage 9 data and train a small supervised GRU baseline, no PPO."""
from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from deception_rl.data.anti_deception import (load_stage6,family_split,balanced_limit,
                                              export_dataset,DeceptionDataset,collate,audit)
from models.anti_deception_gru import AntiDeceptionGRU

def binary_metrics(labels,probabilities):
    labels=np.asarray(labels,dtype=int);probabilities=np.asarray(probabilities,dtype=float)
    if len(labels)==0 or labels.shape!=probabilities.shape:raise ValueError('Nonempty matching metric inputs required')
    if not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any():raise ValueError('Invalid probabilities')
    if not np.isin(labels,[0,1]).all():raise ValueError('Labels must be binary')
    pred=probabilities>=.5;tp=int(((labels==1)&pred).sum());fp=int(((labels==0)&pred).sum());fn=int(((labels==1)&~pred).sum())
    positives=probabilities[labels==1];negatives=np.sort(probabilities[labels==0])
    auc=None
    if len(positives) and len(negatives):
        lower=np.searchsorted(negatives,positives,side='left');upper=np.searchsorted(negatives,positives,side='right')
        auc=float(np.mean((lower+upper)/2)/len(negatives))
    return dict(accuracy=float(np.mean(pred==labels)),f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
                auroc=auc,positive_support=len(positives),negative_support=len(negatives),tp=tp,fp=fp,fn=fn)

def evaluate(model,records,batch_size=64):
    loader=DataLoader(DeceptionDataset(records),batch_size=batch_size,collate_fn=collate)
    model.eval();gp=[];dp=[];gl=[];dl=[];losses=[]
    with torch.inference_mode():
        for x,lengths,g,d in loader:
            goal,deception=model(x,lengths)
            loss=nn.functional.cross_entropy(goal,g)+nn.functional.cross_entropy(deception,d)
            losses.append((float(loss),len(g)));gp.extend(torch.softmax(goal,1)[:,1].tolist());dp.extend(torch.softmax(deception,1)[:,1].tolist());gl.extend(g.tolist());dl.extend(d.tolist())
    gm=binary_metrics(gl,gp);dm=binary_metrics(dl,dp)
    return dict(goal_accuracy=gm['accuracy'],goal_f1=gm['f1'],goal_auroc=gm['auroc'],
                deception_accuracy=dm['accuracy'],deception_f1=dm['f1'],deception_auroc=dm['auroc'],
                deception_positive_support=dm['positive_support'],deception_negative_support=dm['negative_support'],
                loss=sum(v*n for v,n in losses)/len(records)),gp,dp

def run(output_dir,epochs=10,per_family=64,hidden_size=32,batch_size=32,seed=0,
        training_csv='outputs/data_scaling/training_episodes.csv',scaling_csv='outputs/data_scaling/data_scaling_episodes.csv',
        family_csv='outputs/deception_family_generalization/family_generalization_episodes.csv',validation_families=('scripted_4',)):
    if epochs<1 or batch_size<1:raise ValueError('Positive epochs and batch size required')
    destination=Path(output_dir);destination.mkdir(parents=True,exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    candidates,test=load_stage6(training_csv,scaling_csv,family_csv);train,val=family_split(candidates,validation_families)
    splits={'train':train,'val':val,'test':test};export_dataset(destination/'dataset',splits)
    limited={name:balanced_limit(rows,per_family,seed) for name,rows in splits.items()}
    (destination/'smoke_audit.json').write_text(json.dumps(audit(limited),indent=2))
    loader=DataLoader(DeceptionDataset(limited['train']),batch_size=batch_size,shuffle=True,collate_fn=collate,generator=torch.Generator().manual_seed(seed))
    model=AntiDeceptionGRU(hidden_size);optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    history=[];best=float('inf');best_epoch=0
    for epoch in range(1,epochs+1):
        model.train();total=0
        for x,lengths,g,d in loader:
            optimizer.zero_grad();goal,deception=model(x,lengths)
            loss=nn.functional.cross_entropy(goal,g)+nn.functional.cross_entropy(deception,d)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();total+=float(loss.detach())*len(g)
        validation,_,_=evaluate(model,limited['val'],batch_size)
        history.append(dict(epoch=epoch,train_loss=total/len(limited['train']),**{'val_'+k:v for k,v in validation.items()}))
        if validation['loss']<best:
            best=validation['loss'];best_epoch=epoch
            torch.save(dict(model_state_dict=model.state_dict(),hidden_size=hidden_size,seed=seed,epoch=epoch,
                            input='Defender visible (time,row,column) only',goal_labels={'A':0,'B':1},deception_labels={'honest':0,'deceptive':1}),destination/'anti_deception_gru.pt')
    checkpoint=torch.load(destination/'anti_deception_gru.pt',weights_only=True);model.load_state_dict(checkpoint['model_state_dict'])
    metrics={};predictions=[]
    for split,records in limited.items():
        metrics[split],gp,dp=evaluate(model,records,batch_size)
        for r,g,d in zip(records,gp,dp):predictions.append(dict(split=split,sample_id=r['sample_id'],family=r['family'],goal_label=r['goal_label'],deception_label=r['deception_label'],P_B=g,P_deceptive=d))
    for family in sorted({r['family'] for r in limited['test']}):
        records=[r for r in limited['test'] if r['family']==family];metrics['test_'+family]=evaluate(model,records,batch_size)[0]
    # Make the single-class test limitation visible next to model metrics.
    metrics['test_always_deceptive']=binary_metrics([r['deception_label'] for r in limited['test']],np.ones(len(limited['test'])))
    for name,rows in [('training_metrics.csv',history),('predictions.csv',predictions)]:
        with (destination/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(best_epoch=best_epoch,epochs=epochs,per_family=per_family,seed=seed,hidden_size=hidden_size,
                validation_families=list(validation_families),metrics=metrics,
                note='Smoke only. Validation/test deception AUROC undefined when only deceptive labels are present. Input aliases and limited unique trajectories are in dataset/audit.json.')
    (destination/'results.json').write_text(json.dumps(result,indent=2));return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--per-family',type=int,default=64);parser.add_argument('--hidden-size',type=int,default=32)
    parser.add_argument('--batch-size',type=int,default=32);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--validation-families',nargs='+',default=['scripted_4'])
    args=parser.parse_args();print(json.dumps(run(**vars(args)),indent=2))

if __name__=='__main__':main()
