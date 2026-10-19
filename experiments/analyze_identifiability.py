"""Balanced observable-input identifiability analysis; no PPO training."""
from __future__ import annotations
import argparse,csv,hashlib,json,random
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from deception_rl.envs.gridworld import GridWorldParallelEnv
from deception_rl.agents import HonestShortestPath,ScriptedDeception,CommittedDefender,BeliefBasedDefender
from deception_rl.beliefs import GRUBeliefEstimator
from experiments.deception_families import DelayedDeception,DetourDeception,MultiSwitchDeception
from experiments.train_anti_deception import binary_metrics
from models.anti_deception_gru import AntiDeceptionGRU

FAMILIES=('honest',*(f'scripted_{i}' for i in range(1,6)),'delayed_deception','detour_deception','multi_switch_deception')
MODES=('A','B','C')

def policy(family):
    if family=='honest':return HonestShortestPath()
    if family.startswith('scripted_'):return ScriptedDeception(int(family.rsplit('_',1)[1]))
    return {'delayed_deception':DelayedDeception,'detour_deception':DetourDeception,'multi_switch_deception':MultiSwitchDeception}[family]()

def generate_dataset(estimator,per_deceptive_family=32,seed=0):
    if per_deceptive_family<10 or per_deceptive_family%2:raise ValueError('Use even per-family count >=10')
    records=[]
    for family in FAMILIES:
        count=per_deceptive_family*(len(FAMILIES)-1) if family=='honest' else per_deceptive_family
        for episode in range(count):
            goal='A' if episode%2==0 else 'B';episode_seed=seed+900000+episode
            env=GridWorldParallelEnv();obs,_=env.reset(seed=episode_seed,options={'goal':goal})
            agent=policy(family);agent.reset(seed=episode_seed)
            defender=CommittedDefender(5,BeliefBasedDefender(estimator=estimator))
            trajectory=[];beliefs=[]
            while True:
                time=env._step_count
                belief=defender.defender.estimate_belief(obs['defender'])
                if obs['defender']['trajectory_mask'][time]:
                    # Read positions only from the visible Defender observation.
                    point=obs['defender']['trajectory'][time]
                    trajectory.append([time,int(point[0]),int(point[1])]);beliefs.append([belief['A'],belief['B']])
                if not env.agents:break
                obs,_,_,_,_=env.step({'deceiver':agent.act(obs['deceiver']),'defender':defender.act(obs['defender'])})
            records.append(dict(sample_id=f'{family}:{episode}',family=family,goal_label=int(goal=='B'),deception_label=int(family!='honest'),seed=episode_seed,trajectory=trajectory,belief_history=beliefs))
            env.close()
    return records

def features(record,mode):
    if mode not in MODES:raise ValueError('Mode must be A, B or C')
    points=np.asarray(record['trajectory'],dtype=np.float32);position=points[:,1:]/6
    delta=np.zeros_like(position);delta[1:]=position[1:]-position[:-1]
    result=np.zeros((len(points),8),dtype=np.float32);result[:,:4]=np.concatenate([position,delta],axis=1)
    if mode=='B':
        result[:,4]=points[:,0]/30;result[1:,5]=np.diff(points[:,0])/30
    if mode=='C':result[:,6:8]=np.asarray(record['belief_history'],dtype=np.float32)
    return torch.from_numpy(result)

def observable_key(record,mode):
    value=features(record,mode)
    return hashlib.sha256(str(tuple(value.shape)).encode()+value.numpy().tobytes()).hexdigest()

def identifiability(records,mode):
    groups=defaultdict(list)
    for record in records:groups[observable_key(record,mode)].append(record)
    correct=0;conflicts=[];scores=[];labels=[];counts=[]
    for key,group in groups.items():
        c=Counter(r['deception_label'] for r in group);correct+=max(c.values());counts.append((c[1],c[0]))
        labels.extend(r['deception_label'] for r in group);scores.extend([c[1]/len(group)]*len(group))
        if len(c)>1:conflicts.append(dict(input_hash=key,episodes=len(group),honest=c[0],deceptive=c[1],families=dict(Counter(r['family'] for r in group)),minimum_errors=min(c.values())))
    # Oracle F1 on this finite empirical distribution; not a deployable predictor.
    tp=fp=0;best_f1=0.;total_positive=sum(c[0] for c in counts)
    for positive,negative in sorted(counts,key=lambda c:c[0]/sum(c),reverse=True):
        tp+=positive;fp+=negative;fn=total_positive-tp;best_f1=max(best_f1,2*tp/(2*tp+fp+fn))
    return dict(mode=mode,episodes=len(records),unique_inputs=len(groups),conflicting_input_groups=len(conflicts),minimum_errors=sum(c['minimum_errors'] for c in conflicts),empirical_accuracy_ceiling=correct/len(records),empirical_f1_ceiling=best_f1,empirical_auroc_ceiling=binary_metrics(labels,scores)['auroc'],conflicts=conflicts)

def diagnostic_split(records,seed=0):
    """Stratified episode split for fitting only, not unique-trajectory generalization."""
    groups=defaultdict(list);rng=random.Random(seed);splits={'train':[],'val':[],'test':[]}
    for r in records:groups[(r['family'],r['goal_label'])].append(r)
    for key,group in sorted(groups.items()):
        rng.shuffle(group);n=len(group);nval=max(1,round(n*.2));ntest=nval
        splits['train'].extend(group[nval+ntest:]);splits['val'].extend(group[:nval]);splits['test'].extend(group[nval:nval+ntest])
    # Balance the two labels in every split without inventing samples.
    for name,rows in splits.items():
        positive=[r for r in rows if r['deception_label']];negative=[r for r in rows if not r['deception_label']]
        n=min(len(positive),len(negative));rng.shuffle(positive);rng.shuffle(negative);splits[name]=positive[:n]+negative[:n]
    return splits

class ObservableDataset(Dataset):
    def __init__(self,records,mode):self.records=records;self.mode=mode
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        r=self.records[index];return features(r,self.mode),r['goal_label'],r['deception_label']

def collate(batch):
    x,g,d=zip(*batch);return pad_sequence(x,batch_first=True),torch.tensor([len(v) for v in x]),torch.tensor(g),torch.tensor(d)

def evaluate(model,records,mode):
    probabilities=[];goals=[];loss=0.;model.eval()
    with torch.inference_mode():
        for x,lengths,g,d in DataLoader(ObservableDataset(records,mode),batch_size=64,collate_fn=collate):
            gl,dl=model(x,lengths);probabilities.extend(torch.softmax(dl,1)[:,1].tolist());goals.extend(gl.argmax(1).tolist())
            loss+=float(nn.functional.cross_entropy(gl,g)+nn.functional.cross_entropy(dl,d))*len(g)
    result=binary_metrics([r['deception_label'] for r in records],probabilities)
    result.update(loss=loss/len(records),goal_accuracy=float(np.mean([p==r['goal_label'] for p,r in zip(goals,records)])))
    return result,probabilities

def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def run(destination,epochs=20,per_family=32,seed=0):
    if epochs<1:raise ValueError('epochs must be positive')
    p=Path(destination);p.mkdir(parents=True,exist_ok=True)
    checkpoint=Path('outputs/data_scaling/modela.pt');before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    records=generate_dataset(GRUBeliefEstimator.from_checkpoint(checkpoint),per_family,seed);splits=diagnostic_split(records,seed)
    for name,rows in {'all':records,**splits}.items():
        with (p/(name+'.jsonl')).open('w') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
    bounds=[identifiability(records,mode) for mode in MODES]
    (p/'identifiability.json').write_text(json.dumps(bounds,indent=2))
    balance={name:{'episodes':len(rows),'deception_labels':dict(Counter(str(r['deception_label']) for r in rows)),'families':dict(Counter(r['family'] for r in rows))} for name,rows in {'all':records,**splits}.items()}
    for mode in MODES:
        train_keys={observable_key(r,mode) for r in splits['train']}
        balance['overlap_'+mode]={'test_episodes_seen_input_in_train':sum(observable_key(r,mode) in train_keys for r in splits['test']),'test_episodes':len(splits['test'])}
    (p/'balance_and_overlap.json').write_text(json.dumps(balance,indent=2))
    matrix=[]
    for a in FAMILIES:
        row={'family':a};keys={observable_key(r,'A') for r in records if r['family']==a}
        for b in FAMILIES:row[b]=len(keys & {observable_key(r,'A') for r in records if r['family']==b})
        matrix.append(row)
    write_csv(p/'family_observation_alias_matrix.csv',matrix)
    results={};history=[];predictions=[];confusion=[]
    for mode in MODES:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=AntiDeceptionGRU(32,input_size=8);optimizer=torch.optim.Adam(model.parameters(),lr=.003)
        loader=DataLoader(ObservableDataset(splits['train'],mode),batch_size=32,shuffle=True,collate_fn=collate,generator=torch.Generator().manual_seed(seed));best=float('inf')
        for epoch in range(1,epochs+1):
            model.train();total=0.
            for x,lengths,g,d in loader:
                optimizer.zero_grad();gl,dl=model(x,lengths);loss=nn.functional.cross_entropy(gl,g)+nn.functional.cross_entropy(dl,d)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(loss.detach())*len(g)
            metrics,_=evaluate(model,splits['val'],mode);history.append(dict(mode=mode,epoch=epoch,training_loss=total/len(splits['train']),validation_loss=metrics['loss']))
            if metrics['loss']<best:
                best=metrics['loss'];torch.save({'model_state_dict':model.state_dict(),'hidden_size':32,'input_size':8,'mode':mode,'epoch':epoch},p/('gru_'+mode+'.pt'))
        model.load_state_dict(torch.load(p/('gru_'+mode+'.pt'),weights_only=True)['model_state_dict'])
        result,probs=evaluate(model,splits['test'],mode);results[mode]=result
        for r,score in zip(splits['test'],probs):predictions.append(dict(mode=mode,sample_id=r['sample_id'],family=r['family'],label=r['deception_label'],prob_deceptive=score,prediction=int(score>=.5)))
        for family in FAMILIES:
            group=[r for r in predictions if r['mode']==mode and r['family']==family]
            confusion.append(dict(mode=mode,family=family,predicted_honest=sum(not r['prediction'] for r in group),predicted_deceptive=sum(r['prediction'] for r in group),true_deception_label=int(family!='honest')))
    write_csv(p/'training_metrics.csv',history);write_csv(p/'predictions.csv',predictions);write_csv(p/'family_confusion_matrix.csv',confusion)
    (p/'results.json').write_text(json.dumps(results,indent=2))
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before
    return {'results':results,'bounds':bounds,'balance':balance}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True);parser.add_argument('--epochs',type=int,default=20);parser.add_argument('--per-family',type=int,default=32);parser.add_argument('--seed',type=int,default=0)
    a=parser.parse_args();print(json.dumps(run(a.output_dir,a.epochs,a.per_family,a.seed),indent=2))
if __name__=='__main__':main()
