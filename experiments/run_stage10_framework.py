"""512-data/512-evaluation-episode Two-level anti-deception smoke baseline."""
import argparse,csv,hashlib,json,random
from collections import Counter
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from models.anti_deception_gru import AntiDeceptionGRU
from models.intent_gru import trajectory_to_features
from models.defender_actor_critic import DefenderActorCritic
from deception_rl.agents.learned_defender import LearnedActiveDefender
from deception_rl.beliefs import GRUBeliefEstimator
from deception_rl.beliefs.two_level import DeceptionAwareBelief,HonestPrototypeReasoner,PassiveTrajectoryClassifier,TwoLevelAntiDeception
from deception_rl.agents.active_anti_deception import ActiveAntiDeception,DefenderReward,trajectory_evidence
from deception_rl.envs.gridworld import GridWorldParallelEnv
from experiments.analyze_identifiability import policy
from experiments.run_active_anti_deception import summarize
from experiments.train_anti_deception import binary_metrics
from experiments.train_learned_defender import write_csv

SEEN=('scripted_1','scripted_2','scripted_3')
UNSEEN=('scripted_4','scripted_5','delayed_deception','detour_deception','multi_switch_deception')

class PrefixDataset(Dataset):
    def __init__(self,records):
        self.examples=[]
        for r in records:
            n=len(r['trajectory'])
            for end in range(1,n+1):self.examples.append((trajectory_to_features(r['trajectory'][:end]),r['goal_label'],r['deception_label'],1/n))
    def __len__(self):return len(self.examples)
    def __getitem__(self,index):return self.examples[index]

def collate(batch):
    from torch.nn.utils.rnn import pad_sequence
    x,g,d,w=zip(*batch);return pad_sequence(x,batch_first=True),torch.tensor([len(v) for v in x]),torch.tensor(g),torch.tensor(d),torch.tensor(w)

def build_data(path,seed=0):
    records=[json.loads(l) for l in Path(path).read_text().splitlines()];rng=random.Random(seed);train=[];val=[]
    # Exactly512 existing episode IDs, balanced labels/goals; no unseen families.
    quotas={'honest':256,'scripted_1':86,'scripted_2':86,'scripted_3':84}
    for family,n in quotas.items():
        for goal in (0,1):
            group=[r for r in records if r['family']==family and r['goal_label']==goal];rng.shuffle(group)
            validation_count={'honest':32,'scripted_1':11,'scripted_2':11,'scripted_3':10}[family]
            if len(group)<n//2+validation_count:raise ValueError('Insufficient existing data')
            train.extend(group[:n//2]);val.extend(group[n//2:n//2+validation_count])
    assert len(train)==512 and len(val)==128
    assert sum(r['deception_label'] for r in train)==256 and sum(r['deception_label'] for r in val)==64
    assert not {r['sample_id'] for r in train}&{r['sample_id'] for r in val}
    return train,val

def metrics(model,records):
    goal_labels=[];deception_labels=[];gp=[];dp=[];losses=[];model.eval()
    with torch.inference_mode():
        for r in records:
            x=trajectory_to_features(r['trajectory']).unsqueeze(0);g,d=model(x,torch.tensor([x.shape[1]]))
            gp.append(float(torch.softmax(g,1)[0,1]));dp.append(float(torch.softmax(d,1)[0,1]));goal_labels.append(r['goal_label']);deception_labels.append(r['deception_label'])
            losses.append(float(nn.functional.cross_entropy(g,torch.tensor([r['goal_label']]))+nn.functional.cross_entropy(d,torch.tensor([r['deception_label']]))))
    gm=binary_metrics(goal_labels,gp);dm=binary_metrics(deception_labels,dp)
    return dict(goal_accuracy=gm['accuracy'],deception_accuracy=dm['accuracy'],deception_f1=dm['f1'],deception_auroc=dm['auroc'],loss=float(np.mean(losses)))

def fit(train,val,destination,epochs=10,seed=0):
    torch.manual_seed(seed);model=AntiDeceptionGRU(32);optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    loader=DataLoader(PrefixDataset(train),batch_size=64,shuffle=True,collate_fn=collate,generator=torch.Generator().manual_seed(seed));logs=[];best=float('inf')
    for epoch in range(1,epochs+1):
        model.train();total=0.;weight_total=0.
        for x,lengths,g,d,w in loader:
            optimizer.zero_grad();gl,dl=model(x,lengths);individual=nn.functional.cross_entropy(gl,g,reduction='none')+nn.functional.cross_entropy(dl,d,reduction='none');loss=(individual*w).sum()/w.sum()
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float((individual.detach()*w).sum());weight_total+=float(w.sum())
        validation=metrics(model,val);logs.append(dict(epoch=epoch,train_loss=total/weight_total,**{'val_'+k:v for k,v in validation.items()}))
        if validation['loss']<best:
            best=validation['loss'];torch.save(dict(model_state_dict=model.state_dict(),hidden_size=32,epoch=epoch),destination/'belief_model.pt')
    model.load_state_dict(torch.load(destination/'belief_model.pt',weights_only=True)['model_state_dict']);write_csv(destination/'training_metrics.csv',logs);return model

def run_episode(controller,family,goal,seed,episode_id,name,group):
    env=GridWorldParallelEnv();observation,_=env.reset(seed=seed,options={'goal':goal});agent=policy(family);agent.reset(seed=seed)
    controller.reset();t=0;score5=None;goal5=None;model_score5=None;decision_score=None;components={'interception_reward':0.,'observation_cost':0.,'false_commitment_penalty':0.};trace=[]
    while env.agents:
        previous=controller.goal;action=controller.act(observation['defender'],t)
        info=getattr(controller,'info',None)
        score=info['deception_probability'] if info else trajectory_evidence(observation['defender'],t).deception_probability
        if t==5:
            score5=score;goal5=int(np.argmax(info['goal_belief'])) if info else None
            model_score5=info.get('model_deception_probability',info['deception_probability']) if info else None
        new=previous is None and controller.goal is not None
        if new:decision_score=score
        trace.append(dict(timestep=t,action=('commit_A','commit_B','observe','track')[action],P_deceptive=score,info=info))
        observation,rewards,_,_,_=env.step({'deceiver':agent.act(observation['deceiver']),'defender':action})
        intercepted=rewards['defender']>0;escaped=rewards['deceiver']>0
        parts=DefenderReward().components(action,intercepted,new,controller.goal,int(goal=='B'))
        for k,v in parts.items():components[k]+=v
        t+=1
    assert score5 is not None
    return dict(controller=name,group=group,family=family,episode=episode_id,seed=seed,true_goal=goal,deception_label=int(family!='honest'),detection_probability_t5=score5,detection_probability_at_decision=score5 if decision_score is None else decision_score,detection_probability_final=score,module1_deception_probability_t5=model_score5,goal_correct_t5=None if goal5 is None else int(goal5==int(goal=='B')),committed=int(controller.goal is not None),false_commitment=int(controller.goal is not None and controller.goal!=int(goal=='B')),commit_timestep=controller.commit_timestep,defender_task_success=int(intercepted),deceiver_task_success=int(escaped),steps=t,**components,episode_return=components['interception_reward']-components['observation_cost']-components['false_commitment_penalty'],trace=trace)

def summarize_group(rows):
    result=summarize(rows);committed=[r for r in rows if r['committed']];result['commitment_accuracy']=1-result['false_commitment_rate'] if committed else None
    return result

def run(destination,seed=0):
    p=Path(destination);p.mkdir(parents=True,exist_ok=True);source=Path('/workspace/stage9-anti-deception/dataset/train.jsonl')
    train,val=build_data(source,seed)
    for name,records in [('train',train),('val',val)]:
        with (p/(name+'.jsonl')).open('w') as f:
            for r in records:f.write(json.dumps(r)+'\n')
    model=fit(train,val,p,seed=seed);belief=DeceptionAwareBelief(model);reasoner=HonestPrototypeReasoner(train)
    prior_path=Path('/workspace/learned-defender-smoke/learned_defender.pt');prior_hash=hashlib.sha256(prior_path.read_bytes()).hexdigest();prior=torch.load(prior_path,weights_only=True);actor=DefenderActorCritic(prior['history_size']);actor.load_state_dict(prior['model_state_dict']);actor.eval();goal_estimator=GRUBeliefEstimator.from_checkpoint('outputs/data_scaling/modela.pt')
    controllers={'passive_classifier':PassiveTrajectoryClassifier(belief),'rule_active':ActiveAntiDeception(),'learned_active':LearnedActiveDefender(actor,goal_estimator),'two_level':TwoLevelAntiDeception(belief,reasoner)}
    rows=[];scenario=0
    for group,families in [('seen',SEEN),('unseen',UNSEEN)]:
        counts=[('honest',len(families)*8)]+[(family,8) for family in families]
        for family,count in counts:
            for i in range(count):
                for name,controller in controllers.items():rows.append(run_episode(controller,family,'A' if i%2==0 else 'B',seed+1000000+scenario,scenario,name,group))
                scenario+=1
    assert scenario==128 and len(rows)==512
    comparison=[dict(controller=name,group=group,**summarize_group([r for r in rows if r['controller']==name and (group=='overall' or r['group']==group)])) for name in controllers for group in ['seen','unseen','overall']]
    write_csv(p/'comparison.csv',comparison);write_csv(p/'evaluation_episodes.csv',rows)
    family=[dict(controller=name,family=f,**summarize_group([r for r in rows if r['controller']==name and r['family']==f])) for name in controllers for f in ('honest',*SEEN,*UNSEEN)];write_csv(p/'family_comparison.csv',family)
    config=dict(seed=seed,supervised_training_episodes=512,validation_episode_records=128,epochs=10,prefix_weighting='1/visible-length per episode',training_families=sorted({r['family'] for r in train}),evaluation_total_episodes=512,paired_scenarios=128,evaluation_groups='seen48 scenarios/unseen80; each has equally many honest controls',input='visible timed trajectory only; belief history optional and disabled in first baseline',counterfactual='honest starts from training data extended by public row-first shortest path; observed self-compatibility1 minus max exp(-mean normalized distance/.2)',fusion='mean(model deception probability,counterfactual score)',controller='uncertainty H>.85 or deception>=.6 acquires information; commit>=t5 with fresh consistent moves',source_dataset=str(source),source_dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),learned_baseline_sha256=prior_hash,model_metrics={'train':metrics(model,train),'val':metrics(model,val)},limitations='Repeated observable trajectories overlap train/val; source visibility differs from adaptive views; smoke not independent trajectory generalization')
    module_metrics={}
    for name in ('passive_classifier','two_level'):
        for group in ('seen','unseen'):
            subset=[r for r in rows if r['controller']==name and r['group']==group]
            dm=binary_metrics([r['deception_label'] for r in subset],[r['module1_deception_probability_t5'] for r in subset])
            module_metrics[name+'_'+group]=dict(goal_accuracy=float(np.mean([r['goal_correct_t5'] for r in subset])),deception_accuracy=dm['accuracy'],deception_f1=dm['f1'],deception_auroc=dm['auroc'])
    (p/'belief_model_metrics.json').write_text(json.dumps(module_metrics,indent=2))
    (p/'config.json').write_text(json.dumps(config,indent=2));assert hashlib.sha256(prior_path.read_bytes()).hexdigest()==prior_hash
    return comparison

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True);parser.add_argument('--seed',type=int,default=0);a=parser.parse_args();print(json.dumps(run(a.output_dir,a.seed),indent=2))
