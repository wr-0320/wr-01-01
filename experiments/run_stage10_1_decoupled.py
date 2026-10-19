"""512 total simulator episodes: calibration128 + paired comparison384."""
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from models.anti_deception_gru import AntiDeceptionGRU
from deception_rl.beliefs.two_level import DeceptionAwareBelief,HonestPrototypeReasoner,TwoLevelAntiDeception
from deception_rl.beliefs.decoupled import LikelihoodRatioReasoner,EvidenceCalibrator,calibration_features,CalibratedGoalInference
from deception_rl.agents.decoupled_defender import DecoupledAntiDeception
from deception_rl.agents.active_anti_deception import ActiveAntiDeception
from deception_rl.envs.gridworld import GridWorldParallelEnv,TRACK
from experiments.analyze_identifiability import policy
from experiments.run_stage10_framework import SEEN,UNSEEN,run_episode,summarize_group
from experiments.train_learned_defender import write_csv
from experiments.train_anti_deception import binary_metrics

SOURCE=Path('/workspace/stage10-two-level')

def collect(raw,reasoner):
    episodes=[];samples=[];quota={'honest':64,'scripted_1':22,'scripted_2':22,'scripted_3':20}
    for family,n in quota.items():
        for i in range(n):
            goal=i%2;identifier=f'{family}:{i}';env=GridWorldParallelEnv();obs,_=env.reset(seed=1100000+i,options={'goal':goal});agent=policy(family);agent.reset();t=0
            episodes.append(dict(sample_id=identifier,family=family,goal_label=goal,deception_label=int(family!='honest')))
            while env.agents:
                if t<=5:
                    info=raw.predict(obs['defender'],t);cf=reasoner.likelihoods(info['trajectory']);features=calibration_features(info,cf,t)
                    samples.append(dict(sample_id=identifier,timestep=t,features=features.tolist(),goal_label=goal,deception_label=int(family!='honest')))
                obs,_,_,_,_=env.step({'deceiver':agent.act(obs['deceiver']),'defender':TRACK});t+=1
            env.close()
    assert len(episodes)==128
    return episodes,samples

def fit_calibration(episodes,samples,p,seed):
    rng=random.Random(seed);validation_ids=set()
    for family in ('honest',*SEEN):
        for goal in (0,1):
            group=[r for r in episodes if r['family']==family and r['goal_label']==goal];rng.shuffle(group)
            count=6 if family=='honest' else 2
            validation_ids.update(r['sample_id'] for r in group[:count])
    train=[r for r in samples if r['sample_id'] not in validation_ids];val=[r for r in samples if r['sample_id'] in validation_ids]
    torch.manual_seed(seed);calibrator=EvidenceCalibrator();optimizer=torch.optim.Adam(calibrator.parameters(),lr=.03)
    x=torch.tensor([r['features'] for r in train]);y=torch.tensor([[r['goal_label'],r['deception_label']] for r in train],dtype=torch.float32)
    vx=torch.tensor([r['features'] for r in val]);vy=torch.tensor([[r['goal_label'],r['deception_label']] for r in val],dtype=torch.float32)
    best=float('inf');logs=[]
    for epoch in range(150):
        optimizer.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(calibrator(x),y);loss.backward();optimizer.step()
        with torch.inference_mode():v=float(nn.functional.binary_cross_entropy_with_logits(calibrator(vx),vy))
        logs.append(dict(epoch=epoch+1,training_loss=float(loss.detach()),validation_loss=v))
        if v<best:best=v;torch.save(dict(model_state_dict=calibrator.state_dict(),epoch=epoch+1),p/'calibrator.pt')
    calibrator.load_state_dict(torch.load(p/'calibrator.pt',weights_only=True)['model_state_dict'])
    with torch.inference_mode():prob=torch.sigmoid(calibrator(vx)).numpy()
    out={}
    for column,name in [(0,'goal'),(1,'deception')]:
        labels=vy[:,column].numpy();metrics=binary_metrics(labels,prob[:,column]);metrics['brier_score']=float(np.mean((labels-prob[:,column])**2));out[name]=metrics
    write_csv(p/'calibration_training.csv',logs)
    (p/'calibration_metrics.json').write_text(json.dumps(dict(training_episode_ids=104,validation_episode_ids=24,training_snapshots=len(train),validation_snapshots=len(val),metrics=out,validation_ids=sorted(validation_ids)),indent=2))
    return calibrator

def run(destination,seed=0):
    p=Path(destination);p.mkdir(parents=True,exist_ok=True)
    old=torch.load(SOURCE/'belief_model.pt',weights_only=True);model=AntiDeceptionGRU(old['hidden_size']);model.load_state_dict(old['model_state_dict']);model.eval()
    for v in model.parameters():v.requires_grad_(False)
    raw=DeceptionAwareBelief(model);training=[json.loads(l) for l in (SOURCE/'train.jsonl').read_text().splitlines()];reasoner=LikelihoodRatioReasoner(training)
    episodes,samples=collect(raw,reasoner);calibrator=fit_calibration(episodes,samples,p,seed)
    (p/'calibration_episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in episodes)+'\n');(p/'calibration_snapshots.jsonl').write_text('\n'.join(json.dumps(r) for r in samples)+'\n')
    inference=CalibratedGoalInference(raw,reasoner,calibrator)
    controllers={'rule_active':ActiveAntiDeception(),'stage10':TwoLevelAntiDeception(raw,HonestPrototypeReasoner(training)),'stage10_1':DecoupledAntiDeception(inference)}
    rows=[];scenario=0
    for group,families in [('seen',SEEN),('unseen',UNSEEN)]:
        for family,n in [('honest',len(families)*8)]+[(f,8) for f in families]:
            for i in range(n):
                for name,controller in controllers.items():rows.append(run_episode(controller,family,'A' if i%2==0 else 'B',1200000+scenario,scenario,name,group))
                scenario+=1
    assert scenario==128 and len(rows)==384
    comparison=[dict(controller=name,group=group,**summarize_group([r for r in rows if r['controller']==name and (group=='overall' or r['group']==group)])) for name in controllers for group in ['seen','unseen','overall']]
    write_csv(p/'comparison.csv',comparison);write_csv(p/'evaluation_episodes.csv',rows)
    family=[dict(controller=name,family=f,**summarize_group([r for r in rows if r['controller']==name and r['family']==f])) for name in controllers for f in ('honest',*SEEN,*UNSEEN)];write_csv(p/'family_comparison.csv',family)
    (p/'config.json').write_text(json.dumps(dict(seed=seed,total_simulator_episodes=512,calibration_episodes=128,evaluation_episodes=384,evaluation_paired_scenarios=128,calibration_families=['honest',*SEEN],frozen_gru=True,no_policy_training=True,no_reward_changes=True,calibrator='18-parameter evidence affine calibration; frozen model/prototypes;150 small full-batch calibration epochs',likelihoods='same distance kernel; mean over equal-weight candidate goals/prototypes; LR=Ldeceptive/Lhonest',action_scoring='expected interception - sensing cost - .5*false commitment risk; heuristic information gain observe .03/track .08 scaled by entropy and deception',arrival_risk='belief mass of candidate goals reachable next step from last observed position and elapsed time',limitations='single fixed map, repeated routes; heuristic information-gain approximation; no formal guarantee'),indent=2))
    return comparison

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True);parser.add_argument('--seed',type=int,default=0);a=parser.parse_args();print(json.dumps(run(a.output_dir,a.seed),indent=2))
