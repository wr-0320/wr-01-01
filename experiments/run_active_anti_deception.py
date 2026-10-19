"""Stage 9.3 paired passive/active rule-controller evaluation; no PPO."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from deception_rl.envs.gridworld import GridWorldParallelEnv,OBSERVE,TRACK
from deception_rl.agents.active_anti_deception import PassiveAntiDeception,ActiveAntiDeception,DefenderReward,trajectory_evidence
from experiments.analyze_identifiability import FAMILIES,policy
from experiments.train_anti_deception import binary_metrics

ACTION_NAMES={0:'commit_A',1:'commit_B',OBSERVE:'observe',TRACK:'track'}

def episode(family,goal,seed,controller_name,episode_id,reward_config):
    if controller_name not in ('passive','active'):raise ValueError('Unknown controller')
    env=GridWorldParallelEnv();obs,_=env.reset(seed=seed,options={'goal':goal})
    agent=policy(family);agent.reset(seed=seed)
    controller=PassiveAntiDeception() if controller_name=='passive' else ActiveAntiDeception()
    time=0;trace=[];components={'interception_reward':0.,'observation_cost':0.,'false_commitment_penalty':0.};score_t5=None;decision_score=None;intercepted=False;escaped=False
    while env.agents:
        evidence=trajectory_evidence(obs['defender'],time)
        if time==5:score_t5=evidence.deception_probability
        previous_goal=controller.goal;action=controller.act(obs['defender'],time)
        new_commitment=previous_goal is None and controller.goal is not None
        if new_commitment:decision_score=evidence.deception_probability
        trace.append(dict(timestep=time,action=ACTION_NAMES[action],P_deceptive=evidence.deception_probability,predicted_goal='B' if evidence.predicted_goal else 'A',consistent_moves=evidence.consistent_moves,evidence_age=evidence.age,visible_points=int(obs['defender']['trajectory_mask'].sum())))
        obs,rewards,_,_,_=env.step({'deceiver':agent.act(obs['deceiver']),'defender':action})
        intercepted=rewards['defender']>0;escaped=rewards['deceiver']>0
        parts=reward_config.components(action,intercepted,new_commitment,controller.goal,int(goal=='B'))
        for key,value in parts.items():components[key]+=value
        time+=1
    final=trajectory_evidence(obs['defender'],time)
    if decision_score is None:decision_score=final.deception_probability
    # Min path length=6, so t=5 is always evaluated on default GridWorld.
    assert score_t5 is not None
    return dict(controller=controller_name,family=family,episode=episode_id,seed=seed,true_goal=goal,deception_label=int(family!='honest'),detection_probability_t5=score_t5,detection_probability_at_decision=decision_score,detection_probability_final=final.deception_probability,committed=int(controller.goal is not None),false_commitment=int(controller.goal is not None and controller.goal!=int(goal=='B')),commit_timestep=controller.commit_timestep,defender_task_success=int(intercepted),deceiver_task_success=int(escaped),steps=time,**components,episode_return=components['interception_reward']-components['observation_cost']-components['false_commitment_penalty'],trace=trace)

def summarize(rows):
    detection=binary_metrics([r['deception_label'] for r in rows],[r['detection_probability_t5'] for r in rows])
    committed=[r for r in rows if r['committed']]
    return dict(episodes=len(rows),deception_detection_accuracy=detection['accuracy'],deception_f1=detection['f1'],deception_auroc=detection['auroc'],decision_detection_accuracy=float(np.mean([(r['detection_probability_at_decision']>=.5)==r['deception_label'] for r in rows])),false_commitment_rate=sum(r['false_commitment'] for r in committed)/len(committed) if committed else None,false_commitment_episode_rate=float(np.mean([r['false_commitment'] for r in rows])),commitment_coverage=len(committed)/len(rows),mean_commit_timestep=float(np.mean([r['commit_timestep'] for r in committed])) if committed else None,mean_observation_cost=float(np.mean([r['observation_cost'] for r in rows])),defender_task_success=float(np.mean([r['defender_task_success'] for r in rows])),deceiver_task_success=float(np.mean([r['deceiver_task_success'] for r in rows])),mean_episode_return=float(np.mean([r['episode_return'] for r in rows])))

def run(destination,per_family=20,seed=0):
    if per_family<2 or per_family%2:raise ValueError('per_family must be positive even >=2')
    p=Path(destination);p.mkdir(parents=True,exist_ok=True);rows=[];reward=DefenderReward()
    for family in FAMILIES:
        count=per_family*8 if family=='honest' else per_family
        for i in range(count):
            for controller in ['passive','active']:
                rows.append(episode(family,'A' if i%2==0 else 'B',seed+930000+i,controller,i,reward))
    summary={controller:summarize([r for r in rows if r['controller']==controller]) for controller in ['passive','active']}
    (p/'summary.json').write_text(json.dumps(summary,indent=2))
    family_rows=[]
    for controller in summary:
        for family in FAMILIES:family_rows.append(dict(controller=controller,family=family,**summarize([r for r in rows if r['controller']==controller and r['family']==family])))
    for name,data in [('evaluation_episodes.csv',rows),('family_summary.csv',family_rows)]:
        with (p/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in data)
    (p/'config.json').write_text(json.dumps(dict(per_deceptive_family=per_family,honest_episodes=per_family*8,seed=seed,reward=reward.__dict__,detection_time=5,detector='axis-sign reversal heuristic; scores .1/.5/.9, threshold .5, uncalibrated',commitment='irreversible; same goal rule, passive fixed t5, active evidence-based with track near candidate goal (distance <=2)',notes='Paired fixed-policy diagnostic, repeated deterministic routes, no detector fitting or Q/PPO training'),indent=2))
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True);parser.add_argument('--per-family',type=int,default=20);parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();print(json.dumps(run(args.output_dir,args.per_family,args.seed),indent=2))
