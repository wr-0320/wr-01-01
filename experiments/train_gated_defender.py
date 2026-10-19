"""Stage9.6: 512-episode action-masked Defender smoke; rewards unchanged."""
import argparse,hashlib,json,random
from collections import Counter
from pathlib import Path
import numpy as np,torch
from torch import nn
from deception_rl.beliefs import GRUBeliefEstimator
from deception_rl.agents.active_anti_deception import PassiveAntiDeception,ActiveAntiDeception
from models.defender_actor_critic import DefenderActorCritic
from experiments.analyze_identifiability import FAMILIES
from experiments.train_learned_defender import learned_episode,discounted_returns,write_csv
from experiments.run_active_anti_deception import summarize

BASELINE=Path('/workspace/learned-defender-smoke/learned_defender.pt')
RISK=Path('/workspace/stage95-risk-sensitive/risk_sensitive_defender.pt')

def evaluate(model,estimator,per_family,seed,threshold,baseline=None,risk_baseline=None):
    model.eval();rows=[]
    for family in FAMILIES:
        for i in range(per_family*(8 if family=='honest' else 1)):
            goal='A' if i%2==0 else 'B'
            controllers=[('gated',model,None,threshold)]
            if baseline is not None:controllers += [('learned',baseline,None,None),('risk_sensitive',risk_baseline,None,None),('passive',None,PassiveAntiDeception(),None),('active',None,ActiveAntiDeception(),None)]
            for name,net,rule,gate in controllers:
                row=learned_episode(net,estimator,family,goal,seed+i,i,entropy_weight=0.,rule_controller=rule,commitment_entropy_threshold=gate)[0]
                row['controller']=name;rows.append(row)
    return rows

def summary(rows):
    out=summarize(rows);committed=[r for r in rows if r['committed']]
    out.update(average_entropy_before_commitment=float(np.mean([r['entropy_before_commitment'] for r in committed])) if committed else None,
               mean_entropy_commitment_penalty=float(np.mean([r['entropy_commitment_penalty'] for r in rows])),
               mean_base_episode_return=float(np.mean([r['base_episode_return'] for r in rows])),
               immediate_commitment_rate=float(np.mean([r['commit_timestep']==0 for r in rows])))
    return out

def run(destination,threshold=.75,seed=0):
    if not np.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Entropy threshold must be in [0,1]')
    p=Path(destination);p.mkdir(parents=True,exist_ok=True)
    baseline_hash=hashlib.sha256(BASELINE.read_bytes()).hexdigest();risk_hash=hashlib.sha256(RISK.read_bytes()).hexdigest();weights=torch.load(BASELINE,weights_only=True)
    torch.manual_seed(seed);np.random.seed(seed);rng=random.Random(seed)
    estimator=GRUBeliefEstimator.from_checkpoint('outputs/data_scaling/modela.pt')
    for v in estimator.model.parameters():v.requires_grad_(False)
    baseline=DefenderActorCritic(weights['history_size']);baseline.load_state_dict(weights['model_state_dict']);baseline.eval()
    for v in baseline.parameters():v.requires_grad_(False)
    risk_weights=torch.load(RISK,weights_only=True);risk_baseline=DefenderActorCritic(risk_weights['history_size']);risk_baseline.load_state_dict(risk_weights['model_state_dict']);risk_baseline.eval()
    for v in risk_baseline.parameters():v.requires_grad_(False)
    model=DefenderActorCritic(weights['history_size']);model.load_state_dict(weights['model_state_dict']);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    schedule=[]
    for family in FAMILIES:
        for i in range(32*(8 if family=='honest' else 1)):schedule.append((family,'A' if i%2==0 else 'B'))
    rng.shuffle(schedule);training=[];updates=[];validation=[];pending=[];best=-float('inf');best_episode=0
    for episode_id,(family,goal) in enumerate(schedule,1):
        model.train();row,decisions,rewards=learned_episode(model,estimator,family,goal,seed+400000+episode_id,episode_id,training=True,entropy_weight=0.,commitment_entropy_threshold=threshold)
        returns=discounted_returns(rewards)
        pending.extend((logprob,value,entropy,returns[t]) for t,logprob,value,entropy in decisions)
        training.append({k:v for k,v in row.items() if k!='trace'})
        if episode_id%8==0:
            logprobs=torch.stack([r[0] for r in pending]);values=torch.stack([r[1] for r in pending]);entropies=torch.stack([r[2] for r in pending]);target=torch.tensor([r[3] for r in pending])
            advantage=target-values.detach();actor=-(logprobs*advantage).mean();critic=nn.functional.mse_loss(values,target);loss=actor+.5*critic-.01*entropies.mean()
            optimizer.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            updates.append(dict(episode=episode_id,actor_loss=float(actor.detach()),critic_loss=float(critic.detach()),action_entropy=float(entropies.mean().detach()),gradient_norm=float(norm)));pending=[]
        if episode_id%64==0:
            metrics=summary(evaluate(model,estimator,2,seed+500000,threshold));validation.append(dict(training_episode=episode_id,**metrics))
            if metrics['mean_episode_return']>best:
                best=metrics['mean_episode_return'];best_episode=episode_id
                torch.save(dict(model_state_dict=model.state_dict(),history_size=weights['history_size'],training_episode=episode_id,threshold=threshold,seed=seed),p/'gated_defender.pt')
    torch.save(dict(model_state_dict=model.state_dict(),history_size=weights['history_size'],training_episode=512,threshold=threshold,seed=seed),p/'last_gated_defender.pt')
    model.load_state_dict(torch.load(p/'gated_defender.pt',weights_only=True)['model_state_dict'])
    rows=evaluate(model,estimator,10,seed+600000,threshold,baseline=baseline,risk_baseline=risk_baseline)
    results={c:summary([r for r in rows if r['controller']==c]) for c in ['passive','active','learned','risk_sensitive','gated']}
    family=[dict(controller=c,family=f,**summary([r for r in rows if r['controller']==c and r['family']==f])) for c in results for f in FAMILIES]
    for name,data in [('training_episodes.csv',training),('training_updates.csv',updates),('validation.csv',validation),('evaluation_episodes.csv',rows),('family_summary.csv',family)]:write_csv(p/name,data)
    (p/'summary.json').write_text(json.dumps(results,indent=2))
    config=dict(training_episodes=512,training_steps=sum(r['steps'] for r in training),seed=seed,goal_counts=dict(Counter(r['true_goal'] for r in training)),family_counts=dict(Counter(r['family'] for r in training)),initialization='fine-tune existing selected Stage9.4 weights; new Adam state',baseline_path=str(BASELINE),baseline_sha256=baseline_hash,baseline_selected_training_episode=weights['training_episode'],commitment_entropy_threshold=threshold,entropy_penalty_coefficient=0.,entropy_definition='normalized frozen ModelA goal-belief entropy; commit logits masked only when H>threshold',selection='highest validation original reward; 32 validation episodes every64 training episodes',best_training_episode=best_episode,evaluation_episodes_per_controller=160,common_evaluation_reward='all five scored under original Stage9.4 reward; no entropy penalty',algorithm='same small episodic actor-critic, gamma .99, Adam .001, policy entropy .01, value weight .5, batch8, grad clip1',no_additional_training=True)
    config.update(risk_baseline_path=str(RISK),risk_baseline_sha256=risk_hash)
    (p/'config.json').write_text(json.dumps(config,indent=2));assert hashlib.sha256(BASELINE.read_bytes()).hexdigest()==baseline_hash
    assert hashlib.sha256(RISK.read_bytes()).hexdigest()==risk_hash
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True);parser.add_argument('--threshold',type=float,default=.75);parser.add_argument('--seed',type=int,default=0)
    a=parser.parse_args();print(json.dumps(run(a.output_dir,a.threshold,a.seed),indent=2))
