"""Stage9.5: 512-episode entropy-aware Defender smoke, fixed Deceiver policies."""
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

def evaluate(model,estimator,per_family,seed,entropy_weight,baseline=None):
    model.eval();rows=[]
    for family in FAMILIES:
        for i in range(per_family*(8 if family=='honest' else 1)):
            goal='A' if i%2==0 else 'B'
            controllers=[('risk_sensitive',model,None)]
            if baseline is not None:controllers += [('learned',baseline,None),('passive',None,PassiveAntiDeception()),('active',None,ActiveAntiDeception())]
            for name,net,rule in controllers:
                row=learned_episode(net,estimator,family,goal,seed+i,i,entropy_weight=entropy_weight,rule_controller=rule)[0]
                row['controller']=name;rows.append(row)
    return rows

def summary(rows):
    out=summarize(rows);committed=[r for r in rows if r['committed']]
    out.update(average_entropy_before_commitment=float(np.mean([r['entropy_before_commitment'] for r in committed])) if committed else None,
               mean_entropy_commitment_penalty=float(np.mean([r['entropy_commitment_penalty'] for r in rows])),
               mean_base_episode_return=float(np.mean([r['base_episode_return'] for r in rows])),
               immediate_commitment_rate=float(np.mean([r['commit_timestep']==0 for r in rows])))
    return out

def run(destination,entropy_weight=.5,seed=0):
    if not np.isfinite(entropy_weight) or entropy_weight<0:raise ValueError('entropy_weight must be finite and nonnegative')
    p=Path(destination);p.mkdir(parents=True,exist_ok=True)
    baseline_hash=hashlib.sha256(BASELINE.read_bytes()).hexdigest();weights=torch.load(BASELINE,weights_only=True)
    torch.manual_seed(seed);np.random.seed(seed);rng=random.Random(seed)
    estimator=GRUBeliefEstimator.from_checkpoint('outputs/data_scaling/modela.pt')
    for v in estimator.model.parameters():v.requires_grad_(False)
    baseline=DefenderActorCritic(weights['history_size']);baseline.load_state_dict(weights['model_state_dict']);baseline.eval()
    for v in baseline.parameters():v.requires_grad_(False)
    model=DefenderActorCritic(weights['history_size']);model.load_state_dict(weights['model_state_dict']);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    schedule=[]
    for family in FAMILIES:
        for i in range(32*(8 if family=='honest' else 1)):schedule.append((family,'A' if i%2==0 else 'B'))
    rng.shuffle(schedule);training=[];updates=[];validation=[];pending=[];best=-float('inf');best_episode=0
    for episode_id,(family,goal) in enumerate(schedule,1):
        model.train();row,decisions,rewards=learned_episode(model,estimator,family,goal,seed+400000+episode_id,episode_id,training=True,entropy_weight=entropy_weight)
        returns=discounted_returns(rewards)
        pending.extend((logprob,value,entropy,returns[t]) for t,logprob,value,entropy in decisions)
        training.append({k:v for k,v in row.items() if k!='trace'})
        if episode_id%8==0:
            logprobs=torch.stack([r[0] for r in pending]);values=torch.stack([r[1] for r in pending]);entropies=torch.stack([r[2] for r in pending]);target=torch.tensor([r[3] for r in pending])
            advantage=target-values.detach();actor=-(logprobs*advantage).mean();critic=nn.functional.mse_loss(values,target);loss=actor+.5*critic-.01*entropies.mean()
            optimizer.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            updates.append(dict(episode=episode_id,actor_loss=float(actor.detach()),critic_loss=float(critic.detach()),action_entropy=float(entropies.mean().detach()),gradient_norm=float(norm)));pending=[]
        if episode_id%64==0:
            metrics=summary(evaluate(model,estimator,2,seed+500000,entropy_weight));validation.append(dict(training_episode=episode_id,**metrics))
            if metrics['mean_episode_return']>best:
                best=metrics['mean_episode_return'];best_episode=episode_id
                torch.save(dict(model_state_dict=model.state_dict(),history_size=weights['history_size'],training_episode=episode_id,entropy_weight=entropy_weight,seed=seed),p/'risk_sensitive_defender.pt')
    torch.save(dict(model_state_dict=model.state_dict(),history_size=weights['history_size'],training_episode=512,entropy_weight=entropy_weight,seed=seed),p/'last_defender.pt')
    model.load_state_dict(torch.load(p/'risk_sensitive_defender.pt',weights_only=True)['model_state_dict'])
    rows=evaluate(model,estimator,10,seed+600000,entropy_weight,baseline=baseline)
    results={c:summary([r for r in rows if r['controller']==c]) for c in ['passive','active','learned','risk_sensitive']}
    family=[dict(controller=c,family=f,**summary([r for r in rows if r['controller']==c and r['family']==f])) for c in results for f in FAMILIES]
    for name,data in [('training_episodes.csv',training),('training_updates.csv',updates),('validation.csv',validation),('evaluation_episodes.csv',rows),('family_summary.csv',family)]:write_csv(p/name,data)
    (p/'summary.json').write_text(json.dumps(results,indent=2))
    config=dict(training_episodes=512,training_steps=sum(r['steps'] for r in training),seed=seed,goal_counts=dict(Counter(r['true_goal'] for r in training)),family_counts=dict(Counter(r['family'] for r in training)),initialization='fine-tune existing selected Stage9.4 weights; new Adam state',baseline_path=str(BASELINE),baseline_sha256=baseline_hash,baseline_selected_training_episode=weights['training_episode'],entropy_penalty_coefficient=entropy_weight,entropy_definition='normalized binary frozen ModelA goal-belief entropy, measured before first commit only',selection='highest validation risk reward; 32 validation episodes every64 training episodes',best_training_episode=best_episode,evaluation_episodes_per_controller=160,common_evaluation_reward='all four scored under risk reward and original reward separately',algorithm='same small episodic actor-critic, gamma .99, Adam .001, policy entropy .01, value weight .5, batch8, grad clip1',no_additional_training=True)
    (p/'config.json').write_text(json.dumps(config,indent=2));assert hashlib.sha256(BASELINE.read_bytes()).hexdigest()==baseline_hash
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True);parser.add_argument('--entropy-weight',type=float,default=.5);parser.add_argument('--seed',type=int,default=0)
    a=parser.parse_args();print(json.dumps(run(a.output_dir,a.entropy_weight,a.seed),indent=2))
