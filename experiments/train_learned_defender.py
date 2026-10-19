"""Small episodic actor-critic Defender training against unchanged scripted families."""
import argparse,csv,hashlib,json,random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from deception_rl.envs.gridworld import GridWorldParallelEnv
from deception_rl.beliefs import GRUBeliefEstimator
from deception_rl.agents.active_anti_deception import DefenderReward,trajectory_evidence
from deception_rl.agents.learned_defender import ENV_ACTIONS
from models.defender_actor_critic import DefenderActorCritic,encode_state,gate_commit_logits
from experiments.analyze_identifiability import FAMILIES,policy
from experiments.run_active_anti_deception import episode as rule_episode,summarize


def discounted_returns(rewards,gamma=.99):
    values=[];value=0.
    for reward in reversed(rewards):value=reward+gamma*value;values.append(value)
    return list(reversed(values))

def learned_episode(model,estimator,family,goal,seed,episode_id,training=False,entropy_weight=0.,rule_controller=None,commitment_entropy_threshold=None):
    if not np.isfinite(entropy_weight) or entropy_weight<0:raise ValueError('entropy_weight must be finite and nonnegative')
    if rule_controller is not None and training:raise ValueError('Rule controllers cannot be trained here')
    if commitment_entropy_threshold is not None:
        if not np.isfinite(commitment_entropy_threshold) or not 0<=commitment_entropy_threshold<=1:raise ValueError('Entropy threshold must be in [0,1]')
        if rule_controller is not None:raise ValueError('A gate is only applied to the learned policy')
    env=GridWorldParallelEnv();obs,_=env.reset(seed=seed,options={'goal':goal});agent=policy(family);agent.reset(seed=seed)
    reward_config=DefenderReward();committed=None;commit_time=None;time=0;score5=None;decision_score=None;commit_entropy=None
    gate_blocked=0;gate_allowed=0;refrained_when_allowed=0
    decisions=[];rewards=[];trace=[];components={'interception_reward':0.,'observation_cost':0.,'false_commitment_penalty':0.,'entropy_commitment_penalty':0.}
    while env.agents:
        evidence=trajectory_evidence(obs['defender'],time)
        if time==5:score5=evidence.deception_probability
        new=False
        if committed is None:
            history,scalars=encode_state(obs['defender'],time,estimator)
            if rule_controller is None:
                with torch.set_grad_enabled(training):
                    logits,value=model(history,scalars)
                    logits=gate_commit_logits(logits,scalars[:,3],commitment_entropy_threshold)
                    distribution=torch.distributions.Categorical(logits=logits)
                    choice=distribution.sample() if training else logits.argmax(1)
                    index=int(choice);action=ENV_ACTIONS[index]
                    if training:decisions.append((time,distribution.log_prob(choice).squeeze(),value.squeeze(),distribution.entropy().squeeze()))
            else:
                action=rule_controller.act(obs['defender'],time)
                index={2:0,3:1,0:2,1:3}[action]
            if commitment_entropy_threshold is not None:
                allowed=float(scalars[0,3])<=commitment_entropy_threshold
                gate_allowed+=int(allowed);gate_blocked+=int(not allowed)
                refrained_when_allowed+=int(allowed and index<2)
                if index>=2:assert allowed
            if index>=2:
                committed=index-2;commit_time=time;new=True;decision_score=evidence.deception_probability
                commit_entropy=float(scalars[0,3])
        else:action=committed
        trace.append(dict(timestep=time,action=('commit_A','commit_B','observe','track')[action],P_deceptive=evidence.deception_probability,visible_points=int(obs['defender']['trajectory_mask'].sum())))
        obs,game_rewards,_,_,_=env.step({'deceiver':agent.act(obs['deceiver']),'defender':action})
        intercepted=game_rewards['defender']>0;escaped=game_rewards['deceiver']>0
        parts=reward_config.components(action,intercepted,new,committed,int(goal=='B'))
        parts['entropy_commitment_penalty']=entropy_weight*commit_entropy if new else 0.
        for k,v in parts.items():components[k]+=v
        rewards.append(parts['interception_reward']-parts['observation_cost']-parts['false_commitment_penalty']-parts['entropy_commitment_penalty']);time+=1
    final=trajectory_evidence(obs['defender'],time)
    if decision_score is None:decision_score=final.deception_probability
    assert score5 is not None
    row=dict(controller='learned',family=family,episode=episode_id,seed=seed,true_goal=goal,deception_label=int(family!='honest'),detection_probability_t5=score5,detection_probability_at_decision=decision_score,detection_probability_final=final.deception_probability,committed=int(committed is not None),false_commitment=int(committed is not None and committed!=int(goal=='B')),commit_timestep=commit_time,defender_task_success=int(intercepted),deceiver_task_success=int(escaped),steps=time,commitment_entropy_threshold=commitment_entropy_threshold,gate_blocked_decisions=gate_blocked,gate_allowed_decisions=gate_allowed,refrained_when_gate_allowed=refrained_when_allowed,entropy_before_commitment=commit_entropy,**components,episode_return=sum(rewards),base_episode_return=sum(rewards)+components['entropy_commitment_penalty'],trace=trace)
    env.close();return row,decisions,rewards


def evaluate(model,estimator,per_family,seed,include_rules=False):
    model.eval();rows=[]
    for family in FAMILIES:
        count=per_family*8 if family=='honest' else per_family
        for i in range(count):
            goal='A' if i%2==0 else 'B'
            rows.append(learned_episode(model,estimator,family,goal,seed+i,i)[0])
            if include_rules:
                rows.extend(rule_episode(family,goal,seed+i,name,i,DefenderReward()) for name in ['passive','active'])
    return rows


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)


def run(destination,episodes=512,seed=0,batch_episodes=8,validation_interval=64,test_per_family=10):
    if episodes<16 or episodes%16:raise ValueError('Training episode count must be a positive multiple of 16')
    if batch_episodes<1 or validation_interval<1 or test_per_family<2 or test_per_family%2:raise ValueError('Invalid smoke/evaluation settings')
    p=Path(destination);p.mkdir(parents=True,exist_ok=True)
    checkpoint=Path('outputs/data_scaling/modela.pt');fixed_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    torch.manual_seed(seed);np.random.seed(seed);rng=random.Random(seed)
    estimator=GRUBeliefEstimator.from_checkpoint(checkpoint)
    for parameter in estimator.model.parameters():parameter.requires_grad_(False)
    model=DefenderActorCritic();initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    schedule=[]
    for family in FAMILIES:
        count=episodes//16*(8 if family=='honest' else 1)
        for i in range(count):schedule.append((family,'A' if i%2==0 else 'B'))
    rng.shuffle(schedule);training=[];updates=[];validation=[];pending=[];best=-float('inf');best_episode=0
    for i,(family,goal) in enumerate(schedule,1):
        model.train();row,decisions,rewards=learned_episode(model,estimator,family,goal,seed+100000+i,i,training=True)
        returns=discounted_returns(rewards)
        for t,logprob,value,entropy in decisions:pending.append((logprob,value,entropy,returns[t]))
        training.append({k:v for k,v in row.items() if k!='trace'})
        if i%batch_episodes==0 or i==episodes:
            logprob=torch.stack([x[0] for x in pending]);values=torch.stack([x[1] for x in pending]);entropies=torch.stack([x[2] for x in pending]);targets=torch.tensor([x[3] for x in pending])
            advantages=targets-values.detach();actor_loss=-(logprob*advantages).mean();critic_loss=nn.functional.mse_loss(values,targets);loss=actor_loss+.5*critic_loss-.01*entropies.mean()
            optimizer.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            updates.append(dict(episode=i,decision_samples=len(pending),actor_loss=float(actor_loss.detach()),critic_loss=float(critic_loss.detach()),entropy=float(entropies.mean().detach()),gradient_norm=float(norm)))
            pending=[]
        if i%validation_interval==0 or i==episodes:
            rows=evaluate(model,estimator,2,seed+200000);metrics=summarize(rows);validation.append(dict(training_episode=i,**metrics))
            if metrics['mean_episode_return']>best:
                best=metrics['mean_episode_return'];best_episode=i
                torch.save(dict(model_state_dict=model.state_dict(),history_size=16,training_episode=i,seed=seed,action_names=['observe','track','commit_A','commit_B'],state='goal belief(2), deception probability(1), entropy(1), learned history embedding(16)'),p/'learned_defender.pt')
    model.load_state_dict(torch.load(p/'learned_defender.pt',weights_only=True)['model_state_dict'])
    rows=evaluate(model,estimator,test_per_family,seed+300000,include_rules=True)
    summary={c:summarize([r for r in rows if r['controller']==c]) for c in ['passive','active','learned']}
    family_rows=[dict(controller=c,family=f,**summarize([r for r in rows if r['controller']==c and r['family']==f])) for c in summary for f in FAMILIES]
    for name,data in [('training_episodes.csv',training),('training_updates.csv',updates),('validation.csv',validation),('evaluation_episodes.csv',rows),('family_summary.csv',family_rows)]:write_csv(p/name,data)
    (p/'summary.json').write_text(json.dumps(summary,indent=2))
    config=dict(training_episodes=episodes,training_steps=sum(r['steps'] for r in training),training_goal_counts=dict(Counter(r['true_goal'] for r in training)),training_family_counts=dict(Counter(r['family'] for r in training)),best_training_episode=best_episode,validation_episodes_per_check=32,test_episodes_per_controller=test_per_family*16,seed=seed,algorithm='episodic actor-critic with MC discounted returns; no PPO',gamma=.99,learning_rate=.001,entropy_coefficient=.01,value_loss_coefficient=.5,batch_episodes=batch_episodes,model_parameters=sum(v.numel() for v in model.parameters()),model_changed=any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items()),frozen_goal_model_sha256=fixed_hash,reward=DefenderReward().__dict__,commitment='irreversible; policy actions only until commit, terminal reward propagated to commit decision',selection='maximum validation mean return, no test selection')
    (p/'config.json').write_text(json.dumps(config,indent=2));assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==fixed_hash
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True);parser.add_argument('--episodes',type=int,default=512);parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();print(json.dumps(run(args.output_dir,args.episodes,args.seed),indent=2))
