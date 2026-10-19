"""Stage 7.3: reward-only ablations against frozen Model A."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from deception_rl.beliefs import GRUBeliefEstimator
from experiments.ppo_deceiver_env import PPODeceiverEnv
from experiments.run_ppo_stage72 import Stage72EpisodeCallback

CONFIGS={'R1_task_dominant':(5.0,0.05,False),'R2_balanced':(1.0,0.25,False),'R3_delayed_deception':(1.0,0.25,True)}

class RewardDebugEnv(PPODeceiverEnv):
    """Keep base observations/dynamics/costs; adjust success and bonus only."""
    def __init__(self,estimator,config):
        success,bonus,delayed=config
        super().__init__(estimator,step_commit=5,belief_observation=True,
                         task_reward_scale=1.0,misdirection_reward=bonus,
                         movement_cost=0.01,detour_cost=0.05)
        self.success_reward=success;self.bonus=bonus;self.delayed=delayed
    def step(self,action):
        obs,reward,terminated,truncated,info=super().step(action)
        original_task=info['task_reward'];original_bonus=info['deception_reward']
        task=self.success_reward if original_task>0 else original_task
        bonus=original_bonus
        if self.delayed:
            bonus=self.bonus if original_task>0 and self._commitment_correct==0 else 0.0
        info['task_reward']=task;info['deception_reward']=bonus
        self._episode_reward_components['task_reward']+=task-original_task
        self._episode_reward_components['deception_reward']+=bonus-original_bonus
        reward+=task-original_task+bonus-original_bonus
        if terminated or truncated:info['episode_metrics']=self.episode_metrics()
        return obs,float(reward),terminated,truncated,info

def write(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def evaluate(model,estimator,config,name):
    rows=[]
    for episode in range(50):
        goal='A' if episode%2==0 else 'B';seed=230024+episode
        env=RewardDebugEnv(estimator,config);obs,_=env.reset(seed=seed,options={'goal':goal})
        ret=0.;correct=[];collisions=0;done=False
        while not done:
            belief=env.fixed_defender.defender.estimate_belief(env._observations['defender'])
            correct.append(int(('A' if belief['A']>=belief['B'] else 'B')==goal))
            action,_=model.predict(obs,deterministic=True);action=int(action)
            previous=env._positions[-1].copy()
            obs,reward,term,trunc,info=env.step(action);ret+=reward;done=term or trunc
            collisions+=int(action!=4 and env._positions[-1]==previous)
        belief=env.fixed_defender.defender.estimate_belief(env._observations['defender'])
        correct.append(int(('A' if belief['A']>=belief['B'] else 'B')==goal))
        m=env.episode_metrics();a=m['action_sequence'];n=len(a)
        assert m['commitment_accuracy'] is not None
        assert abs(ret-sum(m[k] for k in ['task_reward','deception_reward','movement_cost','detour_cost']))<1e-6
        rows.append(dict(experiment=name,episode=episode,seed=seed,episode_return=ret,
                         belief_accuracy=float(np.mean(correct)),deception_success=1-m['commitment_accuracy'],
                         stay_count=a.count(4),boundary_collision_count=collisions,steps=n,
                         stay_ratio=a.count(4)/n,boundary_collision_ratio=collisions/n,**m))
        env.close()
    return rows

def run(destination):
    p=Path(destination);p.mkdir(parents=True,exist_ok=True)
    estimator=GRUBeliefEstimator.from_checkpoint('outputs/data_scaling/modela.pt')
    curves=[];allrows=[];metrics=[]
    for name,config in CONFIGS.items():
        print('Starting',name,config,flush=True)
        env=RewardDebugEnv(estimator,config);check_env(env,warn=True)
        torch.manual_seed(0)
        model=PPO('MlpPolicy',env,seed=0,n_steps=256,batch_size=128,n_epochs=10,
                  learning_rate=3e-4,gamma=.99,gae_lambda=.95,ent_coef=.02,
                  verbose=0,device='cpu',tensorboard_log=str(p/'tensorboard'/name))
        callback=Stage72EpisodeCallback(name,'with_belief',False,50000)
        model.learn(total_timesteps=50000,callback=callback,tb_log_name=name)
        (p/'checkpoints').mkdir(exist_ok=True);model.save(p/'checkpoints'/name)
        rows=evaluate(model,estimator,config,name);allrows.extend(rows);curves.extend(callback.curve_rows)
        write(p/(name+'_evaluation_episodes.csv'),rows)
        metrics.append(dict(experiment=name,timesteps=model.num_timesteps,seed=0,step_commit=5,
                            success_reward=config[0],deception_reward=config[1],success_gated_bonus=config[2],
                            evaluation_episodes=50,optimization_updates=model._n_updates))
        write(p/'training_metrics.csv',metrics);write(p/'evaluation_episodes.csv',allrows);write(p/'learning_curves.csv',curves)
        print('Finished',name,'task success',np.mean([r['deceiver_success'] for r in rows]),flush=True)
        env.close()
    return allrows

if __name__=='__main__':run('/workspace/stage73-reward-debug')
