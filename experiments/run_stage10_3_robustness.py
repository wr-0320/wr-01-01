"""Frozen Stage10.2 robustness smoke: 504 paired +8 supplementary env episodes."""
import argparse
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from models.anti_deception_gru import AntiDeceptionGRU
from deception_rl.beliefs.two_level import DeceptionAwareBelief
from deception_rl.beliefs.decoupled import LikelihoodRatioReasoner,EvidenceCalibrator,CalibratedGoalInference
from deception_rl.agents.active_anti_deception import ActiveAntiDeception,DefenderReward,trajectory_evidence
from deception_rl.agents.decoupled_defender import DecoupledAntiDeception
from deception_rl.agents.adaptive_defender import AdaptiveActiveDefender
from deception_rl.envs.gridworld import GridWorldParallelEnv
from experiments.stage103_deception_families import FAMILIES,policy,scenario_schedule
from experiments.run_stage10_2_adaptive import add_metrics,summarize
from experiments.train_learned_defender import write_csv

ROOT=Path(__file__).resolve().parents[1]


def evaluate_episode(controller,family,goal,seed,episode_id,name,group):
    """Original evaluator scoring, with a separate robustness policy factory and trace."""
    env=GridWorldParallelEnv();obs,_=env.reset(seed=seed,options={'goal':goal})
    agent=policy(family);agent.reset(seed=seed);controller.reset()
    t=0;score5=None;decision_score=None;trace=[];deceiver_trace=[]
    parts=dict(interception_reward=0.,observation_cost=0.,false_commitment_penalty=0.)
    while env.agents:
        previous=controller.goal
        action=controller.act(obs['defender'],t)
        info=getattr(controller,'info',None)
        score=info['deception_probability'] if info else trajectory_evidence(obs['defender'],t).deception_probability
        if t==5:score5=score
        new=previous is None and controller.goal is not None
        if new:decision_score=score
        # Privileged Deceiver diagnostics are evaluation-only, never passed to Defender.
        daction=agent.act(obs['deceiver'])
        deceiver_trace.append(dict(timestep=t,position=np.asarray(obs['deceiver']['position']).tolist(),action=int(daction)))
        trace.append(dict(timestep=t,action=('commit_A','commit_B','observe','track')[action],P_deceptive=score,info=info))
        obs,rewards,_,_,_=env.step({'deceiver':daction,'defender':action})
        intercepted=rewards['defender']>0;escaped=rewards['deceiver']>0
        components=DefenderReward().components(action,intercepted,new,controller.goal,int(goal=='B'))
        for key,value in components.items():parts[key]+=value
        t+=1
    env.close();assert score5 is not None
    row=dict(controller=name,group=group,family=family,episode=episode_id,seed=seed,true_goal=goal,deception_label=1,
             detection_probability_t5=score5,detection_probability_at_decision=score5 if decision_score is None else decision_score,
             committed=int(controller.goal is not None),false_commitment=int(controller.goal is not None and controller.goal!=int(goal=='B')),
             commit_timestep=controller.commit_timestep,defender_task_success=int(intercepted),deceiver_task_success=int(escaped),
             steps=t,**parts,episode_return=parts['interception_reward']-parts['observation_cost']-parts['false_commitment_penalty'],
             trace=trace,deceiver_trace=deceiver_trace)
    row=add_metrics(row)
    row['decision_latency']=controller.commit_timestep
    row['latency_censored']=int(controller.goal is None)
    return row


def metrics(rows):
    result=summarize(rows)
    committed=[r for r in rows if r['committed']]
    result['decision_latency']=float(np.mean([r['decision_latency'] for r in committed])) if committed else None
    result['median_decision_latency']=float(np.median([r['decision_latency'] for r in committed])) if committed else None
    result['uncommitted_episodes']=len(rows)-len(committed)
    result['truncated_episodes']=sum(not r['defender_task_success'] and not r['deceiver_task_success'] for r in rows)
    return result


def paired_differences(rows):
    by=defaultdict(dict)
    for r in rows:by[r['episode']][r['controller']]=r
    differences=[]
    for key,group in by.items():
        a=group['adaptive'];r=group['rule_active']
        assert len({(v['family'],v['true_goal'],v['seed']) for v in group.values()})==1
        # Equal random streams must yield identical actual behavior, not just equal seeds.
        assert all(v['deceiver_trace']==r['deceiver_trace'] for v in group.values())
        differences.append(dict(episode=key,family=a['family'],goal=a['true_goal'],seed=a['seed'],
                                success_delta=a['defender_task_success']-r['defender_task_success'],
                                false_commitment_delta=a['false_commitment']-r['false_commitment'],
                                cost_delta=a['observation_cost']-r['observation_cost'],
                                regret_delta=a['decision_regret']-r['decision_regret'],
                                latency_delta=None if a['decision_latency'] is None or r['decision_latency'] is None else a['decision_latency']-r['decision_latency']))
    return differences


def render_report(p,comparison,family,supplementary):
    text=['# Stage10.3 Robustness Evaluation','',
          '## 冻结范围与协议','',
          'GridWorld、Rule Active、Stage10.1/10.2 controller、dual-head GRU、calibrator、counterfactual prototypes均冻结，无训练/调参/结果选择。Adaptive没有独立learned policy checkpoint；冻结对象是Stage10.2规则代码及它读取的GRU/calibrator checkpoints。',
          '',
          '512 env episodes：168 paired scenarios×3方法=504；额外8个Adaptive stochastic episodes单独报告。scripted8/10、delay+detour、detour+switch各34 scenarios，stochastic32；每family A/B各半。补充8也A/B平衡，不混入主表。',
          '',
          '长scripted直接复用已有ScriptedDeception(8/10)：到假目标后会stay直到feint结束，之后去真目标。delay+detour：true2→false2→lateral false-side2→true。detour+switch：lateral false-side2→false/true交替4→true。stochastic前8步每步false-target概率.5、true-target.3、随机五动作.2，之后true shortest path。所有规则预先固定，新增family不进入推断原型或校准。',
          '',
          '每scenario三个方法共用family/goal/seed，且验证实际Deceiver动作与位置序列完全一致。标签中的family是真实策略，不保证每个随机轨迹实际出现可观察欺骗。',
          '',
          '## 指标','',
          'False commitment：错误承诺/已承诺episodes；另输出全episode分母版本及coverage。观察成本沿用observe=.01、track=.03。regret=1−(interception−observation cost−.5×false commitment)，oracle是知道true goal后立即正确guard的评价参考，非同信息约束下Bayes regret。decision latency是从t=0到首次不可逆commit的步数，仅对已承诺episodes平均；未承诺、coverage和truncation单独列出。',
          '',
          '|方法|拦截成功|误承诺|观察成本|Regret|Decision latency|Coverage|',
          '|---|---:|---:|---:|---:|---:|---:|']
    for row in comparison:
        lat='N/A' if row['decision_latency'] is None else f"{row['decision_latency']:.2f}"
        false='N/A' if row['false_commitment_rate'] is None else f"{row['false_commitment_rate']:.2%}"
        text.append(f"|{row['controller']}|{row['interception_success']:.2%}|{false}|{row['mean_observation_cost']:.4f}|{row['decision_regret']:.4f}|{lat}|{row['commitment_coverage']:.2%}|")
    text+=['','## 分family结果','','|Family|方法|拦截成功|误承诺|成本|Regret|Latency|','|---|---|---:|---:|---:|---:|---:|']
    for row in family:
        lat='N/A' if row['decision_latency'] is None else f"{row['decision_latency']:.2f}"
        false='N/A' if row['false_commitment_rate'] is None else f"{row['false_commitment_rate']:.2%}"
        text.append(f"|{row['family']}|{row['controller']}|{row['interception_success']:.2%}|{false}|{row['mean_observation_cost']:.4f}|{row['decision_regret']:.4f}|{lat}|")
    a=next(r for r in comparison if r['controller']=='adaptive');r=next(r for r in comparison if r['controller']=='rule_active')
    text+=['','## 结论与限制','',
           f"Adaptive相对Rule：拦截{(a['interception_success']-r['interception_success'])*100:+.2f}个百分点；成本{a['mean_observation_cost']-r['mean_observation_cost']:+.4f}；regret{a['decision_regret']-r['decision_regret']:+.4f}。不能仅凭低成本或早承诺认定robustness更优，错误承诺可能更早。",
           '',
           '前四类确定性策略反复seed产生重复轨迹，真正随机的主比较只有32个scenarios。统计推断不能将所有168个scenario当作独立策略样本；总体均值是该固定family配比的描述。stochastic配对结果另存paired_differences.csv，不进行参数筛选。全部新family均deceptive，本阶段不要求AUROC，CSV中的单类别AUROC为空。',
           '',
           f"额外8个Adaptive stochastic episodes：成功{supplementary['interception_success']:.2%}、误承诺{supplementary['false_commitment_rate']:.2%}、成本{supplementary['mean_observation_cost']:.4f}、regret{supplementary['decision_regret']:.4f}。不混入主比较。",
           '',
           '输出comparison.csv、family_comparison.csv、goal_comparison.csv、evaluation_episodes.csv、paired_differences.csv、supplementary_episodes.csv、supplementary_summary.json、config.json和preserved_inputs.json。已停止，未扩大规模。']
    (p/'stage10_3_report.md').write_text('\n'.join(text)+'\n')


def run(destination,seed=0):
    p=Path(destination);p.mkdir(parents=True,exist_ok=True)
    stage10=Path('/workspace/stage10-two-level');stage101=Path('/workspace/stage10-1-decoupled')
    old_manifest=json.loads(Path('/workspace/stage10-2-adaptive/preserved_inputs.json').read_text())['sha256_before']
    assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in old_manifest.items())
    paths=[Path(f) for f in old_manifest]+[ROOT/'src/deception_rl/agents/adaptive_defender.py',ROOT/'experiments/run_stage10_2_adaptive.py',ROOT/'src/deception_rl/beliefs/two_level.py',ROOT/'src/deception_rl/agents/baselines.py']
    before={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    old=torch.load(stage10/'belief_model.pt',weights_only=True,map_location='cpu')
    model=AntiDeceptionGRU(old['hidden_size']);model.load_state_dict(old['model_state_dict']);model.eval()
    calibrator=EvidenceCalibrator();calibrator.load_state_dict(torch.load(stage101/'calibrator.pt',weights_only=True,map_location='cpu')['model_state_dict']);calibrator.eval()
    for module in (model,calibrator):
        for param in module.parameters():param.requires_grad_(False)
    training=[json.loads(line) for line in (stage10/'train.jsonl').read_text().splitlines()]
    inference=CalibratedGoalInference(DeceptionAwareBelief(model),LikelihoodRatioReasoner(training),calibrator)
    controllers={'rule_active':ActiveAntiDeception(),'stage10_1':DecoupledAntiDeception(inference),'adaptive':AdaptiveActiveDefender(inference)}
    scenarios,audits=scenario_schedule(seed);rows=[]
    # Save protocol before any rollout; no training or data-dependent parameter choices.
    config=dict(seed=seed,total_simulator_episodes=512,paired_scenarios=168,paired_episodes=504,supplementary_adaptive_stochastic_episodes=8,
                family_counts={f:sum(s['family']==f for s in scenarios) for f in FAMILIES},goal_balance='A/B equal within every family',
                delay_detour='true2,false2,lateral_false2,true',detour_switch='lateral_false2,false/true alternation4,true',
                stochastic=dict(prefix_steps=8,false_goal_probability=.5,true_goal_probability=.3,uniform_action_probability=.2,tail='honest row-first shortest path'),
                adaptive_frozen=True,no_training=True,no_unseen_inference_prototypes=True,regret='1 - episode_return; same privileged oracle asStage10.2',
                decision_latency='first irreversible commit timestep; conditional on commitment, coverage reported',
                checkpoint_note='Adaptive is a rule controller with frozen dual-head GRU + calibrator dependencies, not a separately trained policy checkpoint')
    (p/'config.json').write_text(json.dumps(config,indent=2));(p/'protocol_scenarios.json').write_text(json.dumps(dict(paired=scenarios,supplementary=audits),indent=2))
    for scenario in scenarios:
        for name,controller in controllers.items():rows.append(evaluate_episode(controller,name=name,**scenario))
    extra=[evaluate_episode(controllers['adaptive'],name='adaptive',**scenario) for scenario in audits]
    assert len(rows)+len(extra)==512
    differences=paired_differences(rows)
    comparison=[dict(controller=name,**metrics([r for r in rows if r['controller']==name])) for name in controllers]
    family=[dict(controller=name,family=f,**metrics([r for r in rows if r['controller']==name and r['family']==f])) for f in FAMILIES for name in controllers]
    goals=[dict(controller=name,goal=g,**metrics([r for r in rows if r['controller']==name and r['true_goal']==g])) for name in controllers for g in ('A','B')]
    for filename,data in [('comparison.csv',comparison),('family_comparison.csv',family),('goal_comparison.csv',goals),('evaluation_episodes.csv',rows),('paired_differences.csv',differences),('supplementary_episodes.csv',extra)]:write_csv(p/filename,data)
    supplementary=metrics(extra);(p/'supplementary_summary.json').write_text(json.dumps(supplementary,indent=2))
    checks={f:hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in before.items()}
    assert all(checks.values())
    (p/'preserved_inputs.json').write_text(json.dumps(dict(sha256_before=before,unchanged=checks,stage102_manifest_matches=True,paired_deceiver_trajectories_identical=True),indent=2))
    render_report(p,comparison,family,supplementary)
    return comparison


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True);parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();print(json.dumps(run(args.output_dir,args.seed),indent=2))
