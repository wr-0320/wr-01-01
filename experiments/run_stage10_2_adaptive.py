"""Fixed-budget, frozen Stage10.1 adaptive sensing comparison (512 env episodes)."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from models.anti_deception_gru import AntiDeceptionGRU
from deception_rl.beliefs.two_level import DeceptionAwareBelief
from deception_rl.beliefs.decoupled import LikelihoodRatioReasoner, EvidenceCalibrator, CalibratedGoalInference
from deception_rl.agents.active_anti_deception import ActiveAntiDeception
from deception_rl.agents.decoupled_defender import DecoupledAntiDeception
from deception_rl.agents.adaptive_defender import AdaptiveActiveDefender
from experiments.run_stage10_framework import run_episode, summarize_group
from experiments.train_learned_defender import write_csv

FAMILIES = ('delayed_deception','detour_deception','multi_switch_deception')
ROOT = Path(__file__).resolve().parents[1]


def schedule(seed):
    """168 paired scenarios plus 8 separate adaptive-only audits = 512 rollouts."""
    rows=[]
    for family,count in [('honest',84),*((f,28) for f in FAMILIES)]:
        for i in range(count):
            rows.append(dict(family=family,goal='A' if i%2==0 else 'B',
                             seed=seed+1300000+len(rows),episode_id=len(rows),group='paired'))
    assert len(rows)==168
    audits=[]
    for family in ('honest',*FAMILIES):
        for goal in ('A','B'):
            audits.append(dict(family=family,goal=goal,seed=seed+1400000+len(audits),
                               episode_id=168+len(audits),group='supplementary'))
    return rows,audits


def summarize(rows):
    result=summarize_group(rows)
    result['interception_success']=result['defender_task_success']
    result['decision_regret']=float(np.mean([r['decision_regret'] for r in rows]))
    result['mean_observe_actions']=float(np.mean([r['observe_actions'] for r in rows]))
    result['mean_track_actions']=float(np.mean([r['track_actions'] for r in rows]))
    return result


def add_metrics(row):
    # Privileged immediate-correct-guard oracle gets 1, zero sensing cost, zero false penalty.
    # Evaluation-only reference; hidden goal is never passed to the controller.
    row['oracle_utility']=1.
    row['decision_regret']=row['oracle_utility']-row['episode_return']
    counts=Counter(r['action'] for r in row['trace'])
    row['observe_actions']=counts['observe'];row['track_actions']=counts['track']
    return row


def report(destination, comparison, checks):
    lines=['# Stage 10.2 Adaptive Active Defender','',
           '## 实现与预设','',
           'Stage10.1 GRU、证据校准器、counterfactual 原型及 controller 文件冻结。新 controller 独立复用同一个冻结 inference；没有拟合、训练、调参或新增 unseen family 原型。GridWorld、Deceiver 与 evaluator 不修改。',
           '',
           '用最后可见位置和时间间隔构造下一步可达位置集合，均匀集合占比近似 observe/track 的可见覆盖率。两种感知动作共用固定信息增益系数 .08，乘 goal entropy、deception probability 与正 log likelihood-ratio 的有界项。覆盖相同且 cost 更低时优先 observe；track 仅在额外可见覆盖的预期收益超过成本时使用。',
           '',
           'L=deceptive_likelihood/honest_likelihood 沿用冻结 Stage10.1 distance-kernel compatibility 与 seen scripted1–3 原型，不是真实生成密度。commit utility=P(correct)−.5P(false)。感知 utility=预计未来正确率×(1−下一步到达风险)−cost−.5P(false)。信息增益和可达集均匀假设属于启发式估计，没有性能保证。',
           '',
           '## 预算与评价','',
           '总计512 env episodes：168相同scenario×3方法=504配对评估，加8个Adaptive单独补充episodes（honest/delayed/detour/multi-switch各A/B一次）。补充结果单独保存，不混入主表。每方法主评估A/B各84，Honest84，deceptive84（三个unseen family各28）。加入Honest只为提供AUROC负例；纯deceptive子集AUROC无定义，CSV记空。',
           '',
           'Detection AUROC沿用原evaluator共同t=5时点分数，各方法基于自己获得的可见历史；它评价端到端系统，并非同一输入上的分类器比较。false commitment以已承诺episodes为分母，另保存全episode分母指标。decision regret =1−episode_return；oracle知道true goal、t=0直接正确guard、utility=1。该oracle仅用于评价，不输入controller；regret包含失败、误承诺惩罚及感知成本，不是同信息约束下的Bayes regret。',
           '',
           '|范围|方法|AUROC|拦截成功率|误承诺率|观察成本|Decision regret|',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in comparison:
        auc='N/A' if r['deception_auroc'] is None else f"{r['deception_auroc']:.4f}"
        lines.append(f"|{r['group']}|{r['controller']}|{auc}|{r['interception_success']:.2%}|{r['false_commitment_rate']:.2%}|{r['mean_observation_cost']:.4f}|{r['decision_regret']:.4f}|")
    selected={r['controller']:r for r in comparison if r['group']=='overall'}
    rule=selected['rule_active'];adaptive=selected['adaptive']
    lines+=['','## 结论','',
            f"与Rule Active相比，Adaptive拦截变化{(adaptive['interception_success']-rule['interception_success'])*100:+.2f}个百分点，观察成本变化{adaptive['mean_observation_cost']-rule['mean_observation_cost']:+.4f}，decision regret变化{adaptive['decision_regret']-rule['decision_regret']:+.4f}。这里只报告固定配置结果，没有根据评估结果选择或重跑策略。",
            '',
            'Rule在这三个预设family已有100%拦截，成功率存在天花板；优于Rule应同时检查错误风险、效率与regret，不能只比较单项成本。固定family和地图的重复seed生成重复轨迹，512 rollouts不是512独立策略/轨迹；不能据此宣称真实分布泛化或统计显著优势。',
            '',
            '## 验证与产物','',
            f"保护输入哈希保持一致：{all(checks.values())}。config.json记录完整固定设置与指标定义，evaluation_episodes.csv含utility/coverage/LR轨迹，comparison.csv、family_comparison.csv、action_statistics.csv、supplementary_episodes.csv保存结果。未继续训练。"]
    (destination/'stage10_2_report.md').write_text('\n'.join(lines)+'\n')


def run(destination,seed=0,stage10=Path('/workspace/stage10-two-level'),stage101=Path('/workspace/stage10-1-decoupled')):
    p=Path(destination);p.mkdir(parents=True,exist_ok=True)
    protected=[ROOT/'src/deception_rl/envs/gridworld.py', ROOT/'experiments/deception_families.py',
               ROOT/'src/deception_rl/agents/decoupled_defender.py', ROOT/'src/deception_rl/beliefs/decoupled.py',
               ROOT/'experiments/run_stage10_framework.py',ROOT/'src/deception_rl/agents/active_anti_deception.py',
               stage10/'belief_model.pt',stage10/'train.jsonl',stage101/'calibrator.pt',ROOT/'outputs/data_scaling/modela.pt']
    before={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in protected}
    old=torch.load(stage10/'belief_model.pt',map_location='cpu',weights_only=True)
    model=AntiDeceptionGRU(old['hidden_size']);model.load_state_dict(old['model_state_dict']);model.eval()
    calibrator=EvidenceCalibrator();calibrator.load_state_dict(torch.load(stage101/'calibrator.pt',map_location='cpu',weights_only=True)['model_state_dict']);calibrator.eval()
    for module in (model,calibrator):
        for parameter in module.parameters():parameter.requires_grad_(False)
    records=[json.loads(l) for l in (stage10/'train.jsonl').read_text().splitlines()]
    inference=CalibratedGoalInference(DeceptionAwareBelief(model),LikelihoodRatioReasoner(records),calibrator)
    controllers={'rule_active':ActiveAntiDeception(),'stage10_1':DecoupledAntiDeception(inference),'adaptive':AdaptiveActiveDefender(inference)}
    paired,audits=schedule(seed);rows=[]
    for scenario in paired:
        for name,controller in controllers.items():
            rows.append(add_metrics(run_episode(controller,name=name,**scenario)))
    extras=[add_metrics(run_episode(controllers['adaptive'],name='adaptive',**s)) for s in audits]
    assert len(rows)==504 and len(extras)==8
    comparison=[]
    for name in controllers:
        for group in ('overall','unseen_only','honest_control'):
            subset=[r for r in rows if r['controller']==name and (group=='overall' or (r['family']=='honest')==(group=='honest_control'))]
            comparison.append(dict(controller=name,group=group,**summarize(subset)))
    family=[dict(controller=name,family=f,**summarize([r for r in rows if r['controller']==name and r['family']==f])) for name in controllers for f in ('honest',*FAMILIES)]
    write_csv(p/'comparison.csv',comparison);write_csv(p/'family_comparison.csv',family)
    write_csv(p/'evaluation_episodes.csv',rows);write_csv(p/'supplementary_episodes.csv',extras)
    actions=[]
    for name in controllers:
        counts=Counter(step['action'] for r in rows if r['controller']==name for step in r['trace'])
        for action in ('observe','track','commit_A','commit_B'):
            actions.append(dict(controller=name,action=action,count=counts[action],fraction=counts[action]/sum(counts.values())))
    write_csv(p/'action_statistics.csv',actions)
    checks={str(x):hashlib.sha256(x.read_bytes()).hexdigest()==before[str(x)] for x in protected}
    (p/'preserved_inputs.json').write_text(json.dumps(dict(sha256_before=before,unchanged=checks),indent=2));assert all(checks.values())
    config=dict(seed=seed,total_simulator_episodes=512,paired_scenarios=168,paired_evaluation_episodes=504,supplementary_adaptive_episodes=8,
                unseen_families=FAMILIES,honest_controls_per_controller=84,unseen_per_family_per_controller=28,
                no_training=True,stage101_frozen=True,no_unseen_prototypes=True,
                adaptive=dict(observe_cost=.01,track_cost=.03,false_commitment_penalty=.5,information_gain=.08,sensor_radii=[2,4],reachable_prior='uniform feasible Manhattan envelope',
                              information_demand='H(goal)*(1+P(deceptive)+max(0,log(L))/(1+max(0,log(L))))'),
                decision_regret='1 - (interception_success - observation_cost - .5*false_commitment); privileged immediate correct-guard oracle utility1',
                detection_auroc='t=5 policy-specific visible histories; pure deceptive subsets undefined',
                selection='single predeclared config; no post-evaluation tuning',stage10_source=str(stage10),stage101_source=str(stage101))
    (p/'config.json').write_text(json.dumps(config,indent=2));report(p,comparison,checks)
    return comparison


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True);parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();print(json.dumps(run(args.output_dir,args.seed),indent=2))
