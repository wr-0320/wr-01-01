# Deception RL

A research project for deception and counter-deception in partially observable,
two-agent environments.

## First-stage scope

The initial environment is a 7x7 PettingZoo `ParallelEnv`. The Deceiver knows
whether its true objective is A or B. The Defender knows both candidate
locations but not the active objective, and receives only local map visibility
and its observed trajectory. The Defender can guard either objective, observe
with a wider field of view, or track with the widest field of view.

The environment records only positions the Defender has observed. Its
observation does not include the hidden objective identity or the Deceiver's
unseen positions. Belief estimation is intentionally outside the environment;
the planned estimator module will accept this observed trajectory and return
`{"A": probability, "B": probability}` so scripted and learned estimators can
be substituted without changing the environment.

## Planned project areas

- `src/deception_rl/envs/`: environments; the first implementation is
  `gridworld.py`.
- `src/deception_rl/agents/`: planned Deceiver baselines: shortest path
  (honest), random policy, scripted deception, then learned deception. The
  first three baselines and `BeliefBasedDefender` are implemented.
- `src/deception_rl/beliefs/`: independent belief-estimator interface and
  implementations. The current progress-based baseline is replaceable by a
  calibrated or learned GRU/Transformer estimator.
- `src/deception_rl/metrics/`: hypothesis-focused evaluation metrics.
- `src/deception_rl/training/`: future training algorithms, including PPO;
  no PPO implementation is included in this stage.
- `tests/`: environment behavior and information-boundary checks.

## H1/H2 baseline experiment

Run `python -m experiments.run_baseline` to compare the three Deceiver
baselines. The CSV contains one row per episode, a JSON-encoded observed
trajectory, final estimated belief, per-step belief trace, Defender actions,
and Deceiver success rate for each policy. The current belief estimator is an
interpretable progress-score baseline; its outputs are experimental estimates,
not calibrated probabilities.

## H3 commitment experiment

Run `python -m experiments.run_h3_commitment` to compare honest and scripted
deception under an irreversible Defender commitment. `step_commit=2` means the
Defender commits after observing two Deceiver moves and uses that guard choice
for every remaining action. The output records the belief at commitment,
decision correctness, Deceiver success, and the scripted route's excess length
over the shortest path.

## Train and compare intent estimators

Train the GRU from the observed trajectories and `true_goal` labels in the
baseline CSV. By default, ordinary policies train the intent model while
`ScriptedDeception` is held out for deception evaluation:

```bash
python train_intent_model.py --data outputs/baseline_results.csv
```

Training groups identical complete trajectories before validation splitting
and augments each training episode with observed prefixes. The resulting
`outputs/intent_gru.pt` is loaded by default by `BeliefBasedDefender`; when no
checkpoint exists, the rule estimator remains the fallback. To rerun H1/H2 and
H3 side by side with both estimators and write raw plus summary CSVs:

```bash
python -m experiments.compare_estimators
```

The GRU is supervised intent estimation only; no PPO or policy training is
included. To train on all three policy types, pass
`--train-policies honest_shortest_path random_policy scripted_deception`.

To test the frozen Stage 6 checkpoints on scripted duration extrapolation
(`deception_steps=6,7`) and unseen delayed, detour, and multi-switch families:

```bash
python -m experiments.run_deception_family_generalization \
  --model-a outputs/data_scaling/modela.pt \
  --model-b outputs/data_scaling/modelb.pt \
  --episodes 200
```

This evaluation only loads the checkpoints; it does not train or modify them.
The resulting episode and summary CSVs are incorporated by
`experiments.evaluate_paper`, and `experiments.plot_paper` writes
`outputs/figures/robustness_unseen_deception_families.png`. After running the
family experiment, refresh the paper outputs with:

```bash
python -m experiments.evaluate_paper
python -m experiments.plot_paper
```

## Exposure versus strategy diversity

Run `python -m experiments.run_exposure_diversity` to compare C1/C2/C3 over
three paired model-initialization seeds. C1 uses 200 scripted episodes; C2 uses
2,000 scripted episodes from the same family. C2 and C3 both use 2,000
deception episodes and 3,000 total training episodes, while C3 spans four
deception families. Evaluation covers scripted steps 5–7 and shifted delayed,
detour, and multi-switch parameters. The runner emits per-seed metrics and
`outputs/exposure_diversity/robustness_vs_training_diversity.csv`; plot it with
`python -m experiments.plot_exposure_diversity`.

The summary separates all-scenario macro robustness, scripted-duration
extrapolation, and novel-family robustness so the three scripted tests do not
outweigh the three distinct families. Error bars on the diversity figure show
standard deviation across the paired training seeds. To refresh summaries from
existing episode CSVs without retraining, use
`python -m experiments.run_exposure_diversity --summarize-only`.

## Defender adaptation experiment

Run `python -m experiments.run_adaptation` to generate episodes from all three
Deceiver baselines, train Model A on HonestShortestPath + RandomPolicy, and
train Model B on all three policies. Both models are tested against the seen
`deception_steps=2` script and an unseen `deception_steps=3` variant. The output
directory contains the training episodes, both checkpoints, per-episode
results, training metrics, and an aggregate CSV. `belief_accuracy` is computed
over all observed episode states; `commitment_accuracy` and
`deception_success_rate` are episode-level decision and outcome metrics. The
default `step_commit=3` allows the Defender to observe the first reversal of
the seen two-step script while the three-step variant is still feinting.

## Stage 7: Learned Deceiver Sanity

Install the optional PPO dependency and run the intentionally small first
experiment:

```bash
python -m pip install -e '.[rl]'
python -m experiments.run_ppo_deceiver --timesteps 256 --episodes 2 --device cpu
```

PPO optimizes only the Deceiver. The fixed opponent is the Stage 6 Model A
checkpoint (`outputs/data_scaling/modela.pt`) with commitment at step 5. The
actor receives an 8-value encoding of its own position, true-goal one-hot, and
public A/B coordinates; it never receives Defender belief or prediction.
Deception shaping uses only the Defender's publicly emitted wrong-guard action.
Belief shift is read only by the separate evaluation logger. Reward components
are task outcome, public-misdirection bonus, per-step action cost, and
true-goal-distance regression cost.

The run compares HonestShortestPath, RandomPolicy, ScriptedDeception, and PPO
against the same frozen Defender. It writes episode/summary/action-statistics
CSVs, a PPO checkpoint, and `trajectory_diagnostics.png` under
`outputs/ppo_deceiver/`. The 256-step run is a wiring/sanity check, not evidence
of learned deception. Model B/C2/C3 cross-play and larger PPO training are
intentionally deferred.

## Stage 7.2: Belief Ablation and Reward Curriculum

Run `python -m experiments.run_ppo_stage72 --timesteps 50000` only after
reviewing the 128-step smoke outputs in `outputs/ppo_stage72/`. Experiment 1
uses the original 8-value no-belief observation; Experiments 2/3 append Model
A's current belief probabilities, normalized entropy, and the already-public
commitment flag. Beliefs are estimated from Defender-visible trajectories. The
curriculum phases are task-only, task+deception, then task+deception+cost.
TensorBoard logs, per-episode curves, phase timestep records, evaluation CSVs,
checkpoints, `learning_curves.png`, and `belief_ablation.png` are written under
`outputs/ppo_stage72/`. The current run is smoke-only; no 50k training was
started in this stage.

## Stage 7.1: PPO Reward Debug

Run the fixed-Defender reward ablation with 50,000 timesteps per configuration:

```bash
python -m experiments.run_ppo_reward_ablation \
  --timesteps 50000 --evaluation-episodes 20 --device cpu
```

TensorBoard scalars, learning-curve CSV/figure, evaluation summary, action
statistics, and trajectory examples are written under
`outputs/ppo_reward_ablation/`. In the current run, all three task success
rates were 0%. B learned to collect public-misdirection shaping while mostly
staying; C's movement costs also favored staying. Thus the shaping signal is
optimizable but did not produce task-completing deception, and navigation was
not learned in this 50k setup. No self-play was used.

## Expanded Baselines and Generalization

Run `python -m experiments.run_data_scaling` for the larger counter-deception
study. It generates 1,000 episodes each for HonestShortestPath and RandomPolicy,
plus 1,000 episodes for each ScriptedDeception duration from 1 through 4, using
10 seed groups. Model A trains on Honest + Random; Model B also trains on all
four deceptive variants. Both are evaluated for 200 episodes on each duration
1–4 and on unseen duration 5, with commitment at step 5. The runner writes the
training data, checkpoints, training metrics, episode CSV, and generalization
summary under `outputs/data_scaling/`.

## Expanded counter-deception data

Run `python -m experiments.run_data_scaling` to generate 1,000 episodes for
HonestShortestPath, 1,000 for RandomPolicy, and 1,000 for each
ScriptedDeception variant (`deception_steps=1..4`) across ten seed groups. It
re-trains Model A/B and evaluates all seen variants plus the unseen default
`deception_steps=5` using 200 episodes per test condition. Data, checkpoints,
episode-level metrics, and the generalization summary are written to
`outputs/data_scaling/`. Use `--device auto` to train on CUDA when available.

## Install and test

```bash
python -m pip install -e '.[test]'
pytest
```

## Stage 9: Anti-deception dataset and GRU baseline

The two-head supervised baseline predicts A/B intent and honest/deceptive
policy labels from Defender-visible trajectories. It reuses Stage 6 CSVs,
holds out entire trajectory variants for validation, and tests scripted steps
5 plus delayed, detour and multi-switch families. See
[Stage 9 instructions and data limitations](docs/stage9_anti_deception.md).

## Stage 9.2: Identifiability analysis

Compare spatial trajectories, observable timing, and aligned Model A belief
history on a balanced nine-family dataset. The analysis reports conflicting
observable labels and empirical performance ceilings alongside the same GRU
baseline. See [protocol and identifiability limits](docs/stage9_2_identifiability.md).

## Stage 9.3: Active Anti-Deception MVP

Compare a passive fixed-sensing Defender with a rule-based active controller
that chooses observe, track and irreversible commitment. Both use the same
visible-trajectory detector; no PPO or large model is trained. See
[Stage 9.3 protocol and results](docs/stage9_3_active_anti_deception.md).

## Learned Active Defender smoke

A small GRU actor-critic learns observe/track/commit_A/commit_B from frozen goal
belief, deception score, entropy and visible history. Deceiver policies and
GridWorld remain unchanged. See [training and comparison protocol](docs/learned_active_defender.md).

## Stage 14: Paper experiment suite

Stage 14 freezes all earlier algorithms and builds a shared-protocol main
table, cross-stage ablation table, empirical failure catalog, theoretical
appendix, and publication figures. It also provides a common finite-state
environment adapter for the original GridWorld and a separate symmetric
Forked Routes ParallelEnv, plus environment-aware formal Bayesian and
finite-horizon POMDP baselines.

Run the complete evaluation and artifact build with:

```bash
python -m experiments.run_stage14 --seed 0
```

To regenerate only the tables and PNG/PDF figures from the checked-in source
snapshots, run `python -m experiments.build_stage14_paper_suite`. Outputs are
under `outputs/stage14_formal_baselines/` and
`outputs/stage14_paper_suite/`; `artifact_manifest.json` records source hashes
and protocol boundaries.

## Stage 15: LaTeX paper builder

Stage 15 converts the frozen Stage 14 CSV artifacts into a complete LaTeX
skeleton with Abstract, Introduction, Related Work, Method, Theory,
Experiments, and Limitations sections. It generates main-paper and appendix
figures in PNG/PDF, builds LaTeX tables directly from CSV, audits every tagged
claim against an artifact, proof, implementation, or citation, and produces a
rebuttal-style limitation document.

```bash
python -m experiments.build_stage15_paper
```

The command runs `latexmk` when available and writes the compiled paper,
source, figures, tables, claim registry, audit, and manifest under
`outputs/stage15_paper/`. Use `--no-compile` to generate sources without a
local TeX installation.

## Stage 9.5: Risk-sensitive Defender

A 512-episode smoke adds a one-time goal-belief entropy penalty to commitment,
while retaining the learned baseline and unchanged Deceiver families. See
[reward definition, metrics and abstention limitation](docs/stage9_5_risk_sensitive_defender.md).

## Stage 9.6: Uncertainty-gated Defender

A goal-belief entropy action mask prevents commitment above a fixed threshold
without changing rewards. See [512-episode smoke and remaining limitations](docs/stage9_6_uncertainty_gate.md).

## Stage 10: Two-level Anti-Deception Framework

The first method baseline combines a two-head GRU, distance-based Honest
counterfactual prototypes and a reward-free adaptive commitment controller.
See [dataset, 512-episode evaluation and limitations](docs/stage10_two_level_framework.md).

## Stage 10.2: Adaptive Active Defender

Reuse frozen Stage10.1 belief calibration and likelihood ratios with a
cost-aware observation controller that estimates sensor coverage from visible
reachable positions. Compare Rule Active, Stage10.1 and Adaptive on delayed,
detour and multi-switch families with Honest AUROC controls, using exactly
512 environment episodes and no training. See
[protocol, results and regret definition](docs/stage10_2_adaptive_defender.md).

## Stage 10.3: Frozen robustness evaluation

Evaluate the unchanged Adaptive Defender against Rule Active and Stage10.1
on scripted steps 8/10, delay+detour, detour+switch and seeded stochastic
Deceiver strategies. Exactly 512 environment episodes, no training; paired
comparisons verify identical actual Deceiver trajectories. See
[results, latency definition and robustness limitations](docs/stage10_3_robustness.md).

## Stage 11: Adaptive Strategic Deceiver

Evaluate the unchanged Rule Active, Stage10.1 and Adaptive Defenders against a
fixed ScriptedDeception baseline and a lightweight attacker that adapts to
prior sensing, commitment timing and target bias. The experiment runs exactly
512 environment episodes without PPO or changes to GridWorld. Results and
failure traces are under `outputs/stage11_adaptive_attacker/`.

## Stage 12: Bayesian and finite-horizon POMDP baselines

Compare Rule Active, classical Bayesian filtering, finite-horizon belief-state
planning, Stage10.1 and Adaptive Defender against Scripted and online Adaptive
attackers. The fixed 512-episode smoke leaves GridWorld, Deceivers and existing
models unchanged. Results and posterior/action traces are under
`outputs/stage12_bayesian_pomdp/`.

## Stage 13: Learned decision controller

Train a 1,412-parameter decision MLP by offline imitation of frozen Stage10.1
actions plus expected teacher-utility regret. The belief encoder and Deceivers
remain unchanged. A fixed 512-episode smoke compares the teacher and learned
controller on Scripted and online Adaptive attackers. Outputs are under
`outputs/stage13_learned_controller/`.
