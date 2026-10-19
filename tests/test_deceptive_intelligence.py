import csv

import numpy as np
from pettingzoo.test import parallel_api_test

from deception_rl.agents.intelligence_policies import (
    AdaptiveDeceptionRedPolicy,
    FixedTimeBluePolicy,
    HonestRedPolicy,
    IntelligenceBeliefEstimator,
)
from deception_rl.envs.deceptive_intelligence import (
    BLUE_WAIT,
    NUM_HYPOTHESES,
    OPERATIONAL_HYPOTHESES,
    RED_HONEST,
    DeceptiveIntelligenceEnv,
    decode_red_action,
    encode_commit_action,
    encode_decoy_action,
    encode_verify_action,
)
from experiments.run_deceptive_intelligence import run_experiment


def test_operational_hypotheses_are_joint_intent_and_deception_structures():
    assert NUM_HYPOTHESES > 3
    assert len({hypothesis.main_region for hypothesis in OPERATIONAL_HYPOTHESES}) == 3
    assert len({hypothesis.objective for hypothesis in OPERATIONAL_HYPOTHESES}) == 3
    assert any(hypothesis.feint_region is not None for hypothesis in OPERATIONAL_HYPOTHESES)
    assert len({hypothesis.main_region for hypothesis in OPERATIONAL_HYPOTHESES}) < len(
        OPERATIONAL_HYPOTHESES
    )


def test_blue_information_boundary_excludes_truth_and_deception_labels():
    env = DeceptiveIntelligenceEnv()
    observations, infos = env.reset(seed=7, options={"hypothesis": 3})
    blue = observations["blue"]
    forbidden = {
        "hypothesis",
        "true_hypothesis",
        "main_region",
        "objective",
        "deception_action",
        "is_deception",
        "truth",
    }
    assert forbidden.isdisjoint(blue)
    assert infos["blue"] == {}
    assert env.observation_space("blue").contains(blue)
    assert env.observation_space("red").contains(observations["red"])


def test_seeded_reports_are_reproducible_and_reports_do_not_contain_truth_flag():
    def first_report():
        env = DeceptiveIntelligenceEnv()
        env.reset(seed=19, options={"hypothesis": 4})
        observations, *_ = env.step(
            {"red": encode_decoy_action(1, multisource=True), "blue": BLUE_WAIT}
        )
        return observations["blue"]

    first = first_report()
    second = first_report()
    np.testing.assert_array_equal(first["reports"], second["reports"])
    np.testing.assert_array_equal(first["report_mask"], second["report_mask"])
    assert first["reports"].shape[-1] == 4


def test_independent_verification_reduces_decoy_effect_and_spends_budget():
    def observed_activity(blue_action):
        env = DeceptiveIntelligenceEnv()
        env.reset(seed=2, options={"hypothesis": 3})
        observations, *_ = env.step(
            {
                "red": encode_decoy_action(0, multisource=True),
                "blue": blue_action,
            }
        )
        return observations["blue"]

    passive = observed_activity(BLUE_WAIT)
    verified = observed_activity(encode_verify_action(0, 0))
    assert verified["reports"][0, 0, 0, 0] < passive["reports"][0, 0, 0, 0]
    assert verified["reports"][0, 0, 0, 3] == 1
    assert verified["remaining_verification"].tolist() == [2]


def test_deception_budget_prevents_unlimited_report_manipulation():
    env = DeceptiveIntelligenceEnv(deception_budget=2)
    observations, _ = env.reset(seed=0, options={"hypothesis": 3})
    action = encode_decoy_action(0, multisource=True)
    observations, *_ = env.step({"red": action, "blue": BLUE_WAIT})
    assert observations["red"]["remaining_deception"].tolist() == [0]
    assert observations["red"]["action_mask"][action] == 0
    observations, rewards, *_ = env.step({"red": action, "blue": BLUE_WAIT})
    assert rewards["red"] == 0


def test_truth_is_revealed_only_after_terminal_commitment():
    env = DeceptiveIntelligenceEnv()
    env.reset(seed=1, options={"hypothesis": 4})
    _, _, _, _, ongoing_infos = env.step(
        {"red": RED_HONEST, "blue": BLUE_WAIT}
    )
    assert ongoing_infos["blue"] == {}
    _, rewards, terminations, _, terminal_infos = env.step(
        {"red": RED_HONEST, "blue": encode_commit_action(4)}
    )
    metrics = terminal_infos["blue"]["episode_metrics"]
    assert all(terminations.values())
    assert metrics["true_hypothesis"] == 4
    assert metrics["decision_correct"] == 1
    assert rewards["blue"] > 0


def test_adaptive_rule_based_deception_moves_away_from_verified_decoy():
    env = DeceptiveIntelligenceEnv()
    policy = AdaptiveDeceptionRedPolicy()
    policy.reset()
    observations, _ = env.reset(seed=0, options={"hypothesis": 3})
    first_action = policy.act(observations["red"])
    first_region = decode_red_action(first_action)[1]
    observations, *_ = env.step(
        {"red": first_action, "blue": encode_verify_action(0, int(first_region))}
    )
    next_action = policy.act(observations["red"])
    next_region = decode_red_action(next_action)[1]
    assert next_region != first_region
    assert next_region != observations["red"]["main_region"]


def test_interpretable_belief_estimator_returns_normalized_latent_belief():
    env = DeceptiveIntelligenceEnv()
    observations, _ = env.reset(seed=3, options={"hypothesis": 0})
    for _ in range(3):
        observations, *_ = env.step({"red": RED_HONEST, "blue": BLUE_WAIT})
    belief = IntelligenceBeliefEstimator().estimate(observations["blue"])
    assert belief.hypothesis_probabilities.shape == (NUM_HYPOTHESES,)
    assert np.isclose(belief.hypothesis_probabilities.sum(), 1)
    assert 0 <= belief.deception_probability <= 1
    assert 0 <= belief.entropy <= 1


def test_experiment_writes_episode_and_aggregate_outputs(tmp_path):
    episodes, summary = run_experiment(
        tmp_path,
        episodes_per_hypothesis=1,
        red_policies=[HonestRedPolicy()],
        blue_policies=[FixedTimeBluePolicy(commit_step=2)],
    )
    with episodes.open(newline="") as handle:
        episode_rows = list(csv.DictReader(handle))
    with summary.open(newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(episode_rows) == NUM_HYPOTHESES
    assert len(summary_rows) == 1
    assert {row["true_hypothesis"] for row in episode_rows} == {
        str(index) for index in range(NUM_HYPOTHESES)
    }


def test_parallel_api_contract():
    parallel_api_test(DeceptiveIntelligenceEnv(), num_cycles=50)

