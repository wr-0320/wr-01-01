from experiments.analyze_stage16_failures import (
    attribute_failures,
    group_for,
    trajectory_signature,
)


def test_stage16_group_definition_targets_representation_exposure():
    assert group_for("scripted_3") == "seen"
    assert group_for("scripted_8") == "unseen"
    assert group_for("honest") == "honest_reference"


def test_timed_trajectory_signature_is_exact_and_ordered():
    left = trajectory_signature([[0, 3, 3], [2, 2, 3]])
    assert left == trajectory_signature([[0, 3, 3], [2, 2, 3]])
    assert left != trajectory_signature([[0, 3, 3], [1, 2, 3]])


def test_failure_attribution_preserves_aliasing_and_premature_overlap():
    states = [
        {"group": "seen", "precommit": 1, "counterfactual_score": -0.5, "scenario_id": 10, "timestep": 0, "goal_correct": 1},
        {"group": "seen", "precommit": 1, "counterfactual_score": 0.5, "scenario_id": 11, "timestep": 0, "goal_correct": 1},
        {"group": "unseen", "precommit": 1, "counterfactual_score": -0.2, "scenario_id": 0, "timestep": 3, "goal_correct": 0},
        {"group": "unseen", "precommit": 0, "counterfactual_score": 0.2, "scenario_id": 0, "timestep": 5, "goal_correct": 1},
        {"group": "unseen", "precommit": 1, "counterfactual_score": -0.2, "scenario_id": 1, "timestep": 3, "goal_correct": 0},
    ]
    common = {
        "group": "unseen",
        "failed": 1,
        "decision_trajectory_signature": "same-visible-history",
        "decision_counterfactual_score": -0.2,
        "decision_entropy": 0.5,
        "decision_confidence": 0.9,
        "decision_log_likelihood_ratio": -0.2,
        "false_commitment": 1,
        "committed": 1,
        "decision_timestep": 3,
    }
    episodes = [
        {**common, "scenario_id": 0, "true_goal": "A"},
        {**common, "scenario_id": 1, "true_goal": "B"},
    ]
    failures, _ = attribute_failures(states, episodes)
    first = failures[0]
    assert first["observational_aliasing"] == 1
    assert first["premature_commitment"] == 1
    assert first["wrong_counterfactual_assumption"] == 1
    assert first["primary_attribution"] == "observational_aliasing"
