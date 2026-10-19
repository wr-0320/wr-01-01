"""Feature extraction and deployment for the Stage 13 decision-only policy."""

from __future__ import annotations

import numpy as np
import torch

from deception_rl.envs.gridworld import GUARD_A, GUARD_B, OBSERVE, TRACK
from models.stage13_decision import ACTION_NAMES, Stage13DecisionMLP


ENV_ACTIONS = (OBSERVE, TRACK, GUARD_A, GUARD_B)
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}


def decision_features(
    inference: dict[str, object], observation_cost: float, committed: bool
) -> np.ndarray:
    belief = np.asarray(inference["goal_belief"], dtype=np.float32)
    if belief.shape != (2,) or not np.isfinite(belief).all():
        raise ValueError("goal_belief must be a finite length-two distribution")
    if not np.isclose(float(belief.sum()), 1.0, atol=1e-5):
        raise ValueError("goal_belief must sum to one")
    likelihood_ratio = max(float(inference["likelihood_ratio"]), 1e-8)
    counterfactual_score = float(np.tanh(np.log(likelihood_ratio)))
    features = np.asarray(
        [
            belief[0],
            belief[1],
            float(inference["entropy"]),
            counterfactual_score,
            float(observation_cost),
            float(committed),
        ],
        dtype=np.float32,
    )
    if not np.isfinite(features).all() or observation_cost < 0:
        raise ValueError("Invalid Stage 13 decision features")
    return features


class LearnedDecisionController:
    """Frozen inference plus a learned deterministic four-action controller."""

    def __init__(self, inference, model: Stage13DecisionMLP) -> None:
        self.inference = inference
        self.model = model.eval()
        self.reset()

    def reset(self) -> None:
        self.goal = None
        self.commit_timestep = None
        self.observation_cost = 0.0
        self.info = None

    def act(self, observation, timestep):
        inference = self.inference.predict(observation, timestep)
        features = decision_features(
            inference, self.observation_cost, self.goal is not None
        )
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(features).unsqueeze(0))[0]
            probabilities = torch.softmax(logits, dim=0)
            choice = int(torch.argmax(logits))
        self.info = {
            **inference,
            "decision_features": features.tolist(),
            "decision_logits": logits.tolist(),
            "decision_probabilities": probabilities.tolist(),
            "learned_action": ACTION_NAMES[choice],
        }
        # Commitment remains irreversible even if the network later changes its argmax.
        if self.goal is not None:
            return self.goal
        action = ENV_ACTIONS[choice]
        if action == OBSERVE:
            self.observation_cost += 0.01
        elif action == TRACK:
            self.observation_cost += 0.03
        else:
            self.goal = action
            self.commit_timestep = timestep
        return action

