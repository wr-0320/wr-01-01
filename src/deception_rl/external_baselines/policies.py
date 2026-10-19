"""Stage 17.2 baselines over the common Defender interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from deception_rl.external_envs.interface import DefenderAction, DefenderObservation


def _commit_action(belief: tuple[float, float]) -> DefenderAction:
    return DefenderAction.COMMIT_B if belief[1] > belief[0] else DefenderAction.COMMIT_A


def _repeat_commitment(observation: DefenderObservation) -> DefenderAction | None:
    if observation.committed_goal == 0:
        return DefenderAction.COMMIT_A
    if observation.committed_goal == 1:
        return DefenderAction.COMMIT_B
    return None


def _entropy(probability_b: float) -> float:
    probability_b = float(np.clip(probability_b, 1e-12, 1 - 1e-12))
    return float(
        -probability_b * np.log2(probability_b)
        - (1 - probability_b) * np.log2(1 - probability_b)
    )


class BayesianDefenderAdapter:
    """Posterior stopping rule consuming each environment's analytic belief."""

    name = "bayesian"

    def __init__(self, confidence: float = 0.8, earliest_commit: int = 2, deadline: int = 7):
        if not 0.5 < confidence < 1 or earliest_commit < 0 or deadline < earliest_commit:
            raise ValueError("Invalid Bayesian stopping parameters")
        self.confidence = float(confidence)
        self.earliest_commit = int(earliest_commit)
        self.deadline = int(deadline)
        self.reset()

    def reset(self) -> None:
        self.last_info = None

    def act(self, observation: DefenderObservation) -> DefenderAction:
        repeated = _repeat_commitment(observation)
        if repeated is not None:
            return repeated
        confidence = max(observation.belief)
        should_commit = observation.step >= self.deadline or (
            observation.step >= self.earliest_commit and confidence >= self.confidence
        )
        if should_commit:
            action = _commit_action(observation.belief)
        else:
            action = DefenderAction.TRACK if _entropy(observation.belief[1]) > 0.7 else DefenderAction.OBSERVE
        self.last_info = {
            "belief": list(observation.belief),
            "confidence": confidence,
            "action": action.name,
        }
        return action


@dataclass
class _ActionStats:
    visits: int = 0
    value: float = 0.0


class POMCPBeliefPlanner:
    """Particle Monte Carlo planning on a reduced binary-belief POMDP.

    The online tree is rebuilt at each real action.  Simulations sample a
    hidden goal particle, observation cue, and posterior update.  This is an
    abstract baseline and does not use privileged simulator state.
    """

    name = "pomcp"

    def __init__(
        self,
        simulations: int = 64,
        horizon: int = 5,
        seed: int = 0,
        exploration: float = 1.25,
    ) -> None:
        if simulations < 4 or horizon < 1 or exploration <= 0:
            raise ValueError("Invalid POMCP configuration")
        self.simulations = int(simulations)
        self.horizon = int(horizon)
        self.seed = int(seed)
        self.exploration = float(exploration)
        self.accuracy = {
            DefenderAction.OBSERVE: 0.68,
            DefenderAction.TRACK: 0.84,
        }
        self.cost = {
            DefenderAction.OBSERVE: 0.01,
            DefenderAction.TRACK: 0.03,
        }
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.last_info = None

    @staticmethod
    def _posterior(probability_b: float, cue: int, accuracy: float) -> float:
        likelihood_b = accuracy if cue == 1 else 1 - accuracy
        likelihood_a = 1 - accuracy if cue == 1 else accuracy
        numerator = probability_b * likelihood_b
        return float(numerator / (numerator + (1 - probability_b) * likelihood_a))

    def _simulate(
        self,
        probability_b: float,
        hidden_goal: int,
        depth: int,
        tree: dict[tuple[float, int], dict[DefenderAction, _ActionStats]],
    ) -> float:
        if depth <= 0:
            best = _commit_action((1 - probability_b, probability_b))
            return 1.0 if int(best) - int(DefenderAction.COMMIT_A) == hidden_goal else -0.5
        key = (round(probability_b, 2), depth)
        actions = tuple(DefenderAction)
        node = tree.setdefault(key, {action: _ActionStats() for action in actions})
        unvisited = [action for action in actions if node[action].visits == 0]
        if unvisited:
            action = unvisited[int(self._rng.integers(len(unvisited)))]
        else:
            total = sum(stat.visits for stat in node.values())
            action = max(
                actions,
                key=lambda candidate: node[candidate].value
                + self.exploration
                * np.sqrt(np.log(total + 1) / node[candidate].visits),
            )
        if action in (DefenderAction.COMMIT_A, DefenderAction.COMMIT_B):
            chosen_goal = int(action) - int(DefenderAction.COMMIT_A)
            reward = 1.0 if chosen_goal == hidden_goal else -0.5
        else:
            accuracy = self.accuracy[action]
            cue = hidden_goal if self._rng.random() < accuracy else 1 - hidden_goal
            posterior = self._posterior(probability_b, cue, accuracy)
            reward = -self.cost[action] + self._simulate(
                posterior, hidden_goal, depth - 1, tree
            )
        stats = node[action]
        stats.visits += 1
        stats.value += (reward - stats.value) / stats.visits
        return reward

    def act(self, observation: DefenderObservation) -> DefenderAction:
        repeated = _repeat_commitment(observation)
        if repeated is not None:
            return repeated
        probability_b = observation.belief[1]
        tree: dict[tuple[float, int], dict[DefenderAction, _ActionStats]] = {}
        for _ in range(self.simulations):
            hidden_goal = int(self._rng.random() < probability_b)
            self._simulate(probability_b, hidden_goal, self.horizon, tree)
        root = tree[(round(probability_b, 2), self.horizon)]
        available = [action for action in DefenderAction if observation.action_mask[int(action)]]
        action = max(available, key=lambda candidate: (root[candidate].visits, root[candidate].value))
        self.last_info = {
            "belief": list(observation.belief),
            "root": {
                candidate.name: {
                    "visits": root[candidate].visits,
                    "value": root[candidate].value,
                }
                for candidate in DefenderAction
            },
        }
        return action


class ActiveInferenceDefender:
    """Expected-free-energy controller with fixed observation models."""

    name = "active_inference"

    def __init__(self, earliest_commit: int = 2, deadline: int = 7) -> None:
        self.earliest_commit = int(earliest_commit)
        self.deadline = int(deadline)
        self.accuracy = {
            DefenderAction.OBSERVE: 0.68,
            DefenderAction.TRACK: 0.84,
        }
        self.cost = {
            DefenderAction.OBSERVE: 0.01,
            DefenderAction.TRACK: 0.03,
        }
        self.reset()

    def reset(self) -> None:
        self.last_info = None

    @staticmethod
    def _posterior(probability_b: float, cue: int, accuracy: float) -> float:
        return POMCPBeliefPlanner._posterior(probability_b, cue, accuracy)

    def _expected_entropy(self, probability_b: float, accuracy: float) -> float:
        cue_b_probability = probability_b * accuracy + (1 - probability_b) * (1 - accuracy)
        posterior_b = self._posterior(probability_b, 1, accuracy)
        posterior_a = self._posterior(probability_b, 0, accuracy)
        return cue_b_probability * _entropy(posterior_b) + (1 - cue_b_probability) * _entropy(
            posterior_a
        )

    def act(self, observation: DefenderObservation) -> DefenderAction:
        repeated = _repeat_commitment(observation)
        if repeated is not None:
            return repeated
        probability_b = observation.belief[1]
        values = {
            DefenderAction.COMMIT_A: probability_b * 1.5 - (1 - probability_b),
            DefenderAction.COMMIT_B: (1 - probability_b) * 1.5 - probability_b,
        }
        for action in (DefenderAction.OBSERVE, DefenderAction.TRACK):
            values[action] = self.cost[action] + self._expected_entropy(
                probability_b, self.accuracy[action]
            )
        if observation.step < self.earliest_commit:
            candidates = (DefenderAction.OBSERVE, DefenderAction.TRACK)
        elif observation.step >= self.deadline:
            candidates = (DefenderAction.COMMIT_A, DefenderAction.COMMIT_B)
        else:
            candidates = tuple(DefenderAction)
        action = min(candidates, key=lambda candidate: values[candidate])
        self.last_info = {
            "belief": list(observation.belief),
            "expected_free_energy": {candidate.name: values[candidate] for candidate in values},
        }
        return action


class MAPPOActorCritic(nn.Module):
    """Shared Defender actor with a centralized environment-aware critic."""

    actor_input_size = 10
    context_size = 4

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.actor = nn.Sequential(
            nn.Linear(self.actor_input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, len(DefenderAction)),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.actor_input_size + self.context_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, actor_features: torch.Tensor, context: torch.Tensor):
        return self.actor(actor_features), self.critic(torch.cat((actor_features, context), dim=-1)).squeeze(-1)


def mappo_features(observation: DefenderObservation) -> np.ndarray:
    vector = np.concatenate(
        (np.asarray(observation.observation, dtype=np.float32), np.asarray(observation.belief, dtype=np.float32))
    )
    if vector.shape != (MAPPOActorCritic.actor_input_size,):
        raise ValueError("MAPPO requires the Stage17 eight-value observation plus two-value belief")
    return vector


class MAPPODefenderPolicy:
    """Deployment wrapper for the shared MAPPO smoke actor."""

    name = "mappo"

    def __init__(
        self,
        model: MAPPOActorCritic,
        context_index: int,
        deterministic: bool = True,
        seed: int = 0,
    ) -> None:
        if not 0 <= context_index < MAPPOActorCritic.context_size:
            raise ValueError("context_index must identify one of four environment backends")
        self.model = model
        self.context_index = int(context_index)
        self.deterministic = bool(deterministic)
        self._generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
        self.reset()

    def reset(self) -> None:
        self.last_info = None

    def tensors(self, observation: DefenderObservation):
        device = next(self.model.parameters()).device
        features = torch.from_numpy(mappo_features(observation)).to(device)
        context = torch.zeros(MAPPOActorCritic.context_size, dtype=torch.float32, device=device)
        context[self.context_index] = 1.0
        return features, context

    def distribution(self, observation: DefenderObservation):
        features, context = self.tensors(observation)
        logits, value = self.model(features.unsqueeze(0), context.unsqueeze(0))
        mask = torch.tensor(observation.action_mask, dtype=torch.bool, device=features.device).unsqueeze(0)
        logits = logits.masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits), value[0], features, context

    def act(self, observation: DefenderObservation) -> DefenderAction:
        with torch.inference_mode():
            distribution, value, _, _ = self.distribution(observation)
            if self.deterministic:
                action_index = int(distribution.logits.argmax(dim=1)[0])
            else:
                probabilities = distribution.probs[0]
                action_index = int(torch.multinomial(probabilities, 1, generator=self._generator))
        self.last_info = {
            "probabilities": distribution.probs[0].tolist(),
            "value": float(value),
        }
        return DefenderAction(action_index)
