"""Frozen common-interface adapters used only by the Stage 17.3 benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from deception_rl.external_envs.interface import DefenderAction, DefenderObservation
from models.stage13_decision import Stage13DecisionMLP


def _entropy(belief: tuple[float, float]) -> float:
    values = np.clip(np.asarray(belief, dtype=float), 1e-12, 1)
    return float(-np.sum(values * np.log2(values)))


def _commit(belief: tuple[float, float]) -> DefenderAction:
    return DefenderAction.COMMIT_B if belief[1] > belief[0] else DefenderAction.COMMIT_A


def _repeat(observation: DefenderObservation) -> DefenderAction | None:
    if observation.committed_goal == 0:
        return DefenderAction.COMMIT_A
    if observation.committed_goal == 1:
        return DefenderAction.COMMIT_B
    return None


class RuleActiveDefender:
    """Stateful fixed rule mirroring the original active sensing principle."""

    name = "rule_active"

    def __init__(self, earliest_commit: int = 4, deadline: int = 9) -> None:
        self.earliest_commit = int(earliest_commit)
        self.deadline = int(deadline)
        self.reset()

    def reset(self) -> None:
        self.history: list[tuple[float, float]] = []
        self.last_info = None

    def act(self, observation: DefenderObservation) -> DefenderAction:
        repeated = _repeat(observation)
        if repeated is not None:
            return repeated
        belief = observation.belief
        self.history.append(belief)
        confidence = max(belief)
        stable = len(self.history) >= 3 and all(
            int(np.argmax(item)) == int(np.argmax(belief)) for item in self.history[-3:]
        )
        reversal = len(self.history) >= 2 and int(np.argmax(self.history[-2])) != int(np.argmax(belief))
        if observation.step >= self.deadline or (
            observation.step >= self.earliest_commit and stable and confidence >= 0.72
        ):
            action = _commit(belief)
        elif reversal or _entropy(belief) > 0.78:
            action = DefenderAction.TRACK
        else:
            action = DefenderAction.OBSERVE
        self.last_info = {
            "belief": list(belief),
            "stable": stable,
            "reversal": reversal,
            "confidence": confidence,
        }
        return action


class RIDeDefenderAdapter:
    """Apply the frozen Stage 13 RIDe decision checkpoint to common features.

    The external adapter never exposes hidden goals.  Its counterfactual input is
    a fixed online likelihood-ratio proxy derived from belief confidence and
    belief reversals, while all learned weights remain byte-for-byte frozen.
    """

    name = "ride"

    def __init__(self, checkpoint: str | Path) -> None:
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        self.model = Stage13DecisionMLP(int(payload["hidden_size"]))
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.reset()

    def reset(self) -> None:
        self.previous_belief: tuple[float, float] | None = None
        self.last_info = None

    def _counterfactual_score(self, belief: tuple[float, float]) -> float:
        confidence = max(belief)
        reversal = 0.0
        if self.previous_belief is not None:
            reversal = float(int(np.argmax(self.previous_belief)) != int(np.argmax(belief)))
        # Fixed bounded log-likelihood-ratio proxy; no fitted parameters.
        return float(np.clip(2.0 * (confidence - 0.5) + 0.75 * reversal, -1, 1))

    def act(self, observation: DefenderObservation) -> DefenderAction:
        repeated = _repeat(observation)
        if repeated is not None:
            return repeated
        counterfactual = self._counterfactual_score(observation.belief)
        features = np.asarray(
            [
                observation.belief[0],
                observation.belief[1],
                _entropy(observation.belief),
                counterfactual,
                observation.cumulative_cost,
                0.0,
            ],
            dtype=np.float32,
        )
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(features).unsqueeze(0))[0]
            probabilities = torch.softmax(logits, dim=0)
        available = torch.tensor(observation.action_mask, dtype=torch.bool)
        masked = logits.masked_fill(~available, -1e9)
        action = DefenderAction(int(torch.argmax(masked)))
        self.previous_belief = observation.belief
        self.last_info = {
            "features": features.tolist(),
            "probabilities": probabilities.tolist(),
            "checkpoint_frozen": True,
        }
        return action

