"""RIDe belief, counterfactual, calibration, and commitment components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from deception_rl.envs.deceptive_intelligence import (
    BLUE_COMMIT_START,
    BLUE_TRACK_START,
    BLUE_VERIFY_START,
    BLUE_WAIT,
    NUM_BLUE_ACTIONS,
    NUM_HYPOTHESES,
    NUM_REGIONS,
    NUM_SOURCES,
    OPERATIONAL_HYPOTHESES,
    encode_commit_action,
    encode_track_action,
    encode_verify_action,
)
from deception_rl.intelligence.adapter import (
    CELL_FEATURES,
    INTELLIGENCE_FEATURE_SIZE,
    encode_intelligence_sequence,
)


class IntelligenceRIDeModel(nn.Module):
    """Temporal encoder with separate operational and deception heads."""

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = int(hidden_size)
        self.encoder = nn.GRU(
            INTELLIGENCE_FEATURE_SIZE, hidden_size, batch_first=True
        )
        self.hypothesis_head = nn.Linear(hidden_size, NUM_HYPOTHESES)
        self.deception_head = nn.Linear(hidden_size, 2)

    def forward(
        self, sequences: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequences.ndim != 3 or sequences.shape[-1] != INTELLIGENCE_FEATURE_SIZE:
            raise ValueError("sequences have an unexpected shape")
        if lengths.ndim != 1 or len(lengths) != len(sequences):
            raise ValueError("lengths must contain one value per sequence")
        packed = pack_padded_sequence(
            sequences,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)
        representation = hidden[-1]
        return (
            self.hypothesis_head(representation),
            self.deception_head(representation),
        )


def collate_intelligence_records(
    records: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not records:
        raise ValueError("records must be nonempty")
    sequences = [torch.as_tensor(row["sequence"], dtype=torch.float32) for row in records]
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    hypotheses = torch.tensor(
        [int(row["hypothesis_label"]) for row in records], dtype=torch.long
    )
    deception = torch.tensor(
        [int(row["deception_label"]) for row in records], dtype=torch.long
    )
    return padded, lengths, hypotheses, deception


class TemperatureCalibrator(nn.Module):
    """Two scalar temperatures fitted on episode-disjoint validation data."""

    def __init__(self) -> None:
        super().__init__()
        self.hypothesis_log_temperature = nn.Parameter(torch.zeros(()))
        self.deception_log_temperature = nn.Parameter(torch.zeros(()))

    def forward(
        self, hypothesis_logits: torch.Tensor, deception_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hypothesis_temperature = self.hypothesis_log_temperature.exp().clamp(0.05, 20)
        deception_temperature = self.deception_log_temperature.exp().clamp(0.05, 20)
        return (
            hypothesis_logits / hypothesis_temperature,
            deception_logits / deception_temperature,
        )

    def fit(
        self,
        hypothesis_logits: torch.Tensor,
        deception_logits: torch.Tensor,
        hypothesis_labels: torch.Tensor,
        deception_labels: torch.Tensor,
        steps: int = 100,
    ) -> dict[str, float]:
        if not len(hypothesis_logits):
            raise ValueError("calibration data must be nonempty")
        optimizer = torch.optim.Adam(self.parameters(), lr=0.04)
        for _ in range(steps):
            optimizer.zero_grad()
            calibrated_h, calibrated_d = self(hypothesis_logits, deception_logits)
            loss = nn.functional.cross_entropy(
                calibrated_h, hypothesis_labels
            ) + nn.functional.cross_entropy(calibrated_d, deception_labels)
            loss.backward()
            optimizer.step()
        return {
            "hypothesis_temperature": float(
                self.hypothesis_log_temperature.detach().exp()
            ),
            "deception_temperature": float(
                self.deception_log_temperature.detach().exp()
            ),
        }


class HonestIntelligenceReasoner:
    """Compare reports with honest counterfactual prototypes for each hypothesis."""

    def __init__(self, records: Iterable[dict[str, Any]], temperature: float = 0.08):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        rows = list(records)
        self.prototypes = np.zeros(
            (NUM_HYPOTHESES, INTELLIGENCE_FEATURE_SIZE), dtype=np.float32
        )
        for hypothesis in range(NUM_HYPOTHESES):
            examples = [
                np.asarray(row["sequence"], dtype=np.float32).mean(axis=0)
                for row in rows
                if int(row["hypothesis_label"]) == hypothesis
                and int(row["deception_label"]) == 0
            ]
            if not examples:
                raise ValueError(
                    f"honest counterfactual examples required for hypothesis {hypothesis}"
                )
            self.prototypes[hypothesis] = np.mean(examples, axis=0)

    def score(
        self, sequence: torch.Tensor, hypothesis_probabilities: np.ndarray
    ) -> dict[str, Any]:
        observed = sequence.detach().cpu().numpy().mean(axis=0)
        # Public source reliability and normalized time are context, not evidence
        # whose mismatch should be interpreted as deception.
        evidence_end = NUM_SOURCES * NUM_REGIONS * CELL_FEATURES
        difference = self.prototypes[:, :evidence_end] - observed[:evidence_end]
        distances = np.mean(difference**2, axis=1)
        honest_likelihood = np.exp(-distances / self.temperature)
        expected_honest_likelihood = float(
            np.dot(hypothesis_probabilities, honest_likelihood)
        )
        counterfactual_score = float(np.clip(1 - expected_honest_likelihood, 0, 1))
        return {
            "honest_likelihood": expected_honest_likelihood,
            "counterfactual_deception_score": counterfactual_score,
            "honest_likelihood_by_hypothesis": honest_likelihood.tolist(),
            "distance_by_hypothesis": distances.tolist(),
        }

    def discriminating_cell(self, first: int, second: int) -> tuple[int, int]:
        activity_indices = np.arange(NUM_SOURCES * NUM_REGIONS) * CELL_FEATURES
        difference = np.abs(
            self.prototypes[first, activity_indices]
            - self.prototypes[second, activity_indices]
        )
        cell = int(np.argmax(difference))
        return divmod(cell, NUM_REGIONS)


@dataclass(frozen=True)
class RIDeVariant:
    name: str
    full_history: bool = True
    use_deception_head: bool = True
    use_counterfactual: bool = True
    use_calibration: bool = True
    adaptive_commitment: bool = True


class RIDeIntelligenceController:
    """Inference and collection policy with independently ablatable components."""

    def __init__(
        self,
        model: IntelligenceRIDeModel,
        reasoner: HonestIntelligenceReasoner,
        calibrator: TemperatureCalibrator | None,
        variant: RIDeVariant,
        *,
        max_cycles: int = 10,
        fixed_commit_step: int = 4,
    ) -> None:
        self.model = model.eval()
        self.reasoner = reasoner
        self.calibrator = calibrator
        self.variant = variant
        self.name = variant.name
        self.max_cycles = int(max_cycles)
        self.fixed_commit_step = int(fixed_commit_step)
        self.reset()

    def reset(self) -> None:
        self.info: dict[str, Any] | None = None

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        sequence = encode_intelligence_sequence(observation)
        model_sequence = sequence if self.variant.full_history else sequence[-1:]
        with torch.inference_mode():
            hypothesis_logits, deception_logits = self.model(
                model_sequence.unsqueeze(0), torch.tensor([len(model_sequence)])
            )
            if self.variant.use_calibration and self.calibrator is not None:
                hypothesis_logits, deception_logits = self.calibrator(
                    hypothesis_logits, deception_logits
                )
            hypothesis = torch.softmax(hypothesis_logits, dim=1)[0].cpu().numpy()
            model_deception = float(
                torch.softmax(deception_logits, dim=1)[0, 1].cpu()
            )
        counterfactual = self.reasoner.score(sequence, hypothesis)
        if self.variant.use_counterfactual:
            honest_compatibility = np.asarray(
                counterfactual["honest_likelihood_by_hypothesis"], dtype=float
            )
            hypothesis = hypothesis * (0.25 + 0.75 * honest_compatibility)
            hypothesis = hypothesis / hypothesis.sum()
            counterfactual = self.reasoner.score(sequence, hypothesis)
        if not self.variant.use_deception_head:
            model_deception = 0.0
        if self.variant.use_counterfactual:
            deception_probability = (
                0.5 * model_deception
                + 0.5 * counterfactual["counterfactual_deception_score"]
            )
        else:
            deception_probability = model_deception
        entropy = -float(np.sum(hypothesis * np.log(hypothesis + 1e-12)))
        entropy /= float(np.log(NUM_HYPOTHESES))
        return {
            "hypothesis_belief": hypothesis.tolist(),
            "predicted_hypothesis": int(np.argmax(hypothesis)),
            "confidence": float(np.max(hypothesis)),
            "entropy": entropy,
            "model_deception_probability": model_deception,
            "deception_probability": float(deception_probability),
            **counterfactual,
        }

    def _information_action(
        self, observation: dict[str, Any], belief: np.ndarray
    ) -> int:
        ranking = np.argsort(belief)[::-1]
        action_mask = np.asarray(observation["action_mask"], dtype=bool)
        if int(np.asarray(observation["remaining_verification"])[0]) > 0:
            source, region = self.reasoner.discriminating_cell(
                int(ranking[0]), int(ranking[1])
            )
            action = encode_verify_action(source, region)
            if action_mask[action]:
                return action
        if int(np.asarray(observation["remaining_tracking"])[0]) > 0:
            region = OPERATIONAL_HYPOTHESES[int(ranking[0])].main_region
            action = encode_track_action(region)
            if action_mask[action]:
                return action
        return BLUE_WAIT

    def act(self, observation: dict[str, Any]) -> int:
        self.info = self.infer(observation)
        belief = np.asarray(self.info["hypothesis_belief"])
        predicted = int(np.argmax(belief))
        step = int(np.asarray(observation["current_step"])[0])
        if not self.variant.adaptive_commitment:
            if step >= self.fixed_commit_step:
                return encode_commit_action(predicted)
            return BLUE_WAIT
        if step >= self.max_cycles - 1:
            return encode_commit_action(predicted)

        confidence = float(belief[predicted])
        entropy = float(self.info["entropy"])
        deception = float(self.info["deception_probability"])
        commit_value = 20.0 * confidence - 10.0 - 2.5 * deception
        information_value = -0.15 + 2.2 * entropy + 1.2 * deception
        if step >= 2 and commit_value >= information_value:
            return encode_commit_action(predicted)
        action = self._information_action(observation, belief)
        self.info["action_utilities"] = {
            "commit": commit_value,
            "information": information_value,
        }
        return action


ABLATION_VARIANTS = (
    RIDeVariant(
        "single_report",
        full_history=False,
        use_deception_head=False,
        use_counterfactual=False,
        use_calibration=False,
        adaptive_commitment=False,
    ),
    RIDeVariant(
        "multi_source_belief",
        use_deception_head=False,
        use_counterfactual=False,
        use_calibration=False,
        adaptive_commitment=False,
    ),
    RIDeVariant(
        "belief_plus_deception",
        use_counterfactual=False,
        use_calibration=False,
        adaptive_commitment=False,
    ),
    RIDeVariant(
        "belief_deception_counterfactual",
        use_calibration=False,
        adaptive_commitment=False,
    ),
    RIDeVariant("full_ride"),
)

