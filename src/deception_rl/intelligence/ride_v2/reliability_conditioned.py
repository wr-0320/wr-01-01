"""Reliability-conditioned inference that preserves the original observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from deception_rl.envs.deceptive_intelligence import (
    BLUE_WAIT,
    NUM_HYPOTHESES,
    NUM_REGIONS,
    NUM_SOURCES,
    encode_commit_action,
)
from deception_rl.intelligence.adapter import (
    INTELLIGENCE_FEATURE_SIZE,
    audit_blue_observation,
    encode_intelligence_sequence,
)
from deception_rl.intelligence.ride import HonestIntelligenceReasoner, TemperatureCalibrator
from deception_rl.intelligence.ride_v2.calibrated_reliability import (
    CalibratedCellReliabilityFilter,
)


RELIABILITY_CONDITION_SIZE = NUM_SOURCES * NUM_REGIONS * 2
RELIABILITY_CONDITIONED_FEATURE_SIZE = (
    INTELLIGENCE_FEATURE_SIZE + RELIABILITY_CONDITION_SIZE
)


def encode_reliability_conditioned_sequence(
    observation: dict[str, Any],
    predictor: CalibratedCellReliabilityFilter,
    reasoner: HonestIntelligenceReasoner,
) -> torch.Tensor:
    """Append reliability conditions while preserving all original 49 features."""

    audit_blue_observation(observation)
    original = encode_intelligence_sequence(observation)
    estimate = predictor.estimate(observation, reasoner)
    if len(original) != len(estimate.history):
        raise AssertionError("observation and reliability histories are misaligned")
    reliability = torch.as_tensor(
        estimate.history.reshape(len(estimate.history), -1), dtype=torch.float32
    )
    conditioned = torch.cat((original, reliability, 1.0 - reliability), dim=1)
    if conditioned.shape[1] != RELIABILITY_CONDITIONED_FEATURE_SIZE:
        raise AssertionError("reliability-conditioned schema drifted")
    if not torch.equal(conditioned[:, :INTELLIGENCE_FEATURE_SIZE], original):
        raise AssertionError("reliability conditioning modified original features")
    return conditioned


class ReliabilityConditionedModel(nn.Module):
    """GRU trained end-to-end on unchanged reports plus frozen reliability cues."""

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.encoder = nn.GRU(
            RELIABILITY_CONDITIONED_FEATURE_SIZE, hidden_size, batch_first=True
        )
        self.hypothesis_head = nn.Linear(hidden_size, NUM_HYPOTHESES)
        self.deception_head = nn.Linear(hidden_size, 2)

    def forward(
        self, sequences: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequences.ndim != 3 or sequences.shape[-1] != RELIABILITY_CONDITIONED_FEATURE_SIZE:
            raise ValueError("conditioned sequences have an unexpected shape")
        packed = pack_padded_sequence(
            sequences,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)
        representation = hidden[-1]
        return self.hypothesis_head(representation), self.deception_head(representation)


def collate_conditioned_records(
    records: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not records:
        raise ValueError("records must be nonempty")
    sequences = [torch.as_tensor(row["sequence"], dtype=torch.float32) for row in records]
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    hypothesis = torch.tensor(
        [int(row["hypothesis_label"]) for row in records], dtype=torch.long
    )
    deception = torch.tensor(
        [int(row["deception_label"]) for row in records], dtype=torch.long
    )
    return padded, lengths, hypothesis, deception


class ReliabilityConditionedController:
    """Fixed-time controller for the reliability-conditioned neural encoder."""

    privileged_diagnostic_only = False

    def __init__(
        self,
        model: ReliabilityConditionedModel,
        predictor: CalibratedCellReliabilityFilter,
        reasoner: HonestIntelligenceReasoner,
        calibrator: TemperatureCalibrator | None,
        *,
        name: str = "ride_rc_encoder",
        commit_step: int = 4,
    ) -> None:
        self.model = model.eval()
        self.predictor = predictor
        self.reasoner = reasoner
        self.calibrator = calibrator
        self.name = name
        self.commit_step = int(commit_step)
        self.reset()

    def reset(self) -> None:
        self.info: dict[str, Any] | None = None

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        sequence = encode_reliability_conditioned_sequence(
            observation, self.predictor, self.reasoner
        )
        with torch.inference_mode():
            hypothesis_logits, deception_logits = self.model(
                sequence.unsqueeze(0), torch.tensor([len(sequence)])
            )
            if self.calibrator is not None:
                hypothesis_logits, deception_logits = self.calibrator(
                    hypothesis_logits, deception_logits
                )
            hypothesis = torch.softmax(hypothesis_logits, dim=1)[0].cpu().numpy()
            deception = float(torch.softmax(deception_logits, dim=1)[0, 1].cpu())
        entropy = -float(np.sum(hypothesis * np.log(hypothesis + 1e-12)))
        entropy /= float(np.log(NUM_HYPOTHESES))
        return {
            "hypothesis_belief": hypothesis.tolist(),
            "predicted_hypothesis": int(np.argmax(hypothesis)),
            "confidence": float(hypothesis.max()),
            "entropy": entropy,
            "deception_probability": deception,
        }

    def act(self, observation: dict[str, Any]) -> int:
        self.info = self.infer(observation)
        step = int(np.asarray(observation["current_step"])[0])
        if step >= self.commit_step:
            return encode_commit_action(int(self.info["predicted_hypothesis"]))
        return BLUE_WAIT


@dataclass(frozen=True)
class ReliabilityConditionedLikelihood:
    """Mixture likelihood for reliable versus compromised report generation."""

    reliable_mean: np.ndarray
    reliable_scale: np.ndarray
    compromised_mean: np.ndarray
    compromised_scale: np.ndarray
    temperature: float = 1.0

    @classmethod
    def fit(cls, samples: Iterable[dict[str, Any]]) -> "ReliabilityConditionedLikelihood":
        rows = list(samples)
        if not rows:
            raise ValueError("likelihood samples must be nonempty")
        reliable_mean = np.zeros((NUM_HYPOTHESES, NUM_SOURCES, NUM_REGIONS, 2))
        reliable_scale = np.zeros_like(reliable_mean)
        compromised_mean = np.zeros((NUM_SOURCES, NUM_REGIONS, 2))
        compromised_scale = np.zeros_like(compromised_mean)
        all_values = []
        for row in rows:
            values = np.asarray(row["reported_values"], dtype=float)[..., :2]
            visible = np.asarray(row["reported_cells"], dtype=bool)
            all_values.extend(values[visible].tolist())
        global_values = np.asarray(all_values, dtype=float)
        global_mean = global_values.mean(axis=0)
        global_scale = np.maximum(global_values.std(axis=0), 0.08)

        def selected_values(hypothesis: int | None, source: int, region: int, state: int):
            selected = []
            for row in rows:
                if hypothesis is not None and int(row["hypothesis_label"]) != hypothesis:
                    continue
                if not np.asarray(row["reported_cells"], dtype=bool)[source, region]:
                    continue
                if int(np.asarray(row["compromised_cells"])[source, region]) != state:
                    continue
                selected.append(
                    np.asarray(row["reported_values"], dtype=float)[source, region, :2]
                )
            return np.asarray(selected, dtype=float)

        for source in range(NUM_SOURCES):
            for region in range(NUM_REGIONS):
                compromised = selected_values(None, source, region, 1)
                if len(compromised) < 3:
                    compromised_mean[source, region] = global_mean
                    compromised_scale[source, region] = global_scale
                else:
                    compromised_mean[source, region] = compromised.mean(axis=0)
                    compromised_scale[source, region] = np.maximum(
                        compromised.std(axis=0), 0.08
                    )
                for hypothesis in range(NUM_HYPOTHESES):
                    reliable = selected_values(hypothesis, source, region, 0)
                    if len(reliable) < 3:
                        reliable_mean[hypothesis, source, region] = global_mean
                        reliable_scale[hypothesis, source, region] = global_scale
                    else:
                        reliable_mean[hypothesis, source, region] = reliable.mean(axis=0)
                        reliable_scale[hypothesis, source, region] = np.maximum(
                            reliable.std(axis=0), 0.08
                        )
        return cls(
            reliable_mean,
            reliable_scale,
            compromised_mean,
            compromised_scale,
        )

    @staticmethod
    def _log_normal(value: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return -0.5 * np.sum(((value - mean) / scale) ** 2, axis=-1) - np.sum(
            np.log(scale), axis=-1
        )

    def logits(
        self,
        observation: dict[str, Any],
        predictor: CalibratedCellReliabilityFilter,
        reasoner: HonestIntelligenceReasoner,
    ) -> np.ndarray:
        audit_blue_observation(observation)
        estimate = predictor.estimate(observation, reasoner)
        reports = np.asarray(observation["reports"], dtype=float)
        mask = np.asarray(observation["report_mask"], dtype=bool)
        logits = np.full(NUM_HYPOTHESES, -np.log(NUM_HYPOTHESES), dtype=float)
        for row_index, timestep in enumerate(estimate.time_indices):
            if timestep < 0:
                continue
            for source, region in np.argwhere(mask[int(timestep)]):
                source = int(source)
                region = int(region)
                value = reports[int(timestep), source, region, :2]
                reliability = float(
                    np.clip(estimate.history[row_index, source, region], 1e-5, 1 - 1e-5)
                )
                reliable_log = self._log_normal(
                    value,
                    self.reliable_mean[:, source, region],
                    self.reliable_scale[:, source, region],
                )
                compromised_log = float(
                    self._log_normal(
                        value,
                        self.compromised_mean[source, region],
                        self.compromised_scale[source, region],
                    )
                )
                logits += np.logaddexp(
                    np.log(reliability) + reliable_log,
                    np.log(1.0 - reliability) + compromised_log,
                )
        return logits / self.temperature

    def with_temperature(self, temperature: float) -> "ReliabilityConditionedLikelihood":
        return ReliabilityConditionedLikelihood(
            self.reliable_mean,
            self.reliable_scale,
            self.compromised_mean,
            self.compromised_scale,
            float(temperature),
        )


class ReliabilityLikelihoodController:
    privileged_diagnostic_only = False

    def __init__(
        self,
        likelihood: ReliabilityConditionedLikelihood,
        predictor: CalibratedCellReliabilityFilter,
        reasoner: HonestIntelligenceReasoner,
        *,
        name: str = "ride_rcl",
        commit_step: int = 4,
    ) -> None:
        self.likelihood = likelihood
        self.predictor = predictor
        self.reasoner = reasoner
        self.name = name
        self.commit_step = int(commit_step)
        self.reset()

    def reset(self) -> None:
        self.info: dict[str, Any] | None = None

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        logits = self.likelihood.logits(observation, self.predictor, self.reasoner)
        shifted = logits - logits.max()
        hypothesis = np.exp(shifted)
        hypothesis /= hypothesis.sum()
        entropy = -float(np.sum(hypothesis * np.log(hypothesis + 1e-12)))
        entropy /= float(np.log(NUM_HYPOTHESES))
        return {
            "hypothesis_belief": hypothesis.tolist(),
            "predicted_hypothesis": int(np.argmax(hypothesis)),
            "confidence": float(hypothesis.max()),
            "entropy": entropy,
            "deception_probability": 0.0,
        }

    def act(self, observation: dict[str, Any]) -> int:
        self.info = self.infer(observation)
        step = int(np.asarray(observation["current_step"])[0])
        if step >= self.commit_step:
            return encode_commit_action(int(self.info["predicted_hypothesis"]))
        return BLUE_WAIT
