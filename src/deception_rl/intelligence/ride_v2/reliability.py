"""Interpretable dynamic source reliability filtering from blue-visible history."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from deception_rl.envs.deceptive_intelligence import (
    BLUE_WAIT,
    NUM_HYPOTHESES,
    NUM_REGIONS,
    NUM_SOURCES,
    encode_commit_action,
    encode_verify_action,
)
from deception_rl.intelligence.adapter import audit_blue_observation
from deception_rl.intelligence.ride import HonestIntelligenceReasoner, RIDeIntelligenceController


RELIABILITY_FEATURES = (
    "cross_source_agreement",
    "temporal_consistency",
    "verification_agreement",
    "report_age",
    "counterfactual_residual",
)


def _previous_value(
    activity: np.ndarray,
    mask: np.ndarray,
    timestep: int,
    source: int,
    region: int,
) -> float | None:
    previous = np.flatnonzero(mask[:timestep, source, region])
    if not len(previous):
        return None
    return float(activity[int(previous[-1]), source, region])


def extract_reliability_evidence(
    observation: dict[str, Any],
    reasoner: HonestIntelligenceReasoner,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract five source-level evidence terms from the 49-field encoding inputs."""

    audit_blue_observation(observation)
    reports = np.asarray(observation["reports"], dtype=float)
    mask = np.asarray(observation["report_mask"], dtype=bool)
    active_times = np.flatnonzero(np.asarray(observation["time_mask"], dtype=bool))
    if not len(active_times):
        return (
            np.full((1, NUM_SOURCES, len(RELIABILITY_FEATURES)), 0.5),
            np.array([-1], dtype=int),
        )
    activity = reports[..., 0]
    prototypes = reasoner.prototypes
    activity_indices = np.arange(NUM_SOURCES * NUM_REGIONS) * 5
    honest_activity = prototypes[:, activity_indices].reshape(
        NUM_HYPOTHESES, NUM_SOURCES, NUM_REGIONS
    )
    evidence = np.zeros(
        (len(active_times), NUM_SOURCES, len(RELIABILITY_FEATURES)), dtype=float
    )
    for row_index, timestep_value in enumerate(active_times):
        timestep = int(timestep_value)
        for source in range(NUM_SOURCES):
            valid_regions = np.flatnonzero(mask[timestep, source])
            if not len(valid_regions):
                evidence[row_index, source] = (0.5, 0.5, 0.5, 1.0, 0.5)
                continue

            cross_differences = []
            temporal_differences = []
            verification_differences = []
            for region_value in valid_regions:
                region = int(region_value)
                current = float(activity[timestep, source, region])
                peers = [
                    float(activity[timestep, peer, region])
                    for peer in range(NUM_SOURCES)
                    if peer != source and mask[timestep, peer, region]
                ]
                if peers:
                    cross_differences.append(abs(current - float(np.median(peers))))
                previous = _previous_value(
                    activity, mask, timestep, source, region
                )
                if previous is not None:
                    temporal_differences.append(abs(current - previous))
                if reports[timestep, source, region, 3] > 0.5:
                    reference = (
                        previous
                        if previous is not None
                        else float(np.median(peers)) if peers else current
                    )
                    verification_differences.append(abs(current - reference))

            cross_agreement = 1.0 - float(np.mean(cross_differences)) if cross_differences else 0.5
            temporal_consistency = (
                1.0 - float(np.mean(temporal_differences))
                if temporal_differences
                else 0.5
            )
            verification_agreement = (
                1.0 - float(np.mean(verification_differences))
                if verification_differences
                else 0.5
            )
            age = float(np.mean(reports[timestep, source, valid_regions, 2]))
            current_activity = activity[timestep, source, valid_regions]
            prototype_error = np.mean(
                np.abs(
                    honest_activity[:, source, valid_regions]
                    - current_activity[None, :]
                ),
                axis=1,
            )
            counterfactual_residual = float(np.min(prototype_error))
            evidence[row_index, source] = np.clip(
                (
                    cross_agreement,
                    temporal_consistency,
                    verification_agreement,
                    age,
                    counterfactual_residual,
                ),
                0.0,
                1.0,
            )
    return evidence, active_times.astype(int)


@dataclass(frozen=True)
class ReliabilityEstimate:
    current: np.ndarray
    history: np.ndarray
    time_indices: np.ndarray
    evidence: np.ndarray


class DynamicSourceReliabilityFilter:
    """Two-state Gaussian-emission HMM fitted with offline simulator labels.

    Hidden labels are used only by :meth:`fit`. Runtime :meth:`estimate` accepts
    the ordinary blue observation and frozen honest prototypes only.
    """

    state_names = ("reliable", "compromised")

    def __init__(
        self,
        means: np.ndarray,
        standard_deviations: np.ndarray,
        transitions: np.ndarray,
        priors: np.ndarray,
        *,
        emission_temperature: float = 2.0,
    ) -> None:
        expected_emissions = (NUM_SOURCES, 2, len(RELIABILITY_FEATURES))
        if means.shape != expected_emissions or standard_deviations.shape != expected_emissions:
            raise ValueError("reliability emission parameters have unexpected shapes")
        if transitions.shape != (NUM_SOURCES, 2, 2):
            raise ValueError("reliability transitions have an unexpected shape")
        if priors.shape != (NUM_SOURCES, 2):
            raise ValueError("reliability priors have an unexpected shape")
        self.means = means.astype(float)
        self.standard_deviations = np.maximum(
            standard_deviations.astype(float), 0.08
        )
        self.transitions = transitions.astype(float)
        self.priors = priors.astype(float)
        self.emission_temperature = float(emission_temperature)

    @classmethod
    def fit(
        cls,
        samples: Iterable[dict[str, Any]],
        *,
        emission_temperature: float = 2.0,
    ) -> "DynamicSourceReliabilityFilter":
        rows = list(samples)
        if not rows:
            raise ValueError("reliability samples must be nonempty")
        features = np.stack([np.asarray(row["features"], dtype=float) for row in rows])
        labels = np.stack([np.asarray(row["compromised"], dtype=int) for row in rows])
        if features.shape[1:] != (NUM_SOURCES, len(RELIABILITY_FEATURES)):
            raise ValueError("reliability features have an unexpected shape")
        if labels.shape[1:] != (NUM_SOURCES,):
            raise ValueError("reliability labels have an unexpected shape")
        means = np.zeros((NUM_SOURCES, 2, len(RELIABILITY_FEATURES)))
        deviations = np.zeros_like(means)
        priors = np.zeros((NUM_SOURCES, 2))
        for source in range(NUM_SOURCES):
            for state in range(2):
                selected = features[labels[:, source] == state, source]
                if not len(selected):
                    raise ValueError(f"source {source} has no examples for state {state}")
                means[source, state] = selected.mean(axis=0)
                deviations[source, state] = selected.std(axis=0)
            counts = np.bincount(labels[:, source], minlength=2) + 1
            priors[source] = counts / counts.sum()

        transitions = np.ones((NUM_SOURCES, 2, 2), dtype=float)
        by_episode: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_episode.setdefault(str(row["episode_id"]), []).append(row)
        for episode_rows in by_episode.values():
            ordered = sorted(episode_rows, key=lambda row: int(row["step"]))
            for previous, current in zip(ordered[:-1], ordered[1:]):
                previous_labels = np.asarray(previous["compromised"], dtype=int)
                current_labels = np.asarray(current["compromised"], dtype=int)
                for source in range(NUM_SOURCES):
                    transitions[
                        source, previous_labels[source], current_labels[source]
                    ] += 1
        transitions /= transitions.sum(axis=2, keepdims=True)
        return cls(
            means,
            deviations,
            transitions,
            priors,
            emission_temperature=emission_temperature,
        )

    def _emission_log_likelihood(self, source: int, evidence: np.ndarray) -> np.ndarray:
        difference = (
            evidence[None, :] - self.means[source]
        ) / self.standard_deviations[source]
        log_likelihood = -0.5 * np.sum(difference**2, axis=1)
        log_likelihood -= np.sum(np.log(self.standard_deviations[source]), axis=1)
        return log_likelihood / self.emission_temperature

    def estimate(
        self,
        observation: dict[str, Any],
        reasoner: HonestIntelligenceReasoner,
    ) -> ReliabilityEstimate:
        evidence, time_indices = extract_reliability_evidence(observation, reasoner)
        history = self.estimate_evidence(evidence)
        return ReliabilityEstimate(
            current=history[-1],
            history=history,
            time_indices=time_indices,
            evidence=evidence,
        )

    def estimate_evidence(self, evidence: np.ndarray) -> np.ndarray:
        """Filter a precomputed time × source × feature evidence sequence."""

        evidence = np.asarray(evidence, dtype=float)
        if evidence.ndim != 3 or evidence.shape[1:] != (
            NUM_SOURCES,
            len(RELIABILITY_FEATURES),
        ):
            raise ValueError("reliability evidence has an unexpected shape")
        posterior = self.priors.copy()
        history = np.zeros((len(evidence), NUM_SOURCES), dtype=float)
        for row_index, row in enumerate(evidence):
            if row_index:
                posterior = np.einsum("si,sij->sj", posterior, self.transitions)
            for source in range(NUM_SOURCES):
                log_emission = self._emission_log_likelihood(source, row[source])
                log_posterior = np.log(np.clip(posterior[source], 1e-12, 1)) + log_emission
                log_posterior -= np.max(log_posterior)
                posterior[source] = np.exp(log_posterior)
                posterior[source] /= posterior[source].sum()
                history[row_index, source] = posterior[source, 0]
        return history

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_names": list(self.state_names),
            "feature_names": list(RELIABILITY_FEATURES),
            "means": self.means.tolist(),
            "standard_deviations": self.standard_deviations.tolist(),
            "transitions": self.transitions.tolist(),
            "priors": self.priors.tolist(),
            "emission_temperature": self.emission_temperature,
        }


def condition_observation_on_reliability(
    observation: dict[str, Any], estimate: ReliabilityEstimate
) -> dict[str, Any]:
    """Apply a soft mixture-style attenuation without adding observation fields."""

    conditioned = deepcopy(observation)
    reports = np.asarray(conditioned["reports"], dtype=np.float32).copy()
    report_mask = np.asarray(conditioned["report_mask"], dtype=bool)
    for row, timestep in enumerate(estimate.time_indices):
        if timestep < 0:
            continue
        for source in range(NUM_SOURCES):
            reliability = float(estimate.history[row, source])
            visible = report_mask[timestep, source]
            reports[timestep, source, visible, 0] = (
                reliability * reports[timestep, source, visible, 0]
                + (1.0 - reliability) * 0.10
            )
            reports[timestep, source, visible, 1] *= 0.15 + 0.85 * reliability
    conditioned["reports"] = reports
    conditioned["source_reliability"] = estimate.current.astype(np.float32)
    audit_blue_observation(conditioned)
    return conditioned


class DynamicReliabilityController:
    """Condition RIDe inference on filtered, time-varying source reliability."""

    privileged_diagnostic_only = False

    def __init__(
        self,
        inference: RIDeIntelligenceController,
        reliability_filter: DynamicSourceReliabilityFilter,
        *,
        active_verification: bool = False,
        verification_budget: int = 3,
        commit_step: int = 4,
    ) -> None:
        self.inference = inference
        self.reliability_filter = reliability_filter
        self.active_verification = bool(active_verification)
        self.verification_budget = int(verification_budget)
        self.commit_step = int(commit_step)
        self.name = (
            "dynamic_reliability_entropy_verify"
            if active_verification
            else "dynamic_reliability"
        )
        self.reset()

    def reset(self) -> None:
        self.inference.reset()
        self.info: dict[str, Any] | None = None
        self._verified_cells: list[tuple[int, int]] = []

    def _entropy_cell(
        self, observation: dict[str, Any], belief: np.ndarray
    ) -> tuple[int, int]:
        templates = self.inference.reasoner.prototypes
        activity_indices = np.arange(NUM_SOURCES * NUM_REGIONS) * 5
        activity = templates[:, activity_indices].reshape(
            NUM_HYPOTHESES, NUM_SOURCES, NUM_REGIONS
        )
        mean = np.sum(belief[:, None, None] * activity, axis=0)
        disagreement = np.sum(
            belief[:, None, None] * (activity - mean[None, ...]) ** 2,
            axis=0,
        )
        mask = np.asarray(observation["action_mask"], dtype=bool)
        candidates = [
            (source, region)
            for source in range(NUM_SOURCES)
            for region in range(NUM_REGIONS)
            if mask[encode_verify_action(source, region)]
            and (source, region) not in self._verified_cells
        ]
        if not candidates:
            candidates = [
                (source, region)
                for source in range(NUM_SOURCES)
                for region in range(NUM_REGIONS)
                if mask[encode_verify_action(source, region)]
            ]
        return max(candidates, key=lambda cell: float(disagreement[cell]))

    def act(self, observation: dict[str, Any]) -> int:
        estimate = self.reliability_filter.estimate(
            observation, self.inference.reasoner
        )
        conditioned = condition_observation_on_reliability(observation, estimate)
        self.info = self.inference.infer(conditioned)
        self.info["dynamic_source_reliability"] = estimate.current.tolist()
        self.info["reliability_evidence"] = estimate.evidence[-1].tolist()
        belief = np.asarray(self.info["hypothesis_belief"], dtype=float)
        predicted = int(np.argmax(belief))
        step = int(np.asarray(observation["current_step"])[0])
        if step >= self.commit_step:
            return encode_commit_action(predicted)
        remaining = int(np.asarray(observation["remaining_verification"])[0])
        if (
            self.active_verification
            and len(self._verified_cells) < self.verification_budget
            and remaining > 0
        ):
            cell = self._entropy_cell(observation, belief)
            self._verified_cells.append(cell)
            self.info["selected_verification"] = list(cell)
            return encode_verify_action(*cell)
        return BLUE_WAIT
