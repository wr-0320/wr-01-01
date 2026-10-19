"""Hierarchical source-region reliability filtering for adversarial reports."""

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
from deception_rl.intelligence.ride_v2.reliability import (
    RELIABILITY_FEATURES,
    DynamicSourceReliabilityFilter,
)


def _previous_cell_value(
    activity: np.ndarray,
    mask: np.ndarray,
    timestep: int,
    source: int,
    region: int,
) -> float | None:
    previous = np.flatnonzero(mask[:timestep, source, region])
    return None if not len(previous) else float(activity[int(previous[-1]), source, region])


def extract_source_region_evidence(
    observation: dict[str, Any],
    reasoner: HonestIntelligenceReasoner,
) -> tuple[np.ndarray, np.ndarray]:
    """Return time × source × region evidence using only blue-visible fields."""

    audit_blue_observation(observation)
    reports = np.asarray(observation["reports"], dtype=float)
    mask = np.asarray(observation["report_mask"], dtype=bool)
    active_times = np.flatnonzero(np.asarray(observation["time_mask"], dtype=bool))
    if not len(active_times):
        return (
            np.full(
                (1, NUM_SOURCES, NUM_REGIONS, len(RELIABILITY_FEATURES)), 0.5
            ),
            np.array([-1], dtype=int),
        )
    activity = reports[..., 0]
    activity_indices = np.arange(NUM_SOURCES * NUM_REGIONS) * 5
    prototypes = reasoner.prototypes[:, activity_indices].reshape(
        NUM_HYPOTHESES, NUM_SOURCES, NUM_REGIONS
    )
    evidence = np.zeros(
        (
            len(active_times),
            NUM_SOURCES,
            NUM_REGIONS,
            len(RELIABILITY_FEATURES),
        ),
        dtype=float,
    )
    for row_index, timestep_value in enumerate(active_times):
        timestep = int(timestep_value)
        for source in range(NUM_SOURCES):
            for region in range(NUM_REGIONS):
                if not mask[timestep, source, region]:
                    evidence[row_index, source, region] = (0.5, 0.5, 0.5, 1.0, 0.5)
                    continue
                current = float(activity[timestep, source, region])
                peers = [
                    float(activity[timestep, peer, region])
                    for peer in range(NUM_SOURCES)
                    if peer != source and mask[timestep, peer, region]
                ]
                previous = _previous_cell_value(
                    activity, mask, timestep, source, region
                )
                cross_agreement = (
                    1.0 - abs(current - float(np.median(peers))) if peers else 0.5
                )
                temporal_consistency = (
                    1.0 - abs(current - previous) if previous is not None else 0.5
                )
                if reports[timestep, source, region, 3] > 0.5:
                    reference = (
                        previous
                        if previous is not None
                        else float(np.median(peers)) if peers else current
                    )
                    verification_agreement = 1.0 - abs(current - reference)
                else:
                    verification_agreement = 0.5
                age = float(reports[timestep, source, region, 2])
                counterfactual_residual = float(
                    np.min(np.abs(prototypes[:, source, region] - current))
                )
                evidence[row_index, source, region] = np.clip(
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
class SourceRegionReliabilityEstimate:
    current: np.ndarray
    history: np.ndarray
    global_history: np.ndarray
    time_indices: np.ndarray
    evidence: np.ndarray


class HierarchicalSourceRegionReliabilityFilter:
    """Cell-level HMM with source-shared emissions and a global source prior."""

    def __init__(
        self,
        global_filter: DynamicSourceReliabilityFilter,
        means: np.ndarray,
        standard_deviations: np.ndarray,
        transitions: np.ndarray,
        priors: np.ndarray,
        *,
        global_coupling: float = 0.25,
        emission_temperature: float = 2.0,
    ) -> None:
        expected = (NUM_SOURCES, 2, len(RELIABILITY_FEATURES))
        if means.shape != expected or standard_deviations.shape != expected:
            raise ValueError("source-region emission parameters have unexpected shapes")
        if transitions.shape != (NUM_SOURCES, 2, 2):
            raise ValueError("source-region transitions have an unexpected shape")
        if priors.shape != (NUM_SOURCES, 2):
            raise ValueError("source-region priors have an unexpected shape")
        if not 0 <= global_coupling <= 1:
            raise ValueError("global_coupling must be in [0, 1]")
        self.global_filter = global_filter
        self.means = means.astype(float)
        self.standard_deviations = np.maximum(
            standard_deviations.astype(float), 0.08
        )
        self.transitions = transitions.astype(float)
        self.priors = priors.astype(float)
        self.global_coupling = float(global_coupling)
        self.emission_temperature = float(emission_temperature)

    @classmethod
    def fit(
        cls,
        samples: Iterable[dict[str, Any]],
        *,
        global_coupling: float = 0.25,
        emission_temperature: float = 2.0,
    ) -> "HierarchicalSourceRegionReliabilityFilter":
        rows = list(samples)
        if not rows:
            raise ValueError("source-region reliability samples must be nonempty")
        global_filter = DynamicSourceReliabilityFilter.fit(
            rows, emission_temperature=emission_temperature
        )
        features = np.stack(
            [np.asarray(row["cell_features"], dtype=float) for row in rows]
        )
        labels = np.stack(
            [np.asarray(row["compromised_cells"], dtype=int) for row in rows]
        )
        expected_features = (
            NUM_SOURCES,
            NUM_REGIONS,
            len(RELIABILITY_FEATURES),
        )
        if features.shape[1:] != expected_features:
            raise ValueError("source-region features have an unexpected shape")
        if labels.shape[1:] != (NUM_SOURCES, NUM_REGIONS):
            raise ValueError("source-region labels have an unexpected shape")
        means = np.zeros((NUM_SOURCES, 2, len(RELIABILITY_FEATURES)))
        deviations = np.zeros_like(means)
        priors = np.zeros((NUM_SOURCES, 2))
        for source in range(NUM_SOURCES):
            source_features = features[:, source].reshape(-1, len(RELIABILITY_FEATURES))
            source_labels = labels[:, source].reshape(-1)
            for state in range(2):
                selected = source_features[source_labels == state]
                if not len(selected):
                    raise ValueError(f"source {source} has no cell examples for state {state}")
                means[source, state] = selected.mean(axis=0)
                deviations[source, state] = selected.std(axis=0)
            counts = np.bincount(source_labels, minlength=2) + 1
            priors[source] = counts / counts.sum()

        transitions = np.ones((NUM_SOURCES, 2, 2), dtype=float)
        by_episode: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_episode.setdefault(str(row["episode_id"]), []).append(row)
        for episode_rows in by_episode.values():
            ordered = sorted(episode_rows, key=lambda row: int(row["step"]))
            for previous, current in zip(ordered[:-1], ordered[1:]):
                previous_labels = np.asarray(previous["compromised_cells"], dtype=int)
                current_labels = np.asarray(current["compromised_cells"], dtype=int)
                for source in range(NUM_SOURCES):
                    for region in range(NUM_REGIONS):
                        transitions[
                            source,
                            previous_labels[source, region],
                            current_labels[source, region],
                        ] += 1
        transitions /= transitions.sum(axis=2, keepdims=True)
        return cls(
            global_filter,
            means,
            deviations,
            transitions,
            priors,
            global_coupling=global_coupling,
            emission_temperature=emission_temperature,
        )

    def _emission_log_likelihood(
        self, source: int, evidence: np.ndarray
    ) -> np.ndarray:
        difference = (evidence[None, :] - self.means[source]) / self.standard_deviations[source]
        likelihood = -0.5 * np.sum(difference**2, axis=1)
        likelihood -= np.sum(np.log(self.standard_deviations[source]), axis=1)
        return likelihood / self.emission_temperature

    def estimate(
        self,
        observation: dict[str, Any],
        reasoner: HonestIntelligenceReasoner,
    ) -> SourceRegionReliabilityEstimate:
        evidence, time_indices = extract_source_region_evidence(observation, reasoner)
        global_estimate = self.global_filter.estimate(observation, reasoner)
        if len(global_estimate.history) != len(evidence):
            raise AssertionError("global and source-region evidence histories are misaligned")
        history = self.estimate_evidence(evidence, global_estimate.history)
        return SourceRegionReliabilityEstimate(
            current=history[-1],
            history=history,
            global_history=global_estimate.history,
            time_indices=time_indices,
            evidence=evidence,
        )

    def estimate_evidence(
        self, evidence: np.ndarray, global_history: np.ndarray
    ) -> np.ndarray:
        evidence = np.asarray(evidence, dtype=float)
        global_history = np.asarray(global_history, dtype=float)
        if evidence.ndim != 4 or evidence.shape[1:] != (
            NUM_SOURCES,
            NUM_REGIONS,
            len(RELIABILITY_FEATURES),
        ):
            raise ValueError("source-region evidence has an unexpected shape")
        if global_history.shape != (len(evidence), NUM_SOURCES):
            raise ValueError("global reliability history has an unexpected shape")
        posterior = np.repeat(self.priors[:, None, :], NUM_REGIONS, axis=1)
        history = np.zeros((len(evidence), NUM_SOURCES, NUM_REGIONS), dtype=float)
        for row_index, row in enumerate(evidence):
            if row_index:
                posterior = np.einsum("sri,sij->srj", posterior, self.transitions)
            global_reliable = global_history[row_index, :, None]
            reliable_prior = (
                (1.0 - self.global_coupling) * posterior[..., 0]
                + self.global_coupling * global_reliable
            )
            posterior[..., 0] = reliable_prior
            posterior[..., 1] = 1.0 - reliable_prior
            for source in range(NUM_SOURCES):
                for region in range(NUM_REGIONS):
                    log_emission = self._emission_log_likelihood(
                        source, row[source, region]
                    )
                    log_posterior = (
                        np.log(np.clip(posterior[source, region], 1e-12, 1))
                        + log_emission
                    )
                    log_posterior -= np.max(log_posterior)
                    posterior[source, region] = np.exp(log_posterior)
                    posterior[source, region] /= posterior[source, region].sum()
                    history[row_index, source, region] = posterior[source, region, 0]
        return history

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(RELIABILITY_FEATURES),
            "means": self.means.tolist(),
            "standard_deviations": self.standard_deviations.tolist(),
            "transitions": self.transitions.tolist(),
            "priors": self.priors.tolist(),
            "global_coupling": self.global_coupling,
            "emission_temperature": self.emission_temperature,
            "global_filter": self.global_filter.to_dict(),
        }


def condition_observation_on_source_region_reliability(
    observation: dict[str, Any], estimate: SourceRegionReliabilityEstimate
) -> dict[str, Any]:
    conditioned = deepcopy(observation)
    reports = np.asarray(conditioned["reports"], dtype=np.float32).copy()
    report_mask = np.asarray(conditioned["report_mask"], dtype=bool)
    for row, timestep in enumerate(estimate.time_indices):
        if timestep < 0:
            continue
        reliability = estimate.history[row]
        visible = report_mask[timestep]
        reports[timestep, ..., 0][visible] = (
            reliability[visible] * reports[timestep, ..., 0][visible]
            + (1.0 - reliability[visible]) * 0.10
        )
        reports[timestep, ..., 1][visible] *= 0.15 + 0.85 * reliability[visible]
    conditioned["reports"] = reports
    conditioned["source_reliability"] = estimate.current.mean(axis=1).astype(np.float32)
    audit_blue_observation(conditioned)
    return conditioned


class SourceRegionReliabilityController:
    """RIDe inference conditioned on hierarchical source-region trust."""

    privileged_diagnostic_only = False

    def __init__(
        self,
        inference: RIDeIntelligenceController,
        reliability_filter: HierarchicalSourceRegionReliabilityFilter,
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
            "source_region_reliability_entropy_verify"
            if active_verification
            else "source_region_reliability"
        )
        self.reset()

    def reset(self) -> None:
        self.inference.reset()
        self.info: dict[str, Any] | None = None
        self._verified_cells: list[tuple[int, int]] = []

    def _entropy_cell(
        self, observation: dict[str, Any], belief: np.ndarray
    ) -> tuple[int, int]:
        activity_indices = np.arange(NUM_SOURCES * NUM_REGIONS) * 5
        templates = self.inference.reasoner.prototypes[:, activity_indices].reshape(
            NUM_HYPOTHESES, NUM_SOURCES, NUM_REGIONS
        )
        mean = np.sum(belief[:, None, None] * templates, axis=0)
        disagreement = np.sum(
            belief[:, None, None] * (templates - mean[None, ...]) ** 2,
            axis=0,
        )
        action_mask = np.asarray(observation["action_mask"], dtype=bool)
        candidates = [
            (source, region)
            for source in range(NUM_SOURCES)
            for region in range(NUM_REGIONS)
            if action_mask[encode_verify_action(source, region)]
            and (source, region) not in self._verified_cells
        ]
        if not candidates:
            candidates = [
                (source, region)
                for source in range(NUM_SOURCES)
                for region in range(NUM_REGIONS)
                if action_mask[encode_verify_action(source, region)]
            ]
        return max(candidates, key=lambda cell: float(disagreement[cell]))

    def act(self, observation: dict[str, Any]) -> int:
        estimate = self.reliability_filter.estimate(
            observation, self.inference.reasoner
        )
        conditioned = condition_observation_on_source_region_reliability(
            observation, estimate
        )
        self.info = self.inference.infer(conditioned)
        self.info["source_region_reliability"] = estimate.current.tolist()
        self.info["global_source_reliability"] = estimate.global_history[-1].tolist()
        belief = np.asarray(self.info["hypothesis_belief"], dtype=float)
        step = int(np.asarray(observation["current_step"])[0])
        if step >= self.commit_step:
            return encode_commit_action(int(np.argmax(belief)))
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
