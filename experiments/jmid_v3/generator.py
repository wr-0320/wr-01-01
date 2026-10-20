"""Eight-source extension of the independently implemented v2 generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graphs import Graph, canonical_graph


@dataclass(frozen=True)
class ObservedDataset:
    episode_id: np.ndarray
    z: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class GeneratedDataset:
    observed: ObservedDataset
    raw_reports: np.ndarray
    copied: np.ndarray
    true_graph: Graph


def generate_v3(episodes: int, seed: int, true_graph: Graph) -> GeneratedDataset:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    graph = canonical_graph(true_graph)
    flattened = [node for edge in graph for node in edge]
    if sorted(flattened) != list(range(8)) or len(graph) != 4:
        raise ValueError("true_graph must perfectly match sources 0..7")

    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=episodes, dtype=np.int8)
    state_probability = np.where(z == 1, 0.82, 0.18)
    raw = np.empty((episodes, 8), dtype=np.int8)
    copied = np.zeros((episodes, 8), dtype=np.int8)

    for leader, relay in graph:
        leader_signal = rng.binomial(1, state_probability).astype(np.int8)
        relay_signal = rng.binomial(1, state_probability).astype(np.int8)
        leader_report = np.bitwise_xor(
            leader_signal, rng.binomial(1, 0.08, episodes).astype(np.int8)
        )
        independent_relay = np.bitwise_xor(
            relay_signal, rng.binomial(1, 0.08, episodes).astype(np.int8)
        )
        copy = rng.random(episodes) < 0.85
        raw[:, leader] = leader_report
        raw[:, relay] = np.where(copy, leader_report, independent_relay)
        copied[:, relay] = copy.astype(np.int8)

    observed_mask = np.empty((episodes, 8), dtype=bool)
    for leader, relay in graph:
        a, b = raw[:, leader], raw[:, relay]
        leader_probability = 0.90 - np.where(a == 1, 0.43, 0.03)
        relay_probability = 0.88 - np.where(b == 1, 0.38, 0.04)
        relay_probability -= np.where(a != b, 0.10, 0.0)
        observed_mask[:, leader] = rng.random(episodes) < np.clip(
            leader_probability, 0.05, 0.98
        )
        observed_mask[:, relay] = rng.random(episodes) < np.clip(
            relay_probability, 0.05, 0.98
        )

    y = np.where(observed_mask, raw, -1).astype(np.int8)
    observed = ObservedDataset(
        episode_id=np.arange(episodes, dtype=np.int64), z=z, y=y
    )
    return GeneratedDataset(observed=observed, raw_reports=raw, copied=copied, true_graph=graph)


def diagnostics(data: GeneratedDataset) -> dict[str, float]:
    z, y, raw = data.observed.z, data.observed.y, data.raw_reports
    missing = y < 0
    positive = raw == 1
    report_corr, missing_corr = [], []
    for leader, relay in data.true_graph:
        for state in (0, 1):
            rows = z == state
            a, b = raw[rows, leader].astype(float), raw[rows, relay].astype(float)
            if a.std() > 0 and b.std() > 0:
                report_corr.append(float(np.corrcoef(a, b)[0, 1]))
            ma = missing[rows, leader].astype(float)
            mb = missing[rows, relay].astype(float)
            if ma.std() > 0 and mb.std() > 0:
                missing_corr.append(float(np.corrcoef(ma, mb)[0, 1]))
    return {
        "missing_rate": float(missing.mean()),
        "missing_rate_z0": float(missing[z == 0].mean()),
        "missing_rate_z1": float(missing[z == 1].mean()),
        "missing_given_raw1": float(missing[positive].mean()),
        "missing_given_raw0": float(missing[~positive].mean()),
        "copy_rate": float(data.copied.sum() / (len(z) * 4)),
        "within_state_report_correlation": float(np.mean(report_corr)),
        "within_state_missing_correlation": float(np.mean(missing_corr)),
    }

