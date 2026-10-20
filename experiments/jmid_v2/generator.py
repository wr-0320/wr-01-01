"""Independent v2 generator: directed report copying and content censorship."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TRAIN_GROUPS = ((0, 1), (2, 3), (4, 5))
ROTATED_GROUPS = ((1, 2), (3, 4), (5, 0))


@dataclass(frozen=True)
class V2Dataset:
    z: np.ndarray
    y: np.ndarray
    raw_reports: np.ndarray
    copied: np.ndarray
    groups: tuple[tuple[int, int], ...]

    @property
    def selected_signal(self) -> np.ndarray:
        """Compatibility field for the v1 empirical model interface."""
        return self.raw_reports


def generate_v2(
    episodes: int,
    seed: int,
    *,
    censor_strength: float,
    copy_strength: float,
    censor_mode: str = "positive",
    groups: tuple[tuple[int, int], ...] = TRAIN_GROUPS,
) -> V2Dataset:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0.0 <= censor_strength <= 1.0 or not 0.0 <= copy_strength <= 1.0:
        raise ValueError("strengths must be in [0, 1]")
    if censor_mode not in {"positive", "negative", "conflict", "random"}:
        raise ValueError(f"unknown censor_mode: {censor_mode}")
    flattened = [source for pair in groups for source in pair]
    if sorted(flattened) != list(range(6)):
        raise ValueError("groups must partition the six sources")

    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=episodes, dtype=np.int8)
    state_probability = np.where(z == 1, 0.82, 0.18)
    raw = np.empty((episodes, 6), dtype=np.int8)
    copied = np.zeros((episodes, 6), dtype=np.int8)

    for leader, relay in groups:
        leader_signal = rng.binomial(1, state_probability).astype(np.int8)
        relay_signal = rng.binomial(1, state_probability).astype(np.int8)
        leader_report = np.bitwise_xor(
            leader_signal, rng.binomial(1, 0.08, episodes).astype(np.int8)
        )
        relay_independent = np.bitwise_xor(
            relay_signal, rng.binomial(1, 0.08, episodes).astype(np.int8)
        )
        copy = rng.random(episodes) < copy_strength
        raw[:, leader] = leader_report
        raw[:, relay] = np.where(copy, leader_report, relay_independent)
        copied[:, relay] = copy.astype(np.int8)

    observed = np.empty((episodes, 6), dtype=bool)
    for leader, relay in groups:
        a, b = raw[:, leader], raw[:, relay]
        if censor_mode == "positive":
            q_leader = 0.90 - censor_strength * np.where(a == 1, 0.43, 0.03)
            q_relay = 0.88 - censor_strength * np.where(b == 1, 0.38, 0.04)
            q_relay -= censor_strength * np.where(a != b, 0.10, 0.0)
        elif censor_mode == "negative":
            q_leader = 0.90 - censor_strength * np.where(a == 0, 0.43, 0.03)
            q_relay = 0.88 - censor_strength * np.where(b == 0, 0.38, 0.04)
            q_relay -= censor_strength * np.where(a != b, 0.10, 0.0)
        elif censor_mode == "conflict":
            q_leader = np.full(episodes, 0.86)
            q_relay = 0.86 - censor_strength * np.where(a != b, 0.55, 0.0)
        else:
            q_leader = np.full(episodes, 0.78)
            q_relay = np.full(episodes, 0.78)
        observed[:, leader] = rng.random(episodes) < np.clip(q_leader, 0.05, 0.98)
        observed[:, relay] = rng.random(episodes) < np.clip(q_relay, 0.05, 0.98)

    y = np.where(observed, raw, -1).astype(np.int8)
    return V2Dataset(z=z, y=y, raw_reports=raw, copied=copied, groups=groups)


def v2_diagnostics(data: V2Dataset) -> dict[str, float]:
    missing = data.y < 0
    positive = data.raw_reports == 1
    result = {
        "missing_rate": float(missing.mean()),
        "missing_rate_z0": float(missing[data.z == 0].mean()),
        "missing_rate_z1": float(missing[data.z == 1].mean()),
        "missing_given_raw1": float(missing[positive].mean()),
        "missing_given_raw0": float(missing[~positive].mean()),
        "copy_rate": float(data.copied.mean() * 2.0),
    }
    report_corr, missing_corr = [], []
    for leader, relay in data.groups:
        for state in (0, 1):
            rows = data.z == state
            a = data.raw_reports[rows, leader].astype(float)
            b = data.raw_reports[rows, relay].astype(float)
            if a.std() > 0 and b.std() > 0:
                report_corr.append(float(np.corrcoef(a, b)[0, 1]))
            ma = missing[rows, leader].astype(float)
            mb = missing[rows, relay].astype(float)
            if ma.std() > 0 and mb.std() > 0:
                missing_corr.append(float(np.corrcoef(ma, mb)[0, 1]))
    result["within_state_report_correlation"] = float(np.mean(report_corr))
    result["within_state_missing_correlation"] = float(np.mean(missing_corr))
    return result

