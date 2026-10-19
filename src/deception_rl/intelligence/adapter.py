"""Leak-resistant adapter from blue reports to temporal model features."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from deception_rl.envs.deceptive_intelligence import NUM_REGIONS, NUM_SOURCES


REPORT_FEATURES = 4
CELL_FEATURES = REPORT_FEATURES + 1  # report values plus availability mask
INTELLIGENCE_FEATURE_SIZE = NUM_SOURCES * NUM_REGIONS * CELL_FEATURES + NUM_SOURCES + 1

ALLOWED_BLUE_FIELDS = frozenset(
    {
        "reports",
        "report_mask",
        "time_mask",
        "source_reliability",
        "remaining_verification",
        "remaining_tracking",
        "tracked_region",
        "current_step",
        "action_mask",
    }
)
FORBIDDEN_TOKENS = (
    "truth",
    "true_",
    "hypothesis_label",
    "deception_label",
    "red_action",
    "main_region",
    "objective",
)


def audit_blue_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Fail closed if simulator-only fields reach the Defender adapter."""

    fields = set(observation)
    unknown = fields - ALLOWED_BLUE_FIELDS
    suspicious = sorted(
        field
        for field in fields
        if any(token in field.lower() for token in FORBIDDEN_TOKENS)
    )
    if unknown or suspicious:
        raise ValueError(
            f"Unsafe blue observation fields: unknown={sorted(unknown)}, "
            f"truth-like={suspicious}"
        )
    reports = np.asarray(observation["reports"])
    report_mask = np.asarray(observation["report_mask"])
    if reports.ndim != 4 or reports.shape[1:] != (
        NUM_SOURCES,
        NUM_REGIONS,
        REPORT_FEATURES,
    ):
        raise ValueError("reports have an unexpected shape")
    if report_mask.shape != reports.shape[:-1]:
        raise ValueError("report_mask must align with reports")
    return {
        "fields": sorted(fields),
        "report_shape": list(reports.shape),
        "truth_fields_present": False,
    }


def encode_intelligence_sequence(observation: dict[str, Any]) -> torch.Tensor:
    """Encode report history without consulting terminal info or red state.

    Each timestep contains all source-region report values, an explicit
    availability mask per cell, public source reliability, and normalized time.
    A prior token is returned before the first report so recurrent models always
    receive a nonempty sequence.
    """

    audit_blue_observation(observation)
    reports = np.asarray(observation["reports"], dtype=np.float32)
    mask = np.asarray(observation["report_mask"], dtype=np.float32)
    time_mask = np.asarray(observation["time_mask"], dtype=bool)
    source_reliability = np.asarray(
        observation["source_reliability"], dtype=np.float32
    )
    max_cycles = reports.shape[0]
    active_times = np.flatnonzero(time_mask)
    if not len(active_times):
        prior = np.zeros((1, INTELLIGENCE_FEATURE_SIZE), dtype=np.float32)
        prior[0, -NUM_SOURCES - 1 : -1] = source_reliability
        return torch.from_numpy(prior)

    rows = []
    for timestep in active_times:
        cells = np.concatenate(
            (reports[timestep], mask[timestep, ..., None]), axis=-1
        ).reshape(-1)
        normalized_time = np.array(
            [(int(timestep) + 1) / max_cycles], dtype=np.float32
        )
        rows.append(np.concatenate((cells, source_reliability, normalized_time)))
    encoded = np.stack(rows).astype(np.float32)
    if encoded.shape[1] != INTELLIGENCE_FEATURE_SIZE:
        raise AssertionError("intelligence feature schema drifted")
    return torch.from_numpy(encoded)

