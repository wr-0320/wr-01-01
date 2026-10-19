"""Matched reliability priors for RCL structural granularity ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from deception_rl.envs.deceptive_intelligence import NUM_REGIONS, NUM_SOURCES
from deception_rl.intelligence.adapter import audit_blue_observation
from deception_rl.intelligence.ride_v2.source_region_reliability import (
    SourceRegionReliabilityEstimate,
)


@dataclass(frozen=True)
class GroupedFixedReliability:
    """Training-only Beta-smoothed priors expanded to source-region cells.

    ``granularity`` controls which cells share one compromise probability. The
    expanded 3x3 matrix keeps the downstream likelihood and controller identical
    to source-region RCL, isolating only the prior granularity.
    """

    reliable: np.ndarray
    granularity: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.reliable, dtype=float)
        if matrix.shape != (NUM_SOURCES, NUM_REGIONS):
            raise ValueError("reliable must be a source-region matrix")
        if self.granularity not in {"global", "source", "source_region"}:
            raise ValueError("unsupported reliability granularity")
        if not np.all((matrix > 0.0) & (matrix < 1.0)):
            raise ValueError("reliability values must be strictly between zero and one")
        frozen = matrix.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "reliable", frozen)

    @classmethod
    def fit(
        cls,
        samples: Iterable[dict],
        *,
        granularity: str,
    ) -> "GroupedFixedReliability":
        rows = list(samples)
        if not rows:
            raise ValueError("samples must be nonempty")
        counts = np.zeros((NUM_SOURCES, NUM_REGIONS), dtype=float)
        compromised = np.zeros_like(counts)
        for row in rows:
            visible = np.asarray(row["reported_cells"], dtype=bool)
            labels = np.asarray(row["compromised_cells"], dtype=float)
            if visible.shape != counts.shape or labels.shape != counts.shape:
                raise ValueError("sample cell shape does not match the environment")
            counts += visible
            compromised += visible * labels

        if granularity == "global":
            rate = (compromised.sum() + 1.0) / (counts.sum() + 2.0)
            reliable = np.full_like(counts, 1.0 - rate)
        elif granularity == "source":
            rate = (compromised.sum(axis=1) + 1.0) / (counts.sum(axis=1) + 2.0)
            reliable = np.broadcast_to(1.0 - rate[:, None], counts.shape).copy()
        elif granularity == "source_region":
            reliable = 1.0 - (compromised + 1.0) / (counts + 2.0)
        else:
            raise ValueError("granularity must be global, source, or source_region")
        return cls(reliable, granularity)

    def estimate(self, observation, reasoner=None) -> SourceRegionReliabilityEstimate:
        audit_blue_observation(observation)
        times = np.flatnonzero(np.asarray(observation["time_mask"], dtype=bool))
        if not len(times):
            times = np.array([-1])
        history = np.broadcast_to(
            self.reliable, (len(times), NUM_SOURCES, NUM_REGIONS)
        ).copy()
        return SourceRegionReliabilityEstimate(
            history[-1],
            history,
            history.mean(axis=2),
            times,
            np.zeros((len(times), NUM_SOURCES, NUM_REGIONS, 5)),
        )

