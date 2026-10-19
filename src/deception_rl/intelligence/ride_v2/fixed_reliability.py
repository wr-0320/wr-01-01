"""Training-only cell priors for controlled reliability ablation."""
from dataclasses import dataclass
import numpy as np
from deception_rl.intelligence.adapter import audit_blue_observation
from deception_rl.intelligence.ride_v2.source_region_reliability import SourceRegionReliabilityEstimate

@dataclass(frozen=True)
class FixedCellReliability:
    reliable: np.ndarray

    @classmethod
    def fit(cls, samples):
        counts = np.zeros((3, 3))
        compromised = np.zeros((3, 3))
        for row in samples:
            visible = np.asarray(row['reported_cells'], dtype=bool)
            counts += visible
            compromised += visible * np.asarray(row['compromised_cells'])
        return cls(1.0 - (compromised + 1.0) / (counts + 2.0))

    def estimate(self, observation, reasoner=None):
        audit_blue_observation(observation)
        times = np.flatnonzero(np.asarray(observation['time_mask'], dtype=bool))
        if not len(times):
            times = np.array([-1])
        history = np.broadcast_to(self.reliable, (len(times), 3, 3)).copy()
        return SourceRegionReliabilityEstimate(history[-1], history, history.mean(2), times, np.zeros((len(times),3,3,5)))
