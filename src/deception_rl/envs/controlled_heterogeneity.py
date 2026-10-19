"""Controlled manipulation heterogeneity for isolated RCL development."""

from __future__ import annotations

import numpy as np

from deception_rl.envs.deceptive_intelligence import (
    NUM_REGIONS,
    NUM_SOURCES,
    OPERATIONAL_HYPOTHESES,
    DeceptiveIntelligenceEnv,
    decode_red_action,
)


HETEROGENEITY_PROFILES = {
    "low": np.full((NUM_SOURCES, NUM_REGIONS), 0.60, dtype=float),
    "medium": np.asarray(
        [[0.30, 0.30, 0.30], [0.60, 0.60, 0.60], [0.90, 0.90, 0.90]],
        dtype=float,
    ),
    "high": np.asarray(
        [[0.20, 0.60, 1.00], [0.60, 1.00, 0.20], [1.00, 0.20, 0.60]],
        dtype=float,
    ),
    "extreme": np.asarray(
        [[0.02, 0.78, 1.00], [0.78, 1.00, 0.02], [1.00, 0.02, 0.78]],
        dtype=float,
    ),
}


def validate_profiles() -> None:
    expected_shape = (NUM_SOURCES, NUM_REGIONS)
    means = []
    for name, profile in HETEROGENEITY_PROFILES.items():
        if profile.shape != expected_shape:
            raise AssertionError(f"{name} profile has the wrong shape")
        if not np.all((profile >= 0.0) & (profile <= 1.0)):
            raise AssertionError(f"{name} profile is not probabilistic")
        means.append(float(profile.mean()))
    if not np.allclose(means, means[0], atol=1e-12, rtol=0.0):
        raise AssertionError("heterogeneity profiles do not preserve the global mean")


validate_profiles()


class ControlledHeterogeneityEnv(DeceptiveIntelligenceEnv):
    """Apply eligible manipulations with a frozen source-region probability.

    One activation uniform is drawn for every cell at every report cycle before
    coverage and report noise are sampled. Consequently all profiles consume the
    same random-number pattern. Profile changes affect only whether an eligible
    red manipulation changes activity and receives a compromised-cell label.
    """

    def __init__(self, *, heterogeneity: str, **kwargs) -> None:
        if heterogeneity not in HETEROGENEITY_PROFILES:
            raise ValueError(f"unknown heterogeneity profile: {heterogeneity}")
        super().__init__(**kwargs)
        self.heterogeneity = heterogeneity
        self.manipulation_activation = HETEROGENEITY_PROFILES[heterogeneity].copy()
        self._compromised_history = np.zeros(
            (self.max_cycles, NUM_SOURCES, NUM_REGIONS), dtype=np.int8
        )

    def reset(self, seed=None, options=None):
        self._compromised_history.fill(0)
        return super().reset(seed=seed, options=options)

    @property
    def compromised_history(self) -> np.ndarray:
        return self._compromised_history.copy()

    def _generate_reports(self, red_action, verification) -> None:
        row = self._step_count
        if row >= self.max_cycles:
            return
        self._time_mask[row] = 1
        kind, red_region = decode_red_action(red_action)
        hypothesis = OPERATIONAL_HYPOTHESES[self._hypothesis]
        activation_uniform = self.np_random.random((NUM_SOURCES, NUM_REGIONS))

        for source in range(NUM_SOURCES):
            observed_step = max(0, self._step_count - int(self._source_delay[source]))
            for region in range(NUM_REGIONS):
                verified = verification == (source, region)
                tracked = self._tracked_region == region and self._tracking_ttl > 0
                coverage = float(self._source_coverage[source])
                if verified:
                    coverage = 1.0
                elif tracked:
                    coverage = min(1.0, coverage + 0.16)
                if self.np_random.random() > coverage:
                    continue

                targeted = False
                if kind == "concealment" and region == hypothesis.main_region:
                    targeted = True
                elif kind == "decoy" and region == red_region:
                    selected_source = (self._step_count + int(red_region)) % NUM_SOURCES
                    targeted = source == selected_source
                elif kind == "multi_source_deception" and region == red_region:
                    targeted = True
                activated = targeted and (
                    activation_uniform[source, region]
                    < self.manipulation_activation[source, region]
                )
                self._compromised_history[row, source, region] = int(activated)

                activity = self._base_activity(source, region, observed_step)
                susceptibility = float(self._source_deception_susceptibility[source])
                if verified:
                    susceptibility *= 0.30
                elif tracked:
                    susceptibility *= 0.65
                if activated and kind == "concealment":
                    activity -= 0.42 * susceptibility
                elif activated and kind == "decoy":
                    activity += 0.58 * susceptibility
                elif activated and kind == "multi_source_deception":
                    activity += 0.46 * susceptibility

                reliability = float(self._source_reliability[source])
                noise_scale = 0.18 * (1.0 - reliability)
                if verified:
                    noise_scale *= 0.35
                elif tracked:
                    noise_scale *= 0.65
                activity += float(self.np_random.normal(0.0, noise_scale))
                activity = float(np.clip(activity, 0.0, 1.0))

                confidence = 0.40 + 0.45 * reliability
                confidence += 0.10 if verified else 0.04 if tracked else 0.0
                confidence += float(self.np_random.normal(0.0, 0.06))
                confidence = float(np.clip(confidence, 0.05, 0.98))
                age = min(
                    1.0,
                    (int(self._source_delay[source]) + (0 if verified else 1))
                    / self.max_cycles,
                )
                self._reports[row, source, region] = (
                    activity,
                    confidence,
                    age,
                    float(verified),
                )
                self._report_mask[row, source, region] = 1
        if self._tracking_ttl > 0:
            self._tracking_ttl -= 1
            if self._tracking_ttl == 0:
                self._tracked_region = None
