"""Frozen v2 methods, including a fixed M2+M3 composition."""

from __future__ import annotations

import numpy as np

from experiments.jmid_v1.core import (
    DependenceAwareModel,
    JointModel,
    MissingAwareModel,
    NaiveModel,
)


V2_METHODS = ("M2_missing", "M3_dependence", "M23_fixed", "M4_joint")


class FixedComposition:
    """Inclusion--exclusion composition with no fitted combination weights."""

    def __init__(self, m1: NaiveModel, m2: MissingAwareModel, m3: DependenceAwareModel):
        self.m1, self.m2, self.m3 = m1, m2, m3

    @staticmethod
    def _logit(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
        return np.log(clipped) - np.log1p(-clipped)

    def predict_proba(self, y: np.ndarray) -> np.ndarray:
        log_odds = (
            self._logit(self.m2.predict_proba(y))
            + self._logit(self.m3.predict_proba(y))
            - self._logit(self.m1.predict_proba(y))
        )
        return 1.0 / (1.0 + np.exp(-np.clip(log_odds, -40.0, 40.0)))


def fit_v2_models(data, alpha: float = 1.0):
    m1 = NaiveModel(alpha).fit(data)
    m2 = MissingAwareModel(alpha).fit(data)
    m3 = DependenceAwareModel(alpha).fit(data)
    return {
        "M2_missing": m2,
        "M3_dependence": m3,
        "M23_fixed": FixedComposition(m1, m2, m3),
        "M4_joint": JointModel(alpha).fit(data),
    }

