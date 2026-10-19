"""Select the trained estimator when its workspace checkpoint is available."""

from __future__ import annotations

from pathlib import Path

from deception_rl.beliefs.trajectory import TrajectoryBeliefEstimator

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[3] / "outputs" / "intent_gru.pt"


def default_belief_estimator():
    if DEFAULT_CHECKPOINT.is_file():
        from deception_rl.beliefs.gru import GRUBeliefEstimator

        return GRUBeliefEstimator.from_checkpoint(DEFAULT_CHECKPOINT)
    return TrajectoryBeliefEstimator()
