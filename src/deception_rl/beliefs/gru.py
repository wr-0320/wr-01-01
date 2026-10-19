"""Adapter that exposes the trained GRU through the belief-estimator protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from models.intent_gru import IntentGRU, trajectory_to_features


class GRUBeliefEstimator:
    """Estimate P(A), P(B) from only the trajectory visible to the Defender."""

    def __init__(
        self,
        model: IntentGRU,
        grid_size: int = 7,
        max_time: int = 30,
        device: str | torch.device = "cpu",
    ) -> None:
        if grid_size < 2 or max_time < 1:
            raise ValueError("grid_size must be >= 2 and max_time must be positive.")
        self.model = model.to(device).eval()
        self.grid_size = grid_size
        self.max_time = max_time
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "GRUBeliefEstimator":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = IntentGRU(
            hidden_size=int(checkpoint["hidden_size"]),
            num_layers=int(checkpoint["num_layers"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(
            model=model,
            grid_size=int(checkpoint["grid_size"]),
            max_time=int(checkpoint["max_time"]),
            device=device,
        )

    def estimate(
        self,
        trajectory: np.ndarray,
        trajectory_mask: np.ndarray,
        target_positions: np.ndarray,
    ) -> dict[str, float]:
        points = np.asarray(trajectory, dtype=np.int64)
        mask = np.asarray(trajectory_mask, dtype=bool)
        targets = np.asarray(target_positions)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("trajectory must have shape (time, 2).")
        if mask.shape != (points.shape[0],):
            raise ValueError("trajectory_mask must have one entry per timestep.")
        if targets.shape != (2, 2):
            raise ValueError("target_positions must have shape (2, 2).")

        visible_times = np.flatnonzero(mask)
        if visible_times.size == 0:
            return {"A": 0.5, "B": 0.5}
        visible_points = np.column_stack((visible_times, points[visible_times]))
        features = trajectory_to_features(
            visible_points,
            grid_size=self.grid_size,
            max_time=self.max_time,
        ).unsqueeze(0).to(self.device)
        lengths = torch.tensor([features.shape[1]], dtype=torch.long)
        with torch.inference_mode():
            logits = self.model(features, lengths)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return {"A": float(probabilities[0]), "B": float(probabilities[1])}
