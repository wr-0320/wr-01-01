"""GRU model that predicts hidden intent from Defender-visible trajectories."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

INPUT_SIZE = 6


def trajectory_to_features(
    trajectory: Sequence[Sequence[int]] | torch.Tensor,
    grid_size: int = 7,
    max_time: int = 30,
) -> torch.Tensor:
    """Encode rows of ``(time, row, column)`` as normalized motion features."""
    points = torch.as_tensor(trajectory, dtype=torch.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("trajectory must have shape (observations, 3).")
    if grid_size < 2 or max_time < 1:
        raise ValueError("grid_size must be >= 2 and max_time must be positive.")
    if points.shape[0] == 0:
        return torch.empty((0, INPUT_SIZE), dtype=torch.float32)
    if torch.any(points[1:, 0] < points[:-1, 0]):
        raise ValueError("trajectory times must be non-decreasing.")

    times = points[:, 0:1].clamp(min=0, max=max_time) / max_time
    positions = points[:, 1:3] / (grid_size - 1)
    deltas = torch.zeros_like(positions)
    elapsed = torch.zeros_like(times)
    if points.shape[0] > 1:
        deltas[1:] = positions[1:] - positions[:-1]
        elapsed[1:] = times[1:] - times[:-1]
    return torch.cat((positions, times, deltas, elapsed), dim=1)


class IntentGRU(nn.Module):
    """Sequence classifier whose logits correspond to goals A and B."""

    input_size = INPUT_SIZE

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if sequences.ndim != 3 or sequences.shape[2] != self.input_size:
            raise ValueError("sequences must have shape (batch, time, input_size).")
        if lengths.ndim != 1 or lengths.shape[0] != sequences.shape[0]:
            raise ValueError("lengths must contain one value per sequence.")
        if torch.any(lengths < 1) or torch.any(lengths > sequences.shape[1]):
            raise ValueError("sequence lengths must be between 1 and padded length.")

        packed = pack_padded_sequence(
            sequences,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.classifier(hidden[-1])
