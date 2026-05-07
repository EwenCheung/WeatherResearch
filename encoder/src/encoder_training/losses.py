from __future__ import annotations

import torch
from torch import nn


class WeightedRmseLoss(nn.Module):
    """Latitude-weighted RMSE for `(B, lat, lon, variables)` tensors."""

    def __init__(self, latitude_size: int = 121) -> None:
        super().__init__()
        latitudes = torch.linspace(-90.0, 90.0, latitude_size)
        weights = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)
        weights = weights / weights.mean()
        self.register_buffer("weights", weights.reshape(1, latitude_size, 1, 1))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        squared = (prediction - target.to(prediction.device)) ** 2
        weighted = squared * self.weights.to(prediction.device)
        return torch.mean(torch.sqrt(torch.nanmean(weighted, dim=(1, 2, 3))))

