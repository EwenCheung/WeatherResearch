from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from encoder_training.devices import move_to_device


class EncoderLoss(Protocol):
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class EpochMetrics:
    loss: float


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: EncoderLoss,
    *,
    device: torch.device,
    gradient_clip_norm: float | None = None,
) -> EpochMetrics:
    """Train one epoch on batches with `assimilation` and `y_target` keys."""
    model.train()
    losses: list[float] = []

    for batch in tqdm(loader, unit="batch"):
        batch = move_to_device(batch, device)
        prediction = model(batch["assimilation"], film_index=None)
        loss = loss_fn(prediction, batch["y_target"])
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))

    return EpochMetrics(loss=float(np.nanmean(losses)))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: EncoderLoss,
    *,
    device: torch.device,
) -> EpochMetrics:
    """Evaluate Pipeline A on batches with `assimilation` and `y_target` keys."""
    model.eval()
    losses: list[float] = []

    for batch in tqdm(loader, unit="batch"):
        batch = move_to_device(batch, device)
        prediction = model(batch["assimilation"], film_index=None)
        loss = loss_fn(prediction, batch["y_target"])
        losses.append(float(loss.detach().cpu()))

    return EpochMetrics(loss=float(np.nanmean(losses)))


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
) -> None:
    """Save a minimal restartable training checkpoint."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        checkpoint_path,
    )

