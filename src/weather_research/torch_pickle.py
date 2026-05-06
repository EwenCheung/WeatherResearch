"""Pickle helpers for Aardvark sample files containing torch tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_torch_pickle_on_cpu(path: Path) -> Any:
    """Load a pickle containing CUDA torch tensors onto CPU first."""
    import io
    import pickle

    import torch

    original_load_from_bytes = torch.storage._load_from_bytes

    def load_from_bytes_on_cpu(data: bytes) -> Any:
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)

    torch.storage._load_from_bytes = load_from_bytes_on_cpu
    try:
        with path.open("rb") as fp:
            return pickle.load(fp)
    finally:
        torch.storage._load_from_bytes = original_load_from_bytes
