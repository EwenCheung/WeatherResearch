"""Shared helpers for Phase 0 walkthrough notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def find_project_root(start: Path | None = None) -> Path:
    """Find the WeatherResearch project root from a notebook or shell cwd."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not find WeatherResearch pyproject.toml")


PROJECT_ROOT = find_project_root()
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
AARDVARK_REFERENCE_ROOT = WORKSPACE_ROOT / "Reference-Repo-aardvark-weather-public-main"
AARDVARK_DATA_DIR = AARDVARK_REFERENCE_ROOT / "data"
AARDVARK_CODE_DIR = AARDVARK_REFERENCE_ROOT / "aardvark"
_AARDVARK_TRAINED_MODELS_PLURAL = AARDVARK_REFERENCE_ROOT / "trained_models"
_AARDVARK_TRAINED_MODEL_SINGULAR = AARDVARK_REFERENCE_ROOT / "trained_model"
AARDVARK_TRAINED_MODELS_DIR = (
    _AARDVARK_TRAINED_MODELS_PLURAL
    if _AARDVARK_TRAINED_MODELS_PLURAL.exists()
    else _AARDVARK_TRAINED_MODEL_SINGULAR
)


def choose_torch_device(prefer_mps: bool = True) -> str:
    """Return the local torch device preference for notebook smoke runs."""
    import torch

    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def require_path(path: Path, purpose: str) -> Path:
    """Raise a clear notebook error when a required local artifact is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")
    return path


def load_torch_pickle_cpu(path: Path) -> Any:
    """Load a pickle containing CUDA torch tensors onto CPU-only/MPS machines."""
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
