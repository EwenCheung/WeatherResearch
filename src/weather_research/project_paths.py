"""Project and reference-repository path helpers."""

from __future__ import annotations

from pathlib import Path


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
