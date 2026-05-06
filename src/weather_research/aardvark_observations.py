"""Load and validate Aardvark model-ready observation samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from weather_research.project_paths import AARDVARK_DATA_DIR
from weather_research.torch_pickle import load_torch_pickle_on_cpu
from weather_research.weather_state_schema import AardvarkObservationSample

DEFAULT_AARDVARK_SAMPLE_PATH = AARDVARK_DATA_DIR / "sample_data_final.pkl"

REQUIRED_TOP_LEVEL_KEYS = ("assimilation", "forecast", "downscaling", "y_target")
REQUIRED_ASSIMILATION_KEYS = (
    "hirs_current",
    "amsua_current",
    "amsub_current",
    "iasi_current",
    "ascat_current",
    "x_context_hadisd_current",
    "y_context_hadisd_current",
    "icoads_current",
    "igra_current",
)


def validate_aardvark_observation_payload(payload: dict[str, Any]) -> None:
    """Validate the minimum Aardvark sample structure needed for IC generation."""
    missing_top_level = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in payload]
    if missing_top_level:
        raise ValueError(f"Missing top-level sample keys: {missing_top_level}")

    assimilation = payload["assimilation"]
    if not isinstance(assimilation, dict):
        raise TypeError("Expected payload['assimilation'] to be a dictionary.")

    missing_assimilation = [
        key for key in REQUIRED_ASSIMILATION_KEYS if key not in assimilation
    ]
    if missing_assimilation:
        raise ValueError(f"Missing assimilation keys: {missing_assimilation}")


def load_aardvark_observation_sample(
    path: Path | str = DEFAULT_AARDVARK_SAMPLE_PATH,
) -> AardvarkObservationSample:
    """Load Aardvark's model-ready observation sample from pickle."""
    sample_path = Path(path)
    if not sample_path.exists():
        raise FileNotFoundError(f"Aardvark observation sample not found: {sample_path}")

    payload = load_torch_pickle_on_cpu(sample_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected sample payload to be a dict, got {type(payload)}")

    validate_aardvark_observation_payload(payload)
    return AardvarkObservationSample(path=sample_path, payload=payload)
