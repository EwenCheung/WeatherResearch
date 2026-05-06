"""Schema constants for Aardvark IC-like weather states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

AARDVARK_STATE_VARIABLES: tuple[str, ...] = (
    "u10",
    "v10",
    "t2m",
    "mslp",
    "z200",
    "z500",
    "z700",
    "z850",
    "q200",
    "q500",
    "q700",
    "q850",
    "t200",
    "t500",
    "t700",
    "t850",
    "u200",
    "u500",
    "u700",
    "u850",
    "v200",
    "v500",
    "v700",
    "v850",
)

WEATHERBENCH_STYLE_NAMES: dict[str, str] = {
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "t2m": "2m_temperature",
    "mslp": "mean_sea_level_pressure",
    "z200": "geopotential_200",
    "z500": "geopotential_500",
    "z700": "geopotential_700",
    "z850": "geopotential_850",
    "q200": "specific_humidity_200",
    "q500": "specific_humidity_500",
    "q700": "specific_humidity_700",
    "q850": "specific_humidity_850",
    "t200": "temperature_200",
    "t500": "temperature_500",
    "t700": "temperature_700",
    "t850": "temperature_850",
    "u200": "u_component_of_wind_200",
    "u500": "u_component_of_wind_500",
    "u700": "u_component_of_wind_700",
    "u850": "u_component_of_wind_850",
    "v200": "v_component_of_wind_200",
    "v500": "v_component_of_wind_500",
    "v700": "v_component_of_wind_700",
    "v850": "v_component_of_wind_850",
}

AARDVARK_LATITUDE_SIZE = 121
AARDVARK_LONGITUDE_SIZE = 240
AARDVARK_STATE_SHAPE = (
    AARDVARK_LATITUDE_SIZE,
    AARDVARK_LONGITUDE_SIZE,
    len(AARDVARK_STATE_VARIABLES),
)


def aardvark_latitudes() -> np.ndarray:
    """Return the latitude coordinate used by the Aardvark grid."""
    return np.linspace(-90.0, 90.0, AARDVARK_LATITUDE_SIZE, dtype=np.float32)


def aardvark_longitudes() -> np.ndarray:
    """Return the longitude coordinate used by the Aardvark grid."""
    return np.linspace(0.0, 359.0, AARDVARK_LONGITUDE_SIZE, dtype=np.float32)


@dataclass(frozen=True)
class AardvarkObservationSample:
    """Loaded Aardvark model-ready observation sample."""

    path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class AardvarkInitialCondition:
    """Aardvark 24-variable IC-like physical state."""

    values: np.ndarray
    variable_names: tuple[str, ...] = AARDVARK_STATE_VARIABLES
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if self.values.shape != AARDVARK_STATE_SHAPE:
            raise ValueError(
                f"Expected IC values with shape {AARDVARK_STATE_SHAPE}, "
                f"got {self.values.shape}."
            )
        if len(self.variable_names) != self.values.shape[-1]:
            raise ValueError(
                "Variable name count must match the last IC value dimension."
            )
