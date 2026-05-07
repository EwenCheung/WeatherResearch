from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr

AARDVARK_TARGET_VARIABLES: tuple[str, ...] = (
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

SURFACE_TARGET_MAP: dict[str, str] = {
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "t2m": "2m_temperature",
    "mslp": "mean_sea_level_pressure",
}

LEVEL_TARGET_MAP: dict[str, tuple[str, int]] = {
    "z200": ("geopotential", 200),
    "z500": ("geopotential", 500),
    "z700": ("geopotential", 700),
    "z850": ("geopotential", 850),
    "q200": ("specific_humidity", 200),
    "q500": ("specific_humidity", 500),
    "q700": ("specific_humidity", 700),
    "q850": ("specific_humidity", 850),
    "t200": ("temperature", 200),
    "t500": ("temperature", 500),
    "t700": ("temperature", 700),
    "t850": ("temperature", 850),
    "u200": ("u_component_of_wind", 200),
    "u500": ("u_component_of_wind", 500),
    "u700": ("u_component_of_wind", 700),
    "u850": ("u_component_of_wind", 850),
    "v200": ("v_component_of_wind", 200),
    "v500": ("v_component_of_wind", 500),
    "v700": ("v_component_of_wind", 700),
    "v850": ("v_component_of_wind", 850),
}


def open_weatherbench_era5(path: str) -> xr.Dataset:
    """Open local or cloud WeatherBench2 ERA5 Zarr data lazily."""
    return xr.open_zarr(path, consolidated=True)


def load_target_normalization(norm_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Aardvark 24-variable mean/std vectors."""
    root = Path(norm_dir)
    means = np.load(root / "mean_4u_1.npy").astype(np.float32)
    stds = np.load(root / "std_4u_1.npy").astype(np.float32)
    if means.shape != (24,) or stds.shape != (24,):
        raise ValueError(
            f"Expected 24 mean/std values, got {means.shape}, {stds.shape}"
        )
    return means, stds


def extract_aardvark_target(
    era5: xr.Dataset,
    time: np.datetime64 | str,
    *,
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
) -> torch.Tensor:
    """Extract one `(121, 240, 24)` target from WeatherBench2 ERA5."""
    selected = era5.sel(time=np.datetime64(time))
    fields: list[np.ndarray] = []

    for name in AARDVARK_TARGET_VARIABLES:
        if name in SURFACE_TARGET_MAP:
            array = selected[SURFACE_TARGET_MAP[name]].values
        else:
            variable, level = LEVEL_TARGET_MAP[name]
            array = selected[variable].sel(level=level).values
        fields.append(_lon_lat_to_lat_lon(np.asarray(array, dtype=np.float32)))

    target = np.stack(fields, axis=-1)
    target = _ensure_lat_lon_variable_order(target)

    if means is not None and stds is not None:
        target = (target - means.reshape(1, 1, -1)) / stds.reshape(1, 1, -1)

    return torch.from_numpy(target.astype(np.float32, copy=False))


def _ensure_lat_lon_variable_order(target: np.ndarray) -> np.ndarray:
    if target.shape != (121, 240, 24):
        raise ValueError(f"Expected target shape (121, 240, 24), got {target.shape}")
    return target


def extract_aardvark_climatology(
    climatology: xr.Dataset,
    time: np.datetime64 | str,
    *,
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
) -> torch.Tensor:
    """Extract one normalized climatology tensor shaped `(24, 240, 121)`."""
    timestamp = _as_datetime(time)
    selected = climatology.sel(
        hour=timestamp.hour,
        dayofyear=timestamp.timetuple().tm_yday,
    )
    fields: list[np.ndarray] = []

    for name in AARDVARK_TARGET_VARIABLES:
        if name in SURFACE_TARGET_MAP:
            array = selected[SURFACE_TARGET_MAP[name]].values
        else:
            variable, level = LEVEL_TARGET_MAP[name]
            array = selected[variable].sel(level=level).values
        fields.append(np.asarray(array, dtype=np.float32))

    stacked = np.stack(fields, axis=0)
    if stacked.shape != (24, 240, 121):
        raise ValueError(
            f"Expected climatology shape (24, 240, 121), got {stacked.shape}"
        )

    if means is not None and stds is not None:
        stacked = (
            stacked.transpose(1, 2, 0) - means.reshape(1, 1, -1)
        ) / stds.reshape(1, 1, -1)
        stacked = stacked.transpose(2, 0, 1)

    return torch.from_numpy(stacked.astype(np.float32, copy=False))


def _lon_lat_to_lat_lon(array: np.ndarray) -> np.ndarray:
    if array.shape != (240, 121):
        raise ValueError(f"Expected `(longitude, latitude)` array, got {array.shape}")
    return array.T


def _as_datetime(time: np.datetime64 | str) -> datetime:
    value = np.datetime64(time, "s").astype(datetime)
    if not isinstance(value, datetime):
        raise TypeError(f"Expected datetime-compatible value, got {type(value)}")
    return value
