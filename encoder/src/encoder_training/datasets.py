from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from encoder_training.targets import (
    extract_aardvark_climatology,
    extract_aardvark_target,
)
from weather_research.aardvark_observations import load_aardvark_observation_sample

LATLON_SCALE_FACTOR = 360.0


class RepeatedAardvarkSampleDataset(Dataset[dict[str, Any]]):
    """Repeat a model-ready Aardvark sample for trainer smoke tests.

    This is not a meaningful training dataset. It exists to verify the model,
    device handling, loss, and checkpoint path before downloading large data.
    """

    def __init__(self, sample_path: str | Path, *, repeats: int = 2) -> None:
        self.sample = load_aardvark_observation_sample(sample_path)
        self.repeats = repeats

    def __len__(self) -> int:
        return self.repeats

    def __getitem__(self, index: int) -> dict[str, Any]:
        del index
        payload = self.sample.payload
        assimilation = payload["assimilation"]
        y_target = assimilation.get("y_target", payload.get("y_target"))
        if not isinstance(y_target, torch.Tensor):
            raise ValueError("Sample does not contain a tensor y_target for training.")
        return {"assimilation": assimilation, "y_target": y_target}


class AmsuAscatPipelineADataset(Dataset[dict[str, Any]]):
    """Pipeline A debug dataset with real AMSU-A, AMSU-B, and ASCAT observations."""

    def __init__(
        self,
        *,
        observation_root: str | Path,
        era5: xr.Dataset,
        climatology: xr.Dataset,
        times: list[np.datetime64],
        means: np.ndarray,
        stds: np.ndarray,
        elevation: torch.Tensor | None = None,
    ) -> None:
        self.observation_root = Path(observation_root)
        self.era5 = era5
        self.climatology = climatology
        self.times = times
        self.means = means
        self.stds = stds
        self.training_data_root = _resolve_training_data_root(self.observation_root)
        self.amsua = xr.open_dataset(self.training_data_root / "amsua_data_v1.nc")
        self.amsub = xr.open_dataset(self.training_data_root / "amsub_data_v1.nc")
        self.ascat = xr.open_dataset(self.training_data_root / "ascat_data_v1.nc")
        self.elevation = elevation or torch.zeros((1, 7, 121, 240), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, index: int) -> dict[str, Any]:
        time = self.times[index]
        assimilation = self._build_assimilation(time)
        y_target = extract_aardvark_target(
            self.era5,
            time,
            means=self.means,
            stds=self.stds,
        ).unsqueeze(0)
        assimilation["y_target"] = y_target
        assimilation["y_target_current"] = y_target
        return {
            "assimilation": assimilation,
            "y_target": y_target,
            "time": str(time),
        }

    def _build_assimilation(self, time: np.datetime64) -> dict[str, Any]:
        amsua = _dataset_to_tensor(
            self.amsua,
            time,
            [
                "bt_channel_4",
                "bt_channel_5",
                "bt_channel_6",
                "bt_channel_7",
                "bt_channel_8",
                "bt_channel_9",
                "bt_channel_10",
                "bt_channel_11",
                "bt_channel_12",
                "bt_channel_13",
                "bt_channel_14",
                "minimum_view_zenith_angle",
                "obs_time",
            ],
            transpose_to_lon_lat=False,
        )
        amsub = _dataset_to_tensor(
            self.amsub,
            time,
            [
                "lon",
                "lat",
                "obs_time",
                "orbital_mode",
                "surface_type",
                "earth_angle_of_incidence",
                "solar_zenith_angle",
                "bt_channel_1",
                "bt_channel_2",
                "bt_channel_3",
                "bt_channel_4",
                "bt_channel_5",
            ],
            transpose_to_lon_lat=True,
        )
        ascat = _dataset_to_tensor(
            self.ascat,
            time,
            [
                "lon",
                "lat",
                "sat_track_azi",
                "as_des_pass",
                "obs_time",
                "beam_1_sigma0",
                "beam_2_sigma0",
                "beam_3_sigma0",
                "beam_1_inc_angle",
                "beam_2_inc_angle",
                "beam_3_inc_angle",
                "beam_1_azi_angle",
                "beam_2_azi_angle",
                "beam_3_azi_angle",
                "beam_1_kp",
                "beam_2_kp",
                "beam_3_kp",
            ],
            transpose_to_lon_lat=True,
        )
        climatology = extract_aardvark_climatology(
            self.climatology,
            time,
            means=self.means,
            stds=self.stds,
        ).unsqueeze(0)

        return {
            "amsua_current": _normalize_per_sample(amsua),
            "amsua_x_current": _grid_from_dataset(self.amsua),
            "amsub_current": _normalize_per_sample(amsub),
            "amsub_x_current": _grid_from_dataset(self.amsub),
            "ascat_current": _normalize_per_sample(ascat),
            "ascat_x_current": _grid_from_dataset(self.ascat),
            "iasi_current": torch.zeros((1, 360, 181, 52), dtype=torch.float32),
            "iasi_x_current": _regular_grid(360, 181),
            "hirs_current": torch.zeros((1, 360, 181, 26), dtype=torch.float32),
            "hirs_x_current": _regular_grid(360, 181),
            "sat_current": torch.zeros((1, 2, 514, 200), dtype=torch.float32),
            "sat_x_current": _regular_grid(514, 200),
            "icoads_current": torch.full((1, 5, 12000), torch.nan),
            "icoads_x_current": [
                torch.full((1, 12000), torch.nan),
                torch.full((1, 12000), torch.nan),
            ],
            "igra_current": torch.full((1, 24, 1375), torch.nan),
            "igra_x_current": [
                torch.full((1, 1375), torch.nan),
                torch.full((1, 1375), torch.nan),
            ],
            "x_context_hadisd_current": _empty_hadisd_locations(),
            "y_context_hadisd_current": _empty_hadisd_values(),
            "era5_elev_current": self.elevation,
            "era5_lonlat_current": torch.zeros((1, 2, 240, 121), dtype=torch.float32),
            "era5_x_current": _regular_grid(240, 121),
            "climatology_current": climatology,
            "aux_time_current": _aux_time_features(time),
            "lt": torch.ones((1, 1), dtype=torch.float32),
        }


def daily_times(start: str, end: str) -> list[np.datetime64]:
    """Return daily timestamps matching the HF Aardvark observation files."""
    dates = pd.date_range(start, end, freq="1D")
    return [np.datetime64(value, "ns") for value in dates]


def _dataset_to_tensor(
    dataset: xr.Dataset,
    time: np.datetime64,
    variables: list[str],
    *,
    transpose_to_lon_lat: bool,
) -> torch.Tensor:
    fields = []
    selected = dataset.sel(time=time)
    for variable in variables:
        values = np.asarray(selected[variable].values, dtype=np.float32)
        if transpose_to_lon_lat:
            values = values.T
        fields.append(values)
    array = np.stack(fields, axis=-1)
    return torch.from_numpy(array).unsqueeze(0)


def _grid_from_dataset(dataset: xr.Dataset) -> list[torch.Tensor]:
    lon = np.asarray(dataset.longitude.values, dtype=np.float32) / LATLON_SCALE_FACTOR
    lat = np.asarray(dataset.latitude.values, dtype=np.float32) / LATLON_SCALE_FACTOR
    return [
        torch.from_numpy(lon).reshape(1, -1),
        torch.from_numpy(lat).reshape(1, -1),
    ]


def _regular_grid(width: int, height: int) -> list[torch.Tensor]:
    lon = torch.linspace(0.0, 359.0, width).reshape(1, -1) / LATLON_SCALE_FACTOR
    lat = torch.linspace(-90.0, 90.0, height).reshape(1, -1) / LATLON_SCALE_FACTOR
    return [lon.float(), lat.float()]


def _normalize_per_sample(tensor: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(tensor)
    normalized = tensor.clone()
    for channel in range(tensor.shape[-1]):
        channel_values = tensor[..., channel]
        channel_mask = mask[..., channel]
        if not bool(channel_mask.any()):
            continue
        mean = channel_values[channel_mask].mean()
        std = channel_values[channel_mask].std().clamp_min(1e-6)
        normalized[..., channel] = (channel_values - mean) / std
    return normalized


def _empty_hadisd_locations() -> list[torch.Tensor]:
    return [
        torch.full((1, 2, 1), torch.nan),
        torch.full((1, 2, 1), torch.nan),
        torch.full((1, 2, 1), torch.nan),
        torch.full((1, 2, 1), torch.nan),
        torch.full((1, 2, 1), torch.nan),
    ]


def _empty_hadisd_values() -> list[torch.Tensor]:
    return [
        torch.full((1, 1), torch.nan),
        torch.full((1, 1), torch.nan),
        torch.full((1, 1), torch.nan),
        torch.full((1, 1), torch.nan),
        torch.full((1, 1), torch.nan),
    ]


def _aux_time_features(time: np.datetime64) -> torch.Tensor:
    timestamp = np.datetime64(time, "s").astype(datetime)
    if not isinstance(timestamp, datetime):
        raise TypeError(f"Expected datetime-compatible value, got {type(timestamp)}")
    day_fraction = 2 * np.pi * timestamp.timetuple().tm_yday / 365.25
    hour_fraction = 2 * np.pi * timestamp.hour / 24
    year_fraction = (timestamp.year - 2007) / 15
    return torch.tensor(
        [
            [
                np.cos(day_fraction),
                np.sin(day_fraction),
                np.cos(hour_fraction),
                np.sin(hour_fraction),
                year_fraction,
            ]
        ],
        dtype=torch.float32,
    )


def _resolve_training_data_root(root: Path) -> Path:
    if (root / "amsua_data_v1.nc").exists():
        return root
    if (root / "training_data" / "amsua_data_v1.nc").exists():
        return root / "training_data"
    raise FileNotFoundError(
        "Could not find `amsua_data_v1.nc` under "
        f"{root} or {root / 'training_data'}."
    )
