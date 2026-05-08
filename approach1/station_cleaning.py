"""Clean normalized station observations into encoder-ready tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from approach1.station_api import (
    AIR_TEMPERATURE,
    RAINFALL,
    RELATIVE_HUMIDITY,
    WIND_DIRECTION,
    WIND_SPEED,
)

if TYPE_CHECKING:
    import torch

KNOT_TO_MPS = 0.514444

FEATURE_NAMES = [
    "latitude",
    "longitude",
    "hour_sin",
    "hour_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "temperature_c",
    "relative_humidity_pct",
    "rainfall_5min_mm",
    "rainfall_present",
    "log1p_rainfall_5min_mm",
    "wind_speed_mps",
    "wind_dir_sin",
    "wind_dir_cos",
    "wind_u_mps",
    "wind_v_mps",
]


@dataclass(frozen=True)
class StationTensor:
    """Model-ready station inputs plus metadata needed to interpret them."""

    features: np.ndarray | torch.Tensor
    mask: np.ndarray | torch.Tensor
    coords: np.ndarray | torch.Tensor
    station_ids: list[str]
    feature_names: list[str]


def station_observations_to_tensor(
    df: pd.DataFrame,
    *,
    as_torch: bool = False,
) -> StationTensor:
    """Convert normalized station observations into encoder-ready arrays.

    Missing or invalid weather values are represented by a zero-filled feature
    value and a mask value of 0. Valid observations have mask value 1.
    """
    _validate_input_columns(df)

    if df.empty:
        features = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        mask = np.zeros_like(features)
        coords = np.zeros((0, 2), dtype=np.float32)
        return _maybe_torch(
            StationTensor(features, mask, coords, [], FEATURE_NAMES.copy()),
            as_torch,
        )

    station_frame = _build_station_frame(df)
    station_ids = station_frame["station_id"].astype(str).tolist()
    station_index = {station_id: idx for idx, station_id in enumerate(station_ids)}

    features = np.zeros((len(station_ids), len(FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros_like(features)
    coords = station_frame[["latitude", "longitude"]].to_numpy(dtype=np.float32)
    _put_spatial_features(features, mask, coords)
    _put_time_features(features, mask, station_frame)

    for row in df.itertuples(index=False):
        station_id = str(row.station_id)
        if station_id not in station_index:
            continue
        _put_clean_value(
            features,
            mask,
            station_index[station_id],
            str(row.variable),
            row.value,
        )
    _put_derived_wind_components(features, mask)

    return _maybe_torch(
        StationTensor(features, mask, coords, station_ids, FEATURE_NAMES.copy()),
        as_torch,
    )


def _put_clean_value(
    features: np.ndarray,
    mask: np.ndarray,
    station_idx: int,
    variable: str,
    value: object,
) -> None:
    clean_value = _as_float(value)
    if clean_value is None:
        return

    if variable == AIR_TEMPERATURE and 10.0 <= clean_value <= 45.0:
        _set_feature(features, mask, station_idx, "temperature_c", clean_value)
    elif variable == RELATIVE_HUMIDITY and 0.0 <= clean_value <= 100.0:
        _set_feature(features, mask, station_idx, "relative_humidity_pct", clean_value)
    elif variable == RAINFALL and clean_value >= 0.0:
        _set_feature(features, mask, station_idx, "rainfall_5min_mm", clean_value)
        _set_feature(
            features,
            mask,
            station_idx,
            "rainfall_present",
            float(clean_value > 0.0),
        )
        _set_feature(
            features,
            mask,
            station_idx,
            "log1p_rainfall_5min_mm",
            np.log1p(clean_value),
        )
    elif variable == WIND_SPEED and clean_value >= 0.0:
        _set_feature(
            features,
            mask,
            station_idx,
            "wind_speed_mps",
            clean_value * KNOT_TO_MPS,
        )
    elif variable == WIND_DIRECTION and 0.0 <= clean_value <= 360.0:
        radians = np.deg2rad(clean_value)
        _set_feature(features, mask, station_idx, "wind_dir_sin", np.sin(radians))
        _set_feature(features, mask, station_idx, "wind_dir_cos", np.cos(radians))


def _build_station_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["station_id", "latitude", "longitude"]
    if "observed_time" in df.columns:
        columns.append("observed_time")

    station_frame = df[columns].copy()
    station_frame["latitude"] = pd.to_numeric(
        station_frame["latitude"],
        errors="coerce",
    )
    station_frame["longitude"] = pd.to_numeric(
        station_frame["longitude"],
        errors="coerce",
    )
    return (
        station_frame.dropna(subset=["station_id", "latitude", "longitude"])
        .drop_duplicates(subset=["station_id"])
        .sort_values("station_id")
    )


def _put_spatial_features(
    features: np.ndarray,
    mask: np.ndarray,
    coords: np.ndarray,
) -> None:
    if coords.size == 0:
        return

    latitudes = coords[:, 0]
    longitudes = coords[:, 1]

    for station_idx, (latitude, longitude) in enumerate(
        zip(latitudes, longitudes, strict=True),
    ):
        _set_feature(features, mask, station_idx, "latitude", latitude)
        _set_feature(features, mask, station_idx, "longitude", longitude)


def _put_time_features(
    features: np.ndarray,
    mask: np.ndarray,
    station_frame: pd.DataFrame,
) -> None:
    if "observed_time" not in station_frame.columns:
        return

    timestamps = pd.to_datetime(station_frame["observed_time"], errors="coerce")
    for station_idx, timestamp in enumerate(timestamps):
        if pd.isna(timestamp):
            continue

        hour = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
        hour_angle = 2 * np.pi * hour / 24.0
        day_angle = 2 * np.pi * timestamp.dayofyear / 366.0

        _set_feature(features, mask, station_idx, "hour_sin", np.sin(hour_angle))
        _set_feature(features, mask, station_idx, "hour_cos", np.cos(hour_angle))
        _set_feature(features, mask, station_idx, "dayofyear_sin", np.sin(day_angle))
        _set_feature(features, mask, station_idx, "dayofyear_cos", np.cos(day_angle))


def _put_derived_wind_components(features: np.ndarray, mask: np.ndarray) -> None:
    speed_idx = FEATURE_NAMES.index("wind_speed_mps")
    sin_idx = FEATURE_NAMES.index("wind_dir_sin")
    cos_idx = FEATURE_NAMES.index("wind_dir_cos")
    u_idx = FEATURE_NAMES.index("wind_u_mps")
    v_idx = FEATURE_NAMES.index("wind_v_mps")

    has_wind = (
        (mask[:, speed_idx] == 1.0)
        & (mask[:, sin_idx] == 1.0)
        & (mask[:, cos_idx] == 1.0)
    )
    if not np.any(has_wind):
        return

    features[has_wind, u_idx] = features[has_wind, speed_idx] * features[
        has_wind,
        sin_idx,
    ]
    features[has_wind, v_idx] = features[has_wind, speed_idx] * features[
        has_wind,
        cos_idx,
    ]
    mask[has_wind, u_idx] = 1.0
    mask[has_wind, v_idx] = 1.0


def _set_feature(
    features: np.ndarray,
    mask: np.ndarray,
    station_idx: int,
    feature_name: str,
    value: float,
) -> None:
    feature_idx = FEATURE_NAMES.index(feature_name)
    features[station_idx, feature_idx] = np.float32(value)
    mask[station_idx, feature_idx] = 1.0


def _as_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _validate_input_columns(df: pd.DataFrame) -> None:
    required = {"station_id", "latitude", "longitude", "variable", "value"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required station observation columns: {missing}")


def _maybe_torch(result: StationTensor, as_torch: bool) -> StationTensor:
    if not as_torch:
        return result

    import torch

    return StationTensor(
        features=torch.from_numpy(result.features),
        mask=torch.from_numpy(result.mask),
        coords=torch.from_numpy(result.coords),
        station_ids=result.station_ids,
        feature_names=result.feature_names,
    )
