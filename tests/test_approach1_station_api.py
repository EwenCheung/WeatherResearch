from __future__ import annotations

import numpy as np
import pandas as pd

from approach1.station_api import (
    AIR_TEMPERATURE,
    NORMALIZED_COLUMNS,
    RAINFALL,
    WIND_DIRECTION,
    WIND_SPEED,
    normalize_station_payload,
)
from approach1.station_cleaning import (
    FEATURE_NAMES,
    KNOT_TO_MPS,
    station_observations_to_tensor,
)


def _payload(
    value: float,
    *,
    unit: str = "deg C",
    reading_type: str = "reading",
) -> dict:
    return {
        "code": 0,
        "data": {
            "stations": [
                {
                    "id": "S001",
                    "deviceId": "S001",
                    "name": "Station One",
                    "location": {"latitude": 1.3, "longitude": 103.8},
                }
            ],
            "readings": [
                {
                    "timestamp": "2026-05-07T16:00:00+08:00",
                    "data": [{"stationId": "S001", "value": value}],
                }
            ],
            "readingType": reading_type,
            "readingUnit": unit,
        },
        "errorMsg": "",
    }


def test_normalize_station_payload_returns_long_table_schema() -> None:
    table = normalize_station_payload(
        _payload(31.2),
        AIR_TEMPERATURE,
        "2026-05-07T16:00:00+08:00",
    )

    assert list(table.columns) == NORMALIZED_COLUMNS
    assert table.loc[0, "requested_time"] == "2026-05-07T16:00:00+08:00"
    assert table.loc[0, "observed_time"] == "2026-05-07T16:00:00+08:00"
    assert table.loc[0, "station_id"] == "S001"
    assert table.loc[0, "variable"] == AIR_TEMPERATURE
    assert table.loc[0, "value"] == 31.2


def test_station_observations_to_tensor_converts_wind_speed_knots_to_mps() -> None:
    table = normalize_station_payload(
        _payload(10.0, unit="knots"),
        WIND_SPEED,
        "2026-05-07T16:00:00+08:00",
    )

    tensor = station_observations_to_tensor(table)
    speed_idx = FEATURE_NAMES.index("wind_speed_mps")

    assert tensor.features.shape == (1, len(FEATURE_NAMES))
    assert np.isclose(tensor.features[0, speed_idx], 10.0 * KNOT_TO_MPS)
    assert tensor.mask[0, speed_idx] == 1.0


def test_station_observations_to_tensor_adds_spatial_and_time_features() -> None:
    table = normalize_station_payload(
        _payload(31.0),
        AIR_TEMPERATURE,
        "2026-05-07T16:00:00+08:00",
    )

    tensor = station_observations_to_tensor(table)
    lat_idx = FEATURE_NAMES.index("latitude")
    lon_idx = FEATURE_NAMES.index("longitude")
    hour_sin_idx = FEATURE_NAMES.index("hour_sin")
    hour_cos_idx = FEATURE_NAMES.index("hour_cos")

    assert np.isclose(tensor.features[0, lat_idx], 1.3)
    assert np.isclose(tensor.features[0, lon_idx], 103.8)
    assert tensor.mask[0, lat_idx] == 1.0
    assert tensor.mask[0, lon_idx] == 1.0
    assert np.isclose(tensor.features[0, hour_sin_idx], np.sin(2 * np.pi * 16 / 24))
    assert np.isclose(tensor.features[0, hour_cos_idx], np.cos(2 * np.pi * 16 / 24))


def test_station_observations_to_tensor_adds_derived_rainfall_features() -> None:
    table = normalize_station_payload(
        _payload(2.4, unit="mm"),
        RAINFALL,
        "2026-05-07T16:00:00+08:00",
    )

    tensor = station_observations_to_tensor(table)
    rainfall_idx = FEATURE_NAMES.index("rainfall_5min_mm")
    present_idx = FEATURE_NAMES.index("rainfall_present")
    log_idx = FEATURE_NAMES.index("log1p_rainfall_5min_mm")

    assert np.isclose(tensor.features[0, rainfall_idx], 2.4)
    assert tensor.features[0, present_idx] == 1.0
    assert np.isclose(tensor.features[0, log_idx], np.log1p(2.4))
    assert tensor.mask[0, rainfall_idx] == 1.0
    assert tensor.mask[0, present_idx] == 1.0
    assert tensor.mask[0, log_idx] == 1.0


def test_station_observations_to_tensor_encodes_wind_direction_as_sin_cos() -> None:
    table = normalize_station_payload(
        _payload(90.0, unit="degrees"),
        WIND_DIRECTION,
        "2026-05-07T16:00:00+08:00",
    )

    tensor = station_observations_to_tensor(table)
    sin_idx = FEATURE_NAMES.index("wind_dir_sin")
    cos_idx = FEATURE_NAMES.index("wind_dir_cos")

    assert np.isclose(tensor.features[0, sin_idx], 1.0)
    assert np.isclose(tensor.features[0, cos_idx], 0.0, atol=1e-6)
    assert tensor.mask[0, sin_idx] == 1.0
    assert tensor.mask[0, cos_idx] == 1.0


def test_station_observations_to_tensor_adds_wind_vector_components() -> None:
    table = pd.concat(
        [
            normalize_station_payload(
                _payload(10.0, unit="knots"),
                WIND_SPEED,
                "2026-05-07T16:00:00+08:00",
            ),
            normalize_station_payload(
                _payload(90.0, unit="degrees"),
                WIND_DIRECTION,
                "2026-05-07T16:00:00+08:00",
            ),
        ],
        ignore_index=True,
    )

    tensor = station_observations_to_tensor(table)
    u_idx = FEATURE_NAMES.index("wind_u_mps")
    v_idx = FEATURE_NAMES.index("wind_v_mps")

    assert np.isclose(tensor.features[0, u_idx], 10.0 * KNOT_TO_MPS)
    assert np.isclose(tensor.features[0, v_idx], 0.0, atol=1e-6)
    assert tensor.mask[0, u_idx] == 1.0
    assert tensor.mask[0, v_idx] == 1.0


def test_invalid_values_become_missing_in_tensor() -> None:
    table = pd.concat(
        [
            normalize_station_payload(
                _payload(50.0),
                AIR_TEMPERATURE,
                "2026-05-07T16:00:00+08:00",
            ),
            normalize_station_payload(
                _payload(-1.0, unit="mm"),
                RAINFALL,
                "2026-05-07T16:00:00+08:00",
            ),
        ],
        ignore_index=True,
    )

    tensor = station_observations_to_tensor(table)
    temp_idx = FEATURE_NAMES.index("temperature_c")
    rain_idx = FEATURE_NAMES.index("rainfall_5min_mm")

    assert tensor.features[0, temp_idx] == 0.0
    assert tensor.features[0, rain_idx] == 0.0
    assert tensor.mask[0, temp_idx] == 0.0
    assert tensor.mask[0, rain_idx] == 0.0


def test_missing_endpoint_values_do_not_break_tensor_creation() -> None:
    table = normalize_station_payload(
        _payload(31.0),
        AIR_TEMPERATURE,
        "2026-05-07T16:00:00+08:00",
    )

    tensor = station_observations_to_tensor(table)
    temp_idx = FEATURE_NAMES.index("temperature_c")
    humidity_idx = FEATURE_NAMES.index("relative_humidity_pct")

    assert tensor.station_ids == ["S001"]
    assert np.allclose(tensor.coords, [[1.3, 103.8]])
    assert tensor.features[0, temp_idx] == 31.0
    assert tensor.mask[0, temp_idx] == 1.0
    assert tensor.features[0, humidity_idx] == 0.0
    assert tensor.mask[0, humidity_idx] == 0.0
