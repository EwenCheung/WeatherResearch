"""Load and validate Aardvark model-ready observation samples."""

from __future__ import annotations

from datetime import datetime, timedelta
from math import atan2, pi
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

from weather_research.project_paths import AARDVARK_DATA_DIR
from weather_research.torch_pickle import load_torch_pickle_on_cpu
from weather_research.weather_state_schema import AardvarkObservationSample

DEFAULT_AARDVARK_SAMPLE_PATH = AARDVARK_DATA_DIR / "sample_data_final.pkl"
DAYS_IN_YEAR = 365.25

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


class AardvarkObservationSourceSpec(NamedTuple):
    source_id: str
    source_name: str
    family: str
    value_key: str
    location_key: str | None
    channels_or_variables: int | tuple[str, ...]
    information_given: str


AARDVARK_OBSERVATION_SOURCE_SPECS: tuple[AardvarkObservationSourceSpec, ...] = (
    AardvarkObservationSourceSpec(
        source_id="satellite_1",
        source_name="HIRS",
        family="satellite radiance",
        value_key="hirs_current",
        location_key="hirs_x_current",
        channels_or_variables=26,
        information_given="infrared sounding radiance channels",
    ),
    AardvarkObservationSourceSpec(
        source_id="satellite_2",
        source_name="AMSU-A",
        family="satellite radiance",
        value_key="amsua_current",
        location_key="amsua_x_current",
        channels_or_variables=13,
        information_given="microwave temperature sounding channels",
    ),
    AardvarkObservationSourceSpec(
        source_id="satellite_3",
        source_name="AMSU-B / MHS",
        family="satellite radiance",
        value_key="amsub_current",
        location_key="amsub_x_current",
        channels_or_variables=12,
        information_given="microwave humidity sounding channels",
    ),
    AardvarkObservationSourceSpec(
        source_id="satellite_4",
        source_name="IASI",
        family="satellite radiance",
        value_key="iasi_current",
        location_key="iasi_x_current",
        channels_or_variables=52,
        information_given="infrared hyperspectral sounding channels",
    ),
    AardvarkObservationSourceSpec(
        source_id="satellite_5",
        source_name="ASCAT",
        family="satellite scatterometer",
        value_key="ascat_current",
        location_key="ascat_x_current",
        channels_or_variables=17,
        information_given="surface wind-related ocean scatterometer features",
    ),
    AardvarkObservationSourceSpec(
        source_id="satellite_6",
        source_name="GRIDSAT-like sample",
        family="satellite gridded imagery",
        value_key="sat_current",
        location_key="sat_x_current",
        channels_or_variables=2,
        information_given="gridded satellite image channels",
    ),
    AardvarkObservationSourceSpec(
        source_id="station_1",
        source_name="HadISD",
        family="land station",
        value_key="y_context_hadisd_current",
        location_key="x_context_hadisd_current",
        channels_or_variables=("tas", "tds", "psl", "u", "v"),
        information_given=(
            "surface station variables: temperature, dew point, pressure, wind"
        ),
    ),
    AardvarkObservationSourceSpec(
        source_id="station_2",
        source_name="ICOADS",
        family="marine station / ship / buoy",
        value_key="icoads_current",
        location_key="icoads_x_current",
        channels_or_variables=5,
        information_given="marine surface observations",
    ),
    AardvarkObservationSourceSpec(
        source_id="profile_1",
        source_name="IGRA",
        family="radiosonde profile",
        value_key="igra_current",
        location_key="igra_x_current",
        channels_or_variables=24,
        information_given="upper-air profile observations",
    ),
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


def _shape_summary(
    value: object,
) -> tuple[int, ...] | list[tuple[int, ...] | str] | str:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, list):
        return [
            tuple(item.shape) if isinstance(item, torch.Tensor) else type(item).__name__
            for item in value
        ]
    return type(value).__name__


def describe_aardvark_observation_sources(
    payload: dict[str, Any],
) -> list[dict[str, object]]:
    """Build a source summary from the loaded Aardvark observation payload."""
    validate_aardvark_observation_payload(payload)
    assimilation = payload["assimilation"]
    rows: list[dict[str, object]] = []

    for spec in AARDVARK_OBSERVATION_SOURCE_SPECS:
        value = assimilation.get(spec.value_key)
        rows.append(
            {
                "source_id": spec.source_id,
                "source_name": spec.source_name,
                "family": spec.family,
                "value_key": spec.value_key,
                "location_key": spec.location_key,
                "shape": _shape_summary(value),
                "channels_or_variables": spec.channels_or_variables,
                "information_given": spec.information_given,
                "available": value is not None,
            }
        )

    return rows


def infer_aardvark_assimilation_time(payload: dict[str, Any]) -> np.datetime64 | None:
    """Infer the current assimilation timestamp from Aardvark aux-time features.

    Aardvark samples do not store a plain timestamp key. The loader encodes time
    as ``[cos(day), sin(day), cos(hour), sin(hour), normalized_year]`` in
    ``assimilation["aux_time_current"]``. This reverses that encoding to the
    nearest day/hour.
    """
    validate_aardvark_observation_payload(payload)
    aux_time = payload["assimilation"].get("aux_time_current")
    if not isinstance(aux_time, torch.Tensor) or aux_time.numel() < 5:
        return None

    values = aux_time.detach().cpu().numpy().reshape(-1, 5)[0]
    day_angle = atan2(float(values[1]), float(values[0]))
    if day_angle < 0:
        day_angle += 2 * pi
    hour_angle = atan2(float(values[3]), float(values[2]))
    if hour_angle < 0:
        hour_angle += 2 * pi

    day_of_year = max(1, round(day_angle * DAYS_IN_YEAR / (2 * pi)))
    hour = round(hour_angle * 24 / (2 * pi)) % 24
    year = round(2007 + float(values[4]) * 15)
    timestamp = datetime(year, 1, 1) + timedelta(days=day_of_year - 1, hours=hour)
    return np.datetime64(timestamp, "ns")
