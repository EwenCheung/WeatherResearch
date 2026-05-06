"""Convert Aardvark IC-like weather states to xarray and Zarr."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from weather_research.weather_state_schema import (
    WEATHERBENCH_STYLE_NAMES,
    AardvarkInitialCondition,
    aardvark_latitudes,
    aardvark_longitudes,
)


def initial_condition_to_dataset(
    initial_condition: AardvarkInitialCondition,
    *,
    time: np.datetime64 | None = None,
    weatherbench_names: bool = True,
) -> xr.Dataset:
    """Convert an Aardvark IC-like state into a WeatherBench-style xarray dataset."""
    output_time = time or np.datetime64(datetime.now(timezone.utc).replace(tzinfo=None))
    latitudes = aardvark_latitudes()
    longitudes = aardvark_longitudes()
    values = initial_condition.values
    data_vars = {}

    for index, variable_name in enumerate(initial_condition.variable_names):
        output_name = (
            WEATHERBENCH_STYLE_NAMES[variable_name]
            if weatherbench_names
            else variable_name
        )
        data_vars[output_name] = (
            ("time", "latitude", "longitude"),
            values[np.newaxis, :, :, index],
            {
                "aardvark_variable": variable_name,
                "description": "Aardvark IC-like physical state variable.",
            },
        )

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.array([output_time], dtype="datetime64[ns]"),
            "latitude": latitudes,
            "longitude": longitudes,
        },
        attrs={
            "title": "Aardvark IC-like physical weather state",
            "source": str(initial_condition.source_path)
            if initial_condition.source_path
            else "Aardvark observation sample",
            "note": (
                "Generated from Aardvark model-ready observations. This is not "
                "AIFS Appendix A 94-field initial condition data."
            ),
        },
    )
    return dataset


def save_initial_condition_zarr(
    initial_condition: AardvarkInitialCondition,
    output_path: Path | str,
    *,
    time: np.datetime64 | None = None,
    overwrite: bool = True,
) -> xr.Dataset:
    """Save an Aardvark IC-like state to Zarr and return the dataset."""
    dataset = initial_condition_to_dataset(initial_condition, time=time)
    mode = "w" if overwrite else "w-"
    dataset.to_zarr(str(Path(output_path)), mode=mode)  # type: ignore[reportArgumentType]
    return dataset
