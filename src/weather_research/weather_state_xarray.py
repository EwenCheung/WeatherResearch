"""Convert Aardvark IC-like weather states to xarray and Zarr."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
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
    output_time = time or np.datetime64(datetime.now(UTC).replace(tzinfo=None))
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


def initial_conditions_to_dataset(
    initial_conditions: Sequence[AardvarkInitialCondition],
    *,
    times: Sequence[np.datetime64 | str],
    weatherbench_names: bool = True,
) -> xr.Dataset:
    """Convert multiple Aardvark IC-like states into one time-indexed dataset."""
    if len(initial_conditions) != len(times):
        raise ValueError(
            "Initial condition count must match time count: "
            f"{len(initial_conditions)} ICs, {len(times)} times."
        )
    if len(initial_conditions) == 0:
        raise ValueError("Expected at least one initial condition.")

    datasets = [
        initial_condition_to_dataset(
            initial_condition,
            time=np.datetime64(time),
            weatherbench_names=weatherbench_names,
        )
        for initial_condition, time in zip(initial_conditions, times, strict=True)
    ]
    combined = xr.concat(datasets, dim="time").sortby("time")
    combined.attrs["source"] = "Multiple Aardvark observation samples"
    combined.attrs["source_paths"] = [
        str(initial_condition.source_path)
        for initial_condition in initial_conditions
        if initial_condition.source_path is not None
    ]
    return combined


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
