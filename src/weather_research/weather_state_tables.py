"""Tabular views for IC-like weather state datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

type TimeSelector = int | str | np.datetime64 | pd.Timestamp


def dataset_variable_short_names(dataset: xr.Dataset) -> dict[str, str]:
    """Return a mapping from dataset variable names to compact Aardvark names."""
    names = {}
    for variable_name, data_array in dataset.data_vars.items():
        short_name = data_array.attrs.get("aardvark_variable", variable_name)
        names[variable_name] = str(short_name)
    return names


def _format_time_value(value: object) -> str:
    """Return a stable string for numpy/pandas/xarray time values."""
    if isinstance(value, np.datetime64):
        return str(value)
    return str(np.asarray(value).item())


def _ic_time_allowed_range(dataset: xr.Dataset) -> tuple[str, str]:
    if "time" not in dataset.coords:
        raise ValueError("Expected IC dataset to include a 'time' coordinate.")
    if dataset.sizes.get("time", 0) < 1:
        raise ValueError("Expected IC dataset to contain at least one time step.")

    times = dataset["time"].values
    return (_format_time_value(times[0]), _format_time_value(times[-1]))


def _select_time(dataset: xr.Dataset, t: TimeSelector) -> xr.Dataset:
    """Select one IC time by zero-based index or time coordinate value."""
    if isinstance(t, int):
        time_size = dataset.sizes["time"]
        if not -time_size <= t < time_size:
            allowed = _ic_time_allowed_range(dataset)
            raise IndexError(
                f"IC time index t={t} is out of range for {time_size} time steps. "
                f"Allowed IC time range: {allowed}."
            )
        return dataset.isel(time=t)

    try:
        return dataset.sel(time=np.datetime64(t))
    except (KeyError, ValueError) as exc:
        allowed = _ic_time_allowed_range(dataset)
        raise KeyError(
            f"IC time t={t!r} is not available. Allowed IC time range: {allowed}."
        ) from exc


def initial_condition_dataset_to_table(
    dataset: xr.Dataset,
    *,
    t: TimeSelector = 0,
    max_rows: int | None = None,
    short_names: bool = True,
) -> pd.DataFrame:
    """Convert one IC-like xarray time step to a lat/lon table.

    Args:
        dataset: IC-like dataset with ``time``, ``latitude``, and ``longitude`` dims.
        t: Time selector. Integers are zero-based time indexes. Strings,
            ``np.datetime64``, and ``pd.Timestamp`` values select by time coordinate.
        max_rows: Optional preview row limit.
        short_names: Rename WeatherBench-style columns to compact Aardvark names.

    The returned table includes ``ic_time_allowed`` and ``ic_time_selected`` in
    ``DataFrame.attrs`` so notebooks can display the valid time range next to
    the table.
    """
    time_allowed = _ic_time_allowed_range(dataset)
    selected_dataset = _select_time(dataset, t)
    selected_time = _format_time_value(selected_dataset["time"].values)

    table = selected_dataset.to_dataframe().reset_index()
    if short_names:
        table = table.rename(columns=dataset_variable_short_names(dataset))

    ordered_columns = [
        "time",
        "latitude",
        "longitude",
        *[
            column
            for column in table.columns
            if column not in {"time", "latitude", "longitude"}
        ],
    ]
    table = table.loc[:, ordered_columns]
    table.attrs["ic_time_allowed"] = time_allowed
    table.attrs["ic_time_selected"] = selected_time

    if max_rows is not None:
        return table.head(max_rows)
    return table


def open_initial_condition_table(
    zarr_path: Path | str,
    *,
    t: TimeSelector = 0,
    max_rows: int | None = None,
    short_names: bool = True,
) -> pd.DataFrame:
    """Open an IC-like Zarr dataset and return a tabular view."""
    dataset = xr.open_zarr(zarr_path)
    return initial_condition_dataset_to_table(
        dataset,
        t=t,
        max_rows=max_rows,
        short_names=short_names,
    )
