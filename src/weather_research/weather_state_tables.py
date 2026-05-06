"""Tabular views for IC-like weather state datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr


def dataset_variable_short_names(dataset: xr.Dataset) -> dict[str, str]:
    """Return a mapping from dataset variable names to compact Aardvark names."""
    names = {}
    for variable_name, data_array in dataset.data_vars.items():
        short_name = data_array.attrs.get("aardvark_variable", variable_name)
        names[variable_name] = str(short_name)
    return names


def initial_condition_dataset_to_table(
    dataset: xr.Dataset,
    *,
    max_rows: int | None = None,
    short_names: bool = True,
) -> pd.DataFrame:
    """Convert an IC-like xarray dataset to a lat/lon table."""
    table = dataset.to_dataframe().reset_index()
    if short_names:
        table = table.rename(columns=dataset_variable_short_names(dataset))

    ordered_columns = [
        "time",
        "latitude",
        "longitude",
        *[column for column in table.columns if column not in {"time", "latitude", "longitude"}],
    ]
    table = table.loc[:, ordered_columns]

    if max_rows is not None:
        return table.head(max_rows)
    return table


def open_initial_condition_table(
    zarr_path: Path | str,
    *,
    max_rows: int | None = None,
    short_names: bool = True,
) -> pd.DataFrame:
    """Open an IC-like Zarr dataset and return a tabular view."""
    dataset = xr.open_zarr(zarr_path)
    return initial_condition_dataset_to_table(
        dataset,
        max_rows=max_rows,
        short_names=short_names,
    )
