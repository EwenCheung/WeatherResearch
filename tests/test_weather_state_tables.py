from __future__ import annotations

import numpy as np
import xarray as xr

from weather_research.weather_state_schema import (
    AARDVARK_STATE_SHAPE,
    AardvarkInitialCondition,
)
from weather_research.weather_state_tables import initial_condition_dataset_to_table
from weather_research.weather_state_xarray import initial_condition_to_dataset


def test_initial_condition_dataset_to_table_uses_lat_lon_rows() -> None:
    ic = AardvarkInitialCondition(
        values=np.zeros(AARDVARK_STATE_SHAPE, dtype=np.float32),
    )
    dataset = initial_condition_to_dataset(
        ic,
        time=np.datetime64("2019-01-01T00:00:00"),
    )

    table = initial_condition_dataset_to_table(dataset, max_rows=3)

    assert list(table.columns[:4]) == ["time", "latitude", "longitude", "u10"]
    assert len(table) == 3
    assert "t2m" in table.columns
    assert "mslp" in table.columns


def test_initial_condition_dataset_to_table_selects_time_index() -> None:
    times = np.array(
        ["2019-01-01T00:00:00", "2019-01-01T06:00:00"],
        dtype="datetime64[ns]",
    )
    dataset = initial_condition_to_dataset(
        AardvarkInitialCondition(
            values=np.zeros(AARDVARK_STATE_SHAPE, dtype=np.float32),
        ),
        time=times[0],
    )
    next_dataset = initial_condition_to_dataset(
        AardvarkInitialCondition(
            values=np.ones(AARDVARK_STATE_SHAPE, dtype=np.float32),
        ),
        time=times[1],
    )
    dataset = xr.concat([dataset, next_dataset], dim="time")

    table = initial_condition_dataset_to_table(dataset, t=1, max_rows=1)

    assert table.attrs["ic_time_allowed"] == (
        "2019-01-01T00:00:00.000000000",
        "2019-01-01T06:00:00.000000000",
    )
    assert table.attrs["ic_time_selected"] == "2019-01-01T06:00:00.000000000"
    assert table.loc[0, "u10"] == 1.0


def test_initial_condition_dataset_to_table_selects_time_value() -> None:
    first_dataset = initial_condition_to_dataset(
        AardvarkInitialCondition(
            values=np.zeros(AARDVARK_STATE_SHAPE, dtype=np.float32),
        ),
        time=np.datetime64("2019-01-01T00:00:00"),
    )
    second_dataset = initial_condition_to_dataset(
        AardvarkInitialCondition(
            values=np.full(AARDVARK_STATE_SHAPE, 2.0, dtype=np.float32),
        ),
        time=np.datetime64("2019-01-01T06:00:00"),
    )
    dataset = xr.concat([first_dataset, second_dataset], dim="time")

    table = initial_condition_dataset_to_table(
        dataset,
        t="2019-01-01T06:00:00",
        max_rows=1,
    )

    assert table.attrs["ic_time_selected"] == "2019-01-01T06:00:00.000000000"
    assert table.loc[0, "u10"] == 2.0


def test_initial_condition_dataset_to_table_rejects_out_of_range_time_index() -> None:
    dataset = initial_condition_to_dataset(
        AardvarkInitialCondition(
            values=np.zeros(AARDVARK_STATE_SHAPE, dtype=np.float32),
        ),
        time=np.datetime64("2019-01-01T00:00:00"),
    )

    try:
        initial_condition_dataset_to_table(dataset, t=2)
    except IndexError as exc:
        assert "Allowed IC time range" in str(exc)
    else:
        raise AssertionError("Expected out-of-range t to raise IndexError.")
