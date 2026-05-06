from __future__ import annotations

import numpy as np

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
