from __future__ import annotations

import numpy as np
import xarray as xr

from weather_research.weather_state_schema import (
    AARDVARK_STATE_SHAPE,
    AardvarkInitialCondition,
)
from weather_research.weather_state_xarray import (
    initial_condition_to_dataset,
    save_initial_condition_zarr,
)


def test_initial_condition_to_dataset() -> None:
    ic = AardvarkInitialCondition(
        values=np.zeros(AARDVARK_STATE_SHAPE, dtype=np.float32),
    )

    dataset = initial_condition_to_dataset(
        ic,
        time=np.datetime64("2019-01-01T00:00:00"),
    )

    assert dict(dataset.sizes) == {"time": 1, "latitude": 121, "longitude": 240}
    assert len(dataset.data_vars) == 24
    assert "2m_temperature" in dataset


def test_save_initial_condition_zarr(tmp_path) -> None:
    ic = AardvarkInitialCondition(
        values=np.ones(AARDVARK_STATE_SHAPE, dtype=np.float32),
    )
    output_path = tmp_path / "ic.zarr"

    save_initial_condition_zarr(ic, output_path)
    reopened = xr.open_zarr(output_path)

    assert "2m_temperature" in reopened
    assert reopened["2m_temperature"].shape == (1, 121, 240)
