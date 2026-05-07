import numpy as np

from weather_research.aardvark_observations import (
    describe_aardvark_observation_sources,
    infer_aardvark_assimilation_time,
    load_aardvark_observation_sample,
)


def test_load_aardvark_observation_sample() -> None:
    sample = load_aardvark_observation_sample()

    assert sample.path.name == "sample_data_final.pkl"
    assert "assimilation" in sample.payload
    assert "forecast" in sample.payload
    assert "downscaling" in sample.payload
    assert "hirs_current" in sample.payload["assimilation"]


def test_infer_aardvark_assimilation_time_from_sample() -> None:
    sample = load_aardvark_observation_sample()

    assert infer_aardvark_assimilation_time(sample.payload) == np.datetime64(
        "2018-08-18T12:00:00.000000000"
    )


def test_describe_aardvark_observation_sources_uses_loaded_shapes() -> None:
    sample = load_aardvark_observation_sample()

    table_rows = describe_aardvark_observation_sources(sample.payload)
    hirs_row = next(row for row in table_rows if row["source_name"] == "HIRS")

    assert hirs_row["value_key"] == "hirs_current"
    assert hirs_row["shape"] == (1, 360, 181, 26)
    assert hirs_row["available"] is True
