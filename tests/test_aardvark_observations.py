from weather_research.aardvark_observations import load_aardvark_observation_sample


def test_load_aardvark_observation_sample() -> None:
    sample = load_aardvark_observation_sample()

    assert sample.path.name == "sample_data_final.pkl"
    assert "assimilation" in sample.payload
    assert "forecast" in sample.payload
    assert "downscaling" in sample.payload
    assert "hirs_current" in sample.payload["assimilation"]
