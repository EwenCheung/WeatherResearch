from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import weather_research.aardvark_initial_conditions as initial_conditions_module
from weather_research.aardvark_initial_conditions import (
    choose_compute_device,
    generate_aardvark_initial_condition_dataset_from_files,
    generate_aardvark_initial_condition_from_file,
)
from weather_research.aardvark_observations import DEFAULT_AARDVARK_SAMPLE_PATH
from weather_research.weather_state_schema import (
    AARDVARK_STATE_SHAPE,
    AardvarkInitialCondition,
    AardvarkObservationSample,
)


def test_choose_compute_device_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert choose_compute_device() == "cuda"


def test_choose_compute_device_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert choose_compute_device() == "cpu"


def test_generate_initial_condition_from_file_integration() -> None:
    if os.environ.get("INTEGRATION_TESTS") != "1":
        pytest.skip("Set INTEGRATION_TESTS=1 to run the Aardvark encoder.")

    ic = generate_aardvark_initial_condition_from_file(
        DEFAULT_AARDVARK_SAMPLE_PATH,
        device_preference="auto",
    )

    assert ic.values.shape == (121, 240, 24)
    assert np.isfinite(ic.values).all()


def test_generate_initial_condition_dataset_from_files_reuses_encoder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    sample_a = tmp_path / "sample_a.pkl"
    sample_b = tmp_path / "sample_b.pkl"
    encoder = torch.nn.Identity()
    loaded_paths: list[os.PathLike[str] | str] = []
    generated_values = iter(
        [
            np.zeros(AARDVARK_STATE_SHAPE, dtype=np.float32),
            np.ones(AARDVARK_STATE_SHAPE, dtype=np.float32),
        ]
    )

    monkeypatch.setattr(
        initial_conditions_module,
        "choose_compute_device",
        lambda _: "cpu",
    )
    monkeypatch.setattr(
        initial_conditions_module,
        "load_aardvark_encoder",
        lambda *, device: encoder,
    )

    def load_sample(path: os.PathLike[str] | str) -> AardvarkObservationSample:
        loaded_paths.append(path)
        return AardvarkObservationSample(path=sample_a, payload={"assimilation": {}})

    def generate_ic(*args, **kwargs) -> AardvarkInitialCondition:
        assert kwargs["encoder"] is encoder
        assert kwargs["device"] == "cpu"
        return AardvarkInitialCondition(values=next(generated_values))

    monkeypatch.setattr(
        initial_conditions_module,
        "load_aardvark_observation_sample",
        load_sample,
    )
    monkeypatch.setattr(
        initial_conditions_module,
        "generate_aardvark_initial_condition",
        generate_ic,
    )

    dataset = generate_aardvark_initial_condition_dataset_from_files(
        [
            (sample_a, "2019-01-01T00:00:00"),
            (sample_b, "2019-01-01T06:00:00"),
        ],
        device_preference="auto",
    )

    assert loaded_paths == [sample_a, sample_b]
    assert dataset.sizes["time"] == 2
    assert (
        float(dataset["2m_temperature"].isel(time=1, latitude=0, longitude=0)) == 1.0
    )
