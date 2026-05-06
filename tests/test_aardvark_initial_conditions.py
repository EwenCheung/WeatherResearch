from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from weather_research.aardvark_initial_conditions import (
    choose_compute_device,
    generate_aardvark_initial_condition_from_file,
)
from weather_research.aardvark_observations import DEFAULT_AARDVARK_SAMPLE_PATH


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
