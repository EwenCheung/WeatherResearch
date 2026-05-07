"""Generate Aardvark IC-like physical states from observation samples."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import xarray as xr

from weather_research.aardvark_compat import (
    ConvCNPWeather,
    move_tensors_to_device,
    redirect_cuda_to_device,
)
from weather_research.aardvark_observations import load_aardvark_observation_sample
from weather_research.project_paths import (
    AARDVARK_DATA_DIR,
    AARDVARK_TRAINED_MODELS_DIR,
)
from weather_research.weather_state_schema import (
    AARDVARK_STATE_SHAPE,
    AardvarkInitialCondition,
)
from weather_research.weather_state_xarray import initial_conditions_to_dataset

DevicePreference = Literal["auto", "cuda", "cpu"]
type SampleTime = tuple[Path | str, np.datetime64 | str]


def choose_compute_device(preference: DevicePreference = "auto") -> str:
    """Choose CUDA when available, otherwise CPU, for Aardvark IC generation."""
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is false."
            )
        return "cuda"
    if preference == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_aardvark_encoder(
    encoder_dir: Path | str = AARDVARK_TRAINED_MODELS_DIR / "encoder",
    *,
    device: str,
) -> torch.nn.Module:
    """Load Aardvark's pretrained observation encoder on the selected device."""
    encoder_path = Path(encoder_dir)
    config_path = encoder_path / "config.pkl"
    losses_path = encoder_path / "losses_0.npy"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing Aardvark encoder config: {config_path}")
    if not losses_path.exists():
        raise FileNotFoundError(f"Missing Aardvark encoder losses: {losses_path}")

    with config_path.open("rb") as handle:
        config = pickle.load(handle)

    with redirect_cuda_to_device(device):
        model = ConvCNPWeather(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            int_channels=config["int_channels"],
            device=device,
            res=config["res"],
            data_path=str(AARDVARK_DATA_DIR) + "/",
            gnp=bool(0),
            decoder=config["decoder"],
            mode=config["mode"],
            film=bool(0),
        )

    best_epoch = int(np.argmin(np.load(losses_path)))
    checkpoint_path = encoder_path / f"epoch_{best_epoch}"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing Aardvark encoder checkpoint: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint["model_state_dict"]
    state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


def _normalization_tensor(name: str, device: str) -> torch.Tensor:
    values = np.load(AARDVARK_DATA_DIR / "norm_factors" / name)
    return torch.Tensor(values).float().to(device).reshape(1, 1, 1, -1)


def unnormalize_encoder_output(
    encoded_state: torch.Tensor,
    *,
    device: str,
) -> torch.Tensor:
    """Convert normalized encoder output into Aardvark's physical IC-like state."""
    means = _normalization_tensor("mean_4u_1.npy", device)
    stds = _normalization_tensor("std_4u_1.npy", device)
    return encoded_state * stds + means


def generate_aardvark_initial_condition(
    sample_payload: dict[str, Any],
    *,
    encoder: torch.nn.Module | None = None,
    device: str | None = None,
) -> AardvarkInitialCondition:
    """Generate a 24-variable IC-like state from a loaded observation payload."""
    selected_device = device or choose_compute_device()
    model = encoder or load_aardvark_encoder(device=selected_device)
    task = cast(dict[str, Any], move_tensors_to_device(sample_payload, selected_device))

    with torch.no_grad(), redirect_cuda_to_device(selected_device):
        encoded_state = model(task["assimilation"], film_index=None)
        physical_state = unnormalize_encoder_output(
            encoded_state,
            device=selected_device,
        )

    values = physical_state.detach().cpu().numpy()
    if values.shape[0] != 1:
        raise ValueError(f"Expected batch size 1 for sample IC, got {values.shape[0]}")

    ic_values = values[0].astype(np.float32, copy=False)
    if ic_values.shape != AARDVARK_STATE_SHAPE:
        raise ValueError(
            f"Expected IC shape {AARDVARK_STATE_SHAPE}, got {ic_values.shape}"
        )

    return AardvarkInitialCondition(values=ic_values)


def generate_aardvark_initial_condition_from_file(
    sample_path: Path | str,
    *,
    device_preference: DevicePreference = "auto",
) -> AardvarkInitialCondition:
    """Load the Aardvark sample file and generate its IC-like physical state."""
    sample = load_aardvark_observation_sample(sample_path)
    device = choose_compute_device(device_preference)
    ic = generate_aardvark_initial_condition(sample.payload, device=device)
    return AardvarkInitialCondition(
        values=ic.values,
        variable_names=ic.variable_names,
        source_path=sample.path,
    )


def generate_aardvark_initial_condition_dataset_from_files(
    sample_times: Sequence[SampleTime],
    *,
    device_preference: DevicePreference = "auto",
    encoder: torch.nn.Module | None = None,
) -> xr.Dataset:
    """Generate a multi-time IC-like dataset from model-ready observation files.

    Each input sample file represents one assimilation/current time. To build a
    real multi-time IC dataset, pass one sample file and one timestamp per time.
    """
    if len(sample_times) == 0:
        raise ValueError("Expected at least one sample/time pair.")

    device = choose_compute_device(device_preference)
    model = encoder or load_aardvark_encoder(device=device)
    initial_conditions: list[AardvarkInitialCondition] = []
    times: list[np.datetime64 | str] = []

    for sample_path, time in sample_times:
        sample = load_aardvark_observation_sample(sample_path)
        ic = generate_aardvark_initial_condition(
            sample.payload,
            encoder=model,
            device=device,
        )
        initial_conditions.append(
            AardvarkInitialCondition(
                values=ic.values,
                variable_names=ic.variable_names,
                source_path=sample.path,
            )
        )
        times.append(time)

    return initial_conditions_to_dataset(initial_conditions, times=times)
