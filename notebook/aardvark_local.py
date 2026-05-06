"""Local-device compatibility layer for Aardvark Phase 0 notebooks."""

from __future__ import annotations

import pickle
import sys
from contextlib import contextmanager
from inspect import signature
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from weather_research.phase0 import AARDVARK_CODE_DIR, AARDVARK_DATA_DIR

if str(AARDVARK_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(AARDVARK_CODE_DIR))


def _patch_timm_block_drop_arg() -> None:
    """Allow Aardvark's older ViT code to run with newer timm releases."""
    from timm.models import vision_transformer

    if "drop" in signature(vision_transformer.Block.__init__).parameters:
        return

    original_block = vision_transformer.Block

    class DropCompatBlock(original_block):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, drop: float = 0.0, **kwargs: Any) -> None:
            kwargs.setdefault("proj_drop", drop)
            super().__init__(*args, **kwargs)

    vision_transformer.Block = DropCompatBlock  # type: ignore[assignment]


_patch_timm_block_drop_arg()

from e2e_model import ConvCNPWeatherE2E  # type: ignore  # noqa: E402
from misc_downscaling_functionality import ConvCNPWeatherOnToOff  # type: ignore  # noqa: E402
from models import ConvCNPWeather  # type: ignore  # noqa: E402


def load_aardvark_encoder(
    device: str,
    encoder_model_path: str | Path,
) -> torch.nn.Module:
    """Load Aardvark's pretrained observation encoder on a local device."""
    encoder_path = Path(encoder_model_path)
    with open(encoder_path / "config.pkl", "rb") as handle:
        forecast_config = pickle.load(handle)

    with redirect_cuda_to_device(device):
        model = ConvCNPWeather(
            in_channels=forecast_config["in_channels"],
            out_channels=forecast_config["out_channels"],
            int_channels=forecast_config["int_channels"],
            device=device,
            res=forecast_config["res"],
            data_path=str(AARDVARK_DATA_DIR) + "/",
            gnp=bool(0),
            decoder=forecast_config["decoder"],
            mode=forecast_config["mode"],
            film=bool(0),
        )

    best_epoch = np.argmin(np.load(encoder_path / "losses_0.npy"))
    state_dict = torch.load(
        encoder_path / f"epoch_{best_epoch}",
        map_location=device,
        weights_only=False,
    )["model_state_dict"]
    state_dict = {k[7:]: v for k, v in zip(state_dict.keys(), state_dict.values())}
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


@contextmanager
def redirect_cuda_to_device(device: str) -> Iterator[None]:
    """Redirect reference-code `.cuda()` tensor calls to a local device."""
    original_tensor_cuda = torch.Tensor.cuda
    original_module_cuda = torch.nn.Module.cuda

    def tensor_cuda(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        return self.to(device)

    def module_cuda(
        self: torch.nn.Module, *args: object, **kwargs: object
    ) -> torch.nn.Module:
        return self.to(device)

    torch.Tensor.cuda = tensor_cuda  # type: ignore[method-assign]
    torch.nn.Module.cuda = module_cuda  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.Tensor.cuda = original_tensor_cuda  # type: ignore[method-assign]
        torch.nn.Module.cuda = original_module_cuda  # type: ignore[method-assign]


class LocalConvCNPWeatherE2E(ConvCNPWeatherE2E):
    """Aardvark E2E model loader that respects `device` instead of CUDA."""

    def to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.Tensor(arr).float().to(self.device)

    def load_se_model(self, se_model_path: str) -> torch.nn.Module:
        return load_aardvark_encoder(self.device, se_model_path)

    def load_forecast_model(
        self, forecast_model_path: str, lead_time: int
    ) -> torch.nn.Module:
        with open(Path(forecast_model_path) / "config.pkl", "rb") as handle:
            forecast_config = pickle.load(handle)

        with redirect_cuda_to_device(self.device):
            model = ConvCNPWeather(
                in_channels=forecast_config["in_channels"],
                out_channels=forecast_config["out_channels"],
                int_channels=forecast_config["int_channels"],
                device=self.device,
                res=forecast_config["res"],
                data_path=str(AARDVARK_DATA_DIR) + "/",
                gnp=bool(0),
                decoder=forecast_config["decoder"],
                mode=forecast_config["mode"],
                film=False,
            )
        state_dict = torch.load(
            Path(forecast_model_path) / f"forecast_{lead_time}" / "epoch_0",
            map_location=self.device,
            weights_only=False,
        )["model_state_dict"]
        state_dict = {k[7:]: v for k, v in zip(state_dict.keys(), state_dict.values())}
        model.load_state_dict(state_dict)
        return model.to(self.device)

    def load_sf_model(self, sf_model_path: str, lead_time: int) -> torch.nn.Module:
        sf_path = Path(sf_model_path)
        with open(sf_path / "config.pkl", "rb") as handle:
            config = pickle.load(handle)

        with redirect_cuda_to_device(self.device):
            model = ConvCNPWeatherOnToOff(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                int_channels=config["int_channels"],
                device=self.device,
                res=config["res"],
                data_path=str(AARDVARK_DATA_DIR) + "/",
                decoder=config["decoder"],
                mode=config["mode"],
                film=False,
            )

        best_epoch = np.argmin(np.load(sf_path / f"lt_{lead_time}" / "losses_0.npy"))
        full_state_dict = torch.load(
            sf_path / f"lt_{lead_time}" / f"epoch_{best_epoch}",
            map_location=self.device,
            weights_only=False,
        )
        state_dict = full_state_dict["model_state_dict"]
        state_dict = {k[7:]: v for k, v in zip(state_dict.keys(), state_dict.values())}
        model.load_state_dict(state_dict)
        model.eval()
        return model.to(self.device)


def move_tensors_to_device(value: object, device: str) -> object:
    """Recursively move tensors inside Aardvark task dictionaries/lists."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors_to_device(item, device) for item in value)
    return value
