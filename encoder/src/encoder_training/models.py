from __future__ import annotations

from pathlib import Path

import torch

from weather_research.aardvark_compat import ConvCNPWeather, redirect_cuda_to_device


def build_aardvark_pipeline_a_encoder(
    *,
    device: torch.device,
    data_path: str | Path,
    in_channels: int = 277,
    out_channels: int = 24,
    int_channels: int = 24,
    resolution: int = 1,
    decoder: str = "vit_assimilation",
    mode: str = "assimilation",
    two_frames: bool = False,
) -> torch.nn.Module:
    """Build the Aardvark-style encoder used by Pipeline A."""
    data_root = str(Path(data_path)) + "/"
    with redirect_cuda_to_device(str(device)):
        model = ConvCNPWeather(
            in_channels=in_channels,
            out_channels=out_channels,
            int_channels=int_channels,
            device=str(device),
            res=resolution,
            data_path=data_root,
            gnp=bool(0),
            decoder=decoder,
            mode=mode,
            film=bool(0),
            two_frames=two_frames,
        )
    return model.to(device)

