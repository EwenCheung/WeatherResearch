from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from encoder_training.devices import DevicePreference


@dataclass(frozen=True)
class PipelineAConfig:
    name: str
    device_preference: DevicePreference
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    frequency: str
    hf_observation_root: Path
    weatherbench_era5: str
    weatherbench_climatology: str
    aardvark_reference_data: Path
    output_dir: Path
    in_channels: int
    out_channels: int
    int_channels: int
    resolution: int
    decoder: str
    mode: str
    two_frames: bool
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    gradient_clip_norm: float | None


def load_pipeline_a_config(path: str | Path) -> PipelineAConfig:
    """Load a Pipeline A TOML config."""
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    return PipelineAConfig(
        name=raw["pipeline"]["name"],
        device_preference=raw["device"]["preference"],
        train_start=raw["dates"]["train_start"],
        train_end=raw["dates"]["train_end"],
        val_start=raw["dates"]["val_start"],
        val_end=raw["dates"]["val_end"],
        frequency=raw["dates"]["frequency"],
        hf_observation_root=Path(raw["paths"]["hf_observation_root"]),
        weatherbench_era5=raw["paths"]["weatherbench_era5"],
        weatherbench_climatology=raw["paths"]["weatherbench_climatology"],
        aardvark_reference_data=Path(raw["paths"]["aardvark_reference_data"]),
        output_dir=Path(raw["paths"]["output_dir"]),
        in_channels=raw["model"]["in_channels"],
        out_channels=raw["model"]["out_channels"],
        int_channels=raw["model"]["int_channels"],
        resolution=raw["model"]["resolution"],
        decoder=raw["model"]["decoder"],
        mode=raw["model"]["mode"],
        two_frames=raw["model"]["two_frames"],
        batch_size=raw["training"]["batch_size"],
        epochs=raw["training"]["epochs"],
        learning_rate=raw["training"]["learning_rate"],
        weight_decay=raw["training"]["weight_decay"],
        num_workers=raw["training"]["num_workers"],
        gradient_clip_norm=raw["training"].get("gradient_clip_norm"),
    )

