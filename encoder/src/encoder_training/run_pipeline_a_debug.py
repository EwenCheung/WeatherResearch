from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from encoder_training.config import load_pipeline_a_config
from encoder_training.datasets import AmsuAscatPipelineADataset, daily_times
from encoder_training.devices import choose_device
from encoder_training.losses import WeightedRmseLoss
from encoder_training.models import build_aardvark_pipeline_a_encoder
from encoder_training.targets import load_target_normalization, open_weatherbench_era5
from encoder_training.trainer import evaluate, save_checkpoint, train_one_epoch
from weather_research.aardvark_compat import redirect_cuda_to_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="encoder/configs/pipeline_a_debug.toml",
        help="Path to Pipeline A debug TOML config.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=2,
        help="Limit training samples for local smoke testing. Use 0 for all.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=1,
        help="Limit validation samples for local smoke testing. Use 0 for all.",
    )
    args = parser.parse_args()

    config = load_pipeline_a_config(args.config)
    device = choose_device(config.device_preference)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    norm_dir = config.aardvark_reference_data / "norm_factors"
    means, stds = load_target_normalization(norm_dir)
    era5 = open_weatherbench_era5(config.weatherbench_era5)
    climatology = open_weatherbench_era5(config.weatherbench_climatology)

    train_times = _limit_times(
        daily_times(config.train_start, config.train_end),
        args.max_train_samples,
    )
    val_times = _limit_times(
        daily_times(config.val_start, config.val_end),
        args.max_val_samples,
    )

    train_dataset = AmsuAscatPipelineADataset(
        observation_root=config.hf_observation_root,
        era5=era5,
        climatology=climatology,
        times=train_times,
        means=means,
        stds=stds,
    )
    val_dataset = AmsuAscatPipelineADataset(
        observation_root=config.hf_observation_root,
        era5=era5,
        climatology=climatology,
        times=val_times,
        means=means,
        stds=stds,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_single_item_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_single_item_collate,
    )

    model = build_aardvark_pipeline_a_encoder(
        device=device,
        data_path=config.aardvark_reference_data,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        int_channels=config.int_channels,
        resolution=config.resolution,
        decoder=config.decoder,
        mode=config.mode,
        two_frames=config.two_frames,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = WeightedRmseLoss()

    with redirect_cuda_to_device(str(device)):
        before = evaluate(model, val_loader, loss_fn, device=device)
        train = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device=device,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        after = evaluate(model, val_loader, loss_fn, device=device)

    save_checkpoint(
        output_dir / "checkpoint_smoke.pt",
        model=model,
        optimizer=optimizer,
        epoch=0,
        val_loss=after.loss,
    )
    metrics_path = output_dir / "metrics_smoke.txt"
    metrics_path.write_text(
        "\n".join(
            [
                f"device={device}",
                f"train_samples={len(train_dataset)}",
                f"val_samples={len(val_dataset)}",
                "real_observation_sources=amsua,amsub,ascat",
                "placeholder_observation_sources=iasi,hirs,gridsat,hadisd,icoads,igra",
                f"val_loss_before={before.loss:.6f}",
                f"train_loss={train.loss:.6f}",
                f"val_loss_after={after.loss:.6f}",
            ]
        )
        + "\n"
    )
    print(metrics_path.read_text())


def _single_item_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    """Keep Aardvark-style tensors with their existing leading batch dimension."""
    if len(batch) != 1:
        raise ValueError("Pipeline A debug currently expects batch_size=1.")
    return batch[0]


def _limit_times(times: list[object], limit: int) -> list[object]:
    if limit <= 0:
        return times
    return times[:limit]


if __name__ == "__main__":
    main()
