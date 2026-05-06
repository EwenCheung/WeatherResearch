"""Compatibility helpers for using the Aardvark reference repo locally."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from inspect import signature
from typing import Any, Iterator

import torch

from weather_research.project_paths import AARDVARK_CODE_DIR

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

from models import ConvCNPWeather as _ConvCNPWeather  # type: ignore  # noqa: E402

ConvCNPWeather: Any = _ConvCNPWeather


@contextmanager
def redirect_cuda_to_device(device: str) -> Iterator[None]:
    """Redirect reference-code `.cuda()` calls to the requested torch device."""
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
