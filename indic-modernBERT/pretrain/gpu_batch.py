"""Device placement and mixed-precision helpers for pretrain loops."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

import torch


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def add_position_ids(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """RoPE / unpadded models need position_ids derived from attention_mask."""
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        return batch
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    batch["position_ids"] = position_ids
    return batch


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensors in a batch to device; preserve lists (e.g. packed ``cu_seqlens``)."""
    non_blocking = device.type == "cuda"
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=non_blocking)
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            moved[key] = [tensor.to(device, non_blocking=non_blocking) for tensor in value]
        else:
            moved[key] = value
    if isinstance(moved.get("attention_mask"), torch.Tensor):
        return add_position_ids(moved)  # type: ignore[arg-type]
    return moved


@contextmanager
def training_autocast(device: torch.device) -> Iterator[None]:
    """Match upstream ModernBERT pretrain: amp_bf16 on GPU."""
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        with nullcontext():
            yield


def log_device_summary(device: torch.device) -> str:
    if device.type != "cuda":
        return "device=cpu (no CUDA)"
    idx = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    return f"device=cuda:{idx} name={name} cc={capability} mem={mem_gb:.1f}GB bf16={torch.cuda.is_bf16_supported()}"
