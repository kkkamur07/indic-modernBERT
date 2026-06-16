"""Shared runtime helpers for evaluation commands."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.log_helpers import slug


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_output_dir(base: Path, model_name_or_path: str) -> Path:
    name = Path(model_name_or_path).name if "/" in model_name_or_path else model_name_or_path
    if not name:
        name = model_name_or_path
    return base / slug(name)


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}.{key}": value for key, value in metrics.items()}
