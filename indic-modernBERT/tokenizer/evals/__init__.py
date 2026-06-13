"""Tokenizer evaluation entry points."""

from .intrinsic import main as run_intrinsic
from .parity import main as run_parity

__all__ = ["run_intrinsic", "run_parity"]
