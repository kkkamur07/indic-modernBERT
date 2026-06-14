"""GPU alignment metrics for ModernBERT embedding / LM-head GEMMs (ablation + optional enforce)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config.schema import HardwareAlignmentConfig, ModernBertArchConfig


@dataclass(frozen=True)
class HardwareAlignmentReport:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    head_dim: int
    tensor_align: int
    tile_m: int
    tile_n: int
    sm_count: int | None
    tensor_hidden_ok: bool
    tensor_intermediate_ok: bool
    tensor_head_ok: bool
    tensor_vocab_ok: bool
    tile_hidden_ok: bool
    tile_vocab_ok: bool
    lm_head_tiles_m: int
    lm_head_tiles_n: int
    wave_tiles_m_remainder: int | None
    wave_tiles_n_remainder: int | None
    all_tensor_ok: bool
    all_tile_ok: bool
    all_wave_ok: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _divisible(value: int, align: int) -> bool:
    return align > 0 and value % align == 0


def compute_hardware_alignment(
    arch: ModernBertArchConfig,
    *,
    hardware: HardwareAlignmentConfig | None = None,
) -> HardwareAlignmentReport:
    from config.schema import HardwareAlignmentConfig as HWConfig

    hw = hardware if hardware is not None else arch.hardware_alignment
    head_dim = arch.hidden_size // arch.num_attention_heads
    tile_m = hw.tile_m
    tile_n = hw.tile_n
    tensor = hw.tensor_align

    lm_tiles_m = arch.vocab_size // tile_m if _divisible(arch.vocab_size, tile_m) else -1
    lm_tiles_n = arch.hidden_size // tile_n if _divisible(arch.hidden_size, tile_n) else -1

    wave_m: int | None = None
    wave_n: int | None = None
    if hw.sm_count is not None and lm_tiles_m > 0:
        wave_m = lm_tiles_m % hw.sm_count

    tensor_hidden_ok = _divisible(arch.hidden_size, tensor)
    tensor_intermediate_ok = _divisible(arch.intermediate_size, tensor)
    tensor_head_ok = _divisible(head_dim, tensor)
    tensor_vocab_ok = _divisible(arch.vocab_size, tensor)
    tile_hidden_ok = _divisible(arch.hidden_size, tile_n)
    tile_vocab_ok = _divisible(arch.vocab_size, tile_m)

    if hw.sm_count is None:
        all_wave: bool | None = None
    elif wave_m is None:
        all_wave = False
    else:
        all_wave = wave_m == 0

    return HardwareAlignmentReport(
        vocab_size=arch.vocab_size,
        hidden_size=arch.hidden_size,
        intermediate_size=arch.intermediate_size,
        head_dim=head_dim,
        tensor_align=tensor,
        tile_m=tile_m,
        tile_n=tile_n,
        sm_count=hw.sm_count,
        tensor_hidden_ok=tensor_hidden_ok,
        tensor_intermediate_ok=tensor_intermediate_ok,
        tensor_head_ok=tensor_head_ok,
        tensor_vocab_ok=tensor_vocab_ok,
        tile_hidden_ok=tile_hidden_ok,
        tile_vocab_ok=tile_vocab_ok,
        lm_head_tiles_m=lm_tiles_m,
        lm_head_tiles_n=lm_tiles_n,
        wave_tiles_m_remainder=wave_m,
        wave_tiles_n_remainder=wave_n,
        all_tensor_ok=all(
            (tensor_hidden_ok, tensor_intermediate_ok, tensor_head_ok, tensor_vocab_ok)
        ),
        all_tile_ok=tile_hidden_ok and tile_vocab_ok,
        all_wave_ok=all_wave,
    )


def validate_hardware_alignment(
    arch: ModernBertArchConfig,
    *,
    hardware: HardwareAlignmentConfig | None = None,
) -> HardwareAlignmentReport:
    report = compute_hardware_alignment(arch, hardware=hardware)
    hw = hardware if hardware is not None else arch.hardware_alignment
    if not hw.enforce:
        return report

    failures: list[str] = []
    if not report.all_tensor_ok:
        failures.append(
            "tensor alignment failed "
            f"(hidden={report.tensor_hidden_ok}, intermediate={report.tensor_intermediate_ok}, "
            f"head={report.tensor_head_ok}, vocab={report.tensor_vocab_ok})"
        )
    if not report.all_tile_ok:
        failures.append(
            f"tile alignment failed (hidden%{hw.tile_n}={report.tile_hidden_ok}, "
            f"vocab%{hw.tile_m}={report.tile_vocab_ok})"
        )
    if hw.sm_count is not None and report.all_wave_ok is False:
        failures.append(
            f"wave alignment failed (vocab-tiles%{hw.sm_count}={report.wave_tiles_m_remainder})"
        )
    if failures:
        raise ValueError("; ".join(failures))
    return report


def iter_vocab_ablation_sizes(center: int, *, step: int = 64, radius: int = 512) -> list[int]:
    low = max(step, center - radius)
    high = center + radius
    return [size for size in range(low, high + 1, step) if size % 64 == 0]


def run_vocab_ablation(
    base: ModernBertArchConfig,
    *,
    center_vocab_size: int | None = None,
    step: int = 64,
    radius: int = 512,
    sm_counts: list[int | None] | None = None,
) -> list[dict[str, Any]]:
    center = center_vocab_size if center_vocab_size is not None else base.vocab_size
    counts: list[int | None] = [None] if sm_counts is None else sm_counts
    rows: list[dict[str, Any]] = []
    for vocab_size in iter_vocab_ablation_sizes(center, step=step, radius=radius):
        candidate = base.model_copy(update={"vocab_size": vocab_size})
        for sm_count in counts:
            hw = candidate.hardware_alignment.model_copy(update={"sm_count": sm_count})
            report = compute_hardware_alignment(candidate, hardware=hw)
            rows.append(report.to_dict())
    return rows


def write_ablation_results(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    label: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"{label}_{stamp}.json"
    payload = {"label": label, "generated_at": stamp, "results": rows}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    latest = output_dir / "latest.json"
    latest.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def detect_cuda_sm_count() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).multi_processor_count
    except ImportError:
        return None
