"""ModernBERT-style inference length sweep."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from evals.config import EvalSuiteConfig
from evals.runtime import active_context_length, choose_device, set_eval_seed, should_use_bf16


def run_efficiency_sweep(cfg: EvalSuiteConfig, output_dir: Path) -> dict[str, Any]:
    eff_cfg = cfg.efficiency
    set_eval_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_source,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.cls_token

    model_cls = AutoModelForMaskedLM if eff_cfg.use_mlm_head else AutoModel
    device = choose_device(cfg.device)
    model = model_cls.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
    ).to(device)
    model.eval()
    num_parameters = _num_parameters(model)
    num_parameters_m = num_parameters / 1_000_000
    use_bf16_autocast = should_use_bf16(eff_cfg.use_bf16_autocast, device)
    sequence_lengths = eff_cfg.sequence_lengths or [active_context_length(cfg.model)]

    rows = []
    for seq_len in sequence_lengths:
        batch = _build_batch(tokenizer, eff_cfg.sample_texts, eff_cfg.batch_size, seq_len)
        batch = {key: value.to(device) for key, value in batch.items()}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(eff_cfg.warmup_steps):
            _forward(model, batch, device, use_bf16_autocast=use_bf16_autocast)

        latencies = []
        power_readings = []
        for _ in range(eff_cfg.measured_steps):
            start = time.perf_counter()
            _forward(model, batch, device, use_bf16_autocast=use_bf16_autocast)
            latencies.append(time.perf_counter() - start)
            if eff_cfg.measure_power:
                reading = _gpu_power_watts()
                if reading is not None:
                    power_readings.append(reading)

        mean_latency = statistics.fmean(latencies)
        std_latency = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
        examples = eff_cfg.batch_size
        tokens = eff_cfg.batch_size * seq_len
        tokens_per_second = tokens / mean_latency if mean_latency else float("inf")
        rows.append(
            {
                "sequence_length": seq_len,
                "batch_size": eff_cfg.batch_size,
                "num_parameters": num_parameters,
                "num_parameters_m": num_parameters_m,
                "latency_mean_s": mean_latency,
                "latency_std_s": std_latency,
                "examples_per_second": examples / mean_latency if mean_latency else float("inf"),
                "tokens_per_second": tokens_per_second,
                "tokens_per_second_per_million_params": tokens_per_second / num_parameters_m if num_parameters_m else None,
                "peak_cuda_memory_allocated_mb": _peak_cuda_memory_allocated_mb(device),
                "peak_cuda_memory_reserved_mb": _peak_cuda_memory_reserved_mb(device),
                "avg_power_watts": statistics.fmean(power_readings) if power_readings else None,
                "max_power_watts": max(power_readings) if power_readings else None,
                "device": str(device),
            }
        )

    result = {
        "name": "efficiency_sweep",
        "type": "efficiency",
        "status": "completed",
        "metrics": {"lengths": rows},
        "config": {
            "sequence_lengths": sequence_lengths,
            "configured_sequence_lengths": eff_cfg.sequence_lengths,
            "batch_size": eff_cfg.batch_size,
            "warmup_steps": eff_cfg.warmup_steps,
            "measured_steps": eff_cfg.measured_steps,
            "use_mlm_head": eff_cfg.use_mlm_head,
            "use_bf16_autocast": use_bf16_autocast,
            "measure_power": eff_cfg.measure_power,
        },
    }
    (output_dir / "efficiency_metrics.json").write_text(_json_dumps(result), encoding="utf-8")
    return result


def _build_batch(tokenizer: Any, sample_texts: list[str], batch_size: int, seq_len: int) -> dict[str, torch.Tensor]:
    text = " ".join(sample_texts)
    texts = [text for _ in range(batch_size)]
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )


@torch.no_grad()
def _forward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    use_bf16_autocast: bool,
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if device.type == "cuda" and use_bf16_autocast:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(**batch)
    else:
        model(**batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_cuda_memory_allocated_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))


def _peak_cuda_memory_reserved_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_reserved(device) / (1024 * 1024))


def _num_parameters(model: torch.nn.Module) -> int:
    getter = getattr(model, "get_number_parameters", None)
    if callable(getter):
        return int(getter())
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _gpu_power_watts() -> float | None:
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()  # type: ignore[name-defined]
        except Exception:
            pass


def _json_dumps(payload: object) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
