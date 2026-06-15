"""Trace and verify the MLM pipeline step-by-step (used by notebook + CLI)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "indic-modernBERT"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

from config import load_modernbert_arch_config
from model.factory import build_modernbert_config
from model.modernbert.attention import IMPL_USE_FLASH2
from model.modernbert.model import FlexBertForMaskedLM
from pretrain.gpu_batch import log_device_summary, move_batch_to_device, resolve_device, training_autocast


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str


def _sample_hindi_texts() -> list[str]:
    return [
        "भारत एक विशाल देश है और हिंदी इसकी प्रमुख भाषाओं में से एक है।",
        "मशीन लर्निंग मॉडल को अच्छे डेटा पर प्रशिक्षित करना ज़रूरी है।",
    ]


def step_gpu() -> StepResult:
    device = resolve_device()
    detail = log_device_summary(device)
    fa2 = f"flash_attn={IMPL_USE_FLASH2}"
    if device.type == "cuda":
        t = torch.randn(4, 4, device=device)
        assert t.is_cuda
        detail += f" | tensor_device={t.device} bf16_supported={torch.cuda.is_bf16_supported()}"
    return StepResult("gpu", True, f"{detail}; {fa2}")


def step_tokenize(
    tokenizer_dir: Path, max_seq_len: int, device: torch.device
) -> tuple[StepResult, dict[str, torch.Tensor]]:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.3)
    encoded = tokenizer(
        _sample_hindi_texts(),
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    examples = [{k: encoded[k][i] for k in encoded} for i in range(len(_sample_hindi_texts()))]
    batch = collator(examples)
    batch = move_batch_to_device(batch, device)
    labels = batch["labels"]
    masked = int((labels != -100).sum())
    pad = int((batch["attention_mask"] == 0).sum())
    detail = (
        f"batch shape {tuple(batch['input_ids'].shape)} | "
        f"mlm_masked={masked} pad_tokens={pad} | "
        f"input_ids.device={batch['input_ids'].device}"
    )
    return StepResult("tokenize+mlm_mask", masked > 0 and pad > 0, detail), batch


def step_model_config(
    arch_path: Path, *, vocab_size: int | None = None
) -> tuple[StepResult, FlexBertForMaskedLM, Any]:
    arch = load_modernbert_arch_config(arch_path)
    if vocab_size is not None and arch.vocab_size != vocab_size:
        arch = arch.model_copy(update={"vocab_size": vocab_size})
    config = build_modernbert_config(model_config=arch)
    device = resolve_device()
    model = FlexBertForMaskedLM(config).to(device)
    layer0 = model.bert.encoder.layers[0].attn
    use_fa2 = getattr(layer0, "use_fa2", False)
    window = getattr(layer0, "sliding_window", None)
    detail = (
        f"layers={arch.num_hidden_layers} padding={arch.padding} attn={arch.attention_layer} "
        f"use_fa2={use_fa2} sliding_window={window} class={layer0.__class__.__name__}"
    )
    return StepResult("model_config", True, detail), model, arch


def step_forward(
    model: FlexBertForMaskedLM,
    batch: dict[str, torch.Tensor],
    *,
    microbatch_size: int | None = None,
) -> StepResult:
    model.eval()
    if microbatch_size is not None and batch["input_ids"].shape[0] > microbatch_size:
        batch = {key: value[:microbatch_size] for key, value in batch.items()}
    labels = batch["labels"]
    with torch.no_grad(), training_autocast(batch["input_ids"].device):
        out = model(**batch)
    masked_n = int((labels != -100).sum())
    logits_shape = tuple(out.logits.shape)
    ok = out.loss is not None and out.logits.is_cuda == (batch["input_ids"].device.type == "cuda")
    if model.config.masked_prediction:
        ok = ok and logits_shape == (masked_n, model.config.vocab_size)
    detail = f"loss={float(out.loss):.4f} logits={logits_shape} loss_device={out.loss.device}"
    return StepResult("forward+logits", ok, detail)


def step_backward(
    model: FlexBertForMaskedLM,
    batch: dict[str, torch.Tensor],
    *,
    microbatch_size: int | None = None,
) -> StepResult:
    model.train()
    model.zero_grad(set_to_none=True)
    if microbatch_size is not None and batch["input_ids"].shape[0] > microbatch_size:
        batch = {key: value[:microbatch_size] for key, value in batch.items()}
    with training_autocast(batch["input_ids"].device):
        loss = model(**batch).loss
    assert loss is not None
    loss.backward()
    norms = [float(p.grad.norm()) for p in model.parameters() if p.grad is not None]
    ok = len(norms) > 0 and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    grad_max = f"{max(norms):.3e}" if norms else "N/A"
    detail = f"loss={float(loss):.4f} params_with_grad={len(norms)} grad_max={grad_max}"
    return StepResult("backward", ok, detail)


def run_pipeline_trace(
    *,
    arch_config: Path = Path("configs/model/modernbert_base.yaml"),
    tokenizer_path: Path = Path("artifacts/tokenizer/bpe_vs50368"),
    max_seq_len: int = 1024,
    microbatch_size: int | None = 16,
) -> list[StepResult]:
    results: list[StepResult] = []
    results.append(step_gpu())
    device = resolve_device()

    tok_path = tokenizer_path
    if not (tok_path / "tokenizer.json").is_file() and not (tok_path / "tokenizer_config.json").is_file():
        if tok_path.is_file() and tok_path.name == "tokenizer.json":
            tok_path = tok_path.parent
    if not (tok_path / "tokenizer.json").is_file():
        results.append(
            StepResult(
                "tokenize+mlm_mask",
                False,
                f"skip — tokenizer not found at {tok_path}",
            )
        )
        return results

    tok_result, batch = step_tokenize(tok_path, max_seq_len, device)
    results.append(tok_result)
    if not tok_result.ok:
        return results

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_path))
    cfg_result, model, _arch = step_model_config(arch_config, vocab_size=tokenizer.vocab_size)
    results.append(cfg_result)

    results.append(step_forward(model, batch, microbatch_size=microbatch_size))
    results.append(step_backward(model, batch, microbatch_size=microbatch_size))
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch-config", type=Path, default=Path("configs/model/modernbert_base.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/bpe_vs50368"))
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--microbatch-size", type=int, default=16)
    args = parser.parse_args()

    print("Pipeline trace")
    print("=" * 60)
    for step in run_pipeline_trace(
        arch_config=args.arch_config,
        tokenizer_path=args.tokenizer,
        max_seq_len=args.max_seq_len,
        microbatch_size=args.microbatch_size,
    ):
        status = "OK" if step.ok else "FAIL"
        print(f"[{status}] {step.name}: {step.detail}")
        if not step.ok:
            raise SystemExit(1)
    print("=" * 60)
    print("All steps passed.")


if __name__ == "__main__":
    main()
