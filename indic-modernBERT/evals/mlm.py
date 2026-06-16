"""HF masked-language-model holdout evaluation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer

from evals.config import EvalSuiteConfig
from evals.runtime import choose_device, set_eval_seed
from pretrain.evals.mlm import MlmEvalMetrics, masked_accuracy
from pretrain.parquet_mlm import ListMLMDataset, MLMCollator, load_eval_texts


@torch.no_grad()
def _evaluate_hf_mlm(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None,
) -> MlmEvalMetrics:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    tokens = 0

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("HF masked-LM model did not return loss for labelled batch")
        total_loss += float(loss.detach().cpu())
        total_acc += masked_accuracy(outputs.logits.detach().cpu(), batch["labels"].detach().cpu())
        tokens += int((batch["labels"] != -100).sum().detach().cpu())
        steps += 1
        if max_batches is not None and steps >= max_batches:
            break

    if steps == 0:
        return MlmEvalMetrics(loss=float("inf"), masked_accuracy=0.0, steps=0, tokens=0)

    return MlmEvalMetrics(
        loss=total_loss / steps,
        masked_accuracy=total_acc / steps,
        steps=steps,
        tokens=tokens,
    )


def run_mlm_eval(cfg: EvalSuiteConfig, output_dir: Path) -> dict[str, object]:
    mlm_cfg = cfg.mlm
    set_eval_seed(mlm_cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_source,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if tokenizer.mask_token is None:
        raise ValueError(f"Tokenizer at {cfg.model.tokenizer_source} has no mask token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.mask_token

    texts = load_eval_texts(
        mlm_cfg.data_root,
        mlm_cfg.text_column,
        max_rows=mlm_cfg.max_samples if mlm_cfg.max_samples is not None else 1_000_000_000,
    )
    dataset = ListMLMDataset(texts)
    dataloader = DataLoader(
        dataset,
        batch_size=mlm_cfg.batch_size,
        shuffle=False,
        num_workers=mlm_cfg.num_workers,
        collate_fn=MLMCollator(
            tokenizer,
            max_seq_len=mlm_cfg.max_seq_length,
            mlm_probability=mlm_cfg.mlm_probability,
        ),
    )

    device = choose_device(cfg.device)
    model = AutoModelForMaskedLM.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
    ).to(device)
    metrics = _evaluate_hf_mlm(model, dataloader, device, max_batches=mlm_cfg.max_batches)
    result = {
        "name": "mlm_holdout",
        "type": "mlm",
        "status": "completed",
        "metrics": asdict(metrics),
        "config": {
            "data_root": str(mlm_cfg.data_root),
            "max_seq_length": mlm_cfg.max_seq_length,
            "mlm_probability": mlm_cfg.mlm_probability,
            "batch_size": mlm_cfg.batch_size,
            "max_samples": mlm_cfg.max_samples,
            "max_batches": mlm_cfg.max_batches,
        },
    }
    (output_dir / "mlm_metrics.json").write_text(_json_dumps(result), encoding="utf-8")
    return result


def _json_dumps(payload: object) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
