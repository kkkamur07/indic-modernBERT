"""MLM eval metrics (masked accuracy, eval loss)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from model.modernbert.model import FlexBertForMaskedLM
from pretrain.gpu_batch import move_batch_to_device, training_autocast


@dataclass
class MlmEvalMetrics:
    loss: float
    masked_accuracy: float
    steps: int
    tokens: int


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.dim() == 2:
        masked_labels = labels[labels != -100]
        if masked_labels.numel() == 0:
            return 0.0
        preds = logits.argmax(dim=-1)
        return float((preds == masked_labels).float().mean())

    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return float((preds[mask] == labels[mask]).float().mean())


@torch.no_grad()
def evaluate_mlm(
    model: FlexBertForMaskedLM,
    dataloader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = None,
    microbatch_size: int | None = None,
) -> MlmEvalMetrics:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    tokens = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        if microbatch_size is not None and batch["input_ids"].shape[0] > microbatch_size:
            batch = {key: value[:microbatch_size] for key, value in batch.items()}
        labels = batch["labels"]
        with training_autocast(device):
            outputs = model(**batch)
        loss = outputs.loss
        assert loss is not None
        total_loss += float(loss)
        total_acc += masked_accuracy(outputs.logits, labels)
        steps += 1
        tokens += int((labels != -100).sum())
        if max_batches is not None and steps >= max_batches:
            break

    if steps == 0:
        return MlmEvalMetrics(loss=math.inf, masked_accuracy=0.0, steps=0, tokens=0)

    return MlmEvalMetrics(
        loss=total_loss / steps,
        masked_accuracy=total_acc / steps,
        steps=steps,
        tokens=tokens,
    )
