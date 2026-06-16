"""Shared helpers for supervised evaluation task adapters."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
from transformers import Trainer, TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from evals.config import SupervisedDefaultsConfig


def select_rows(dataset: Any, max_samples: int | None) -> Any:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    return dataset.select(range(max_samples))


def training_args(task_cfg: SupervisedDefaultsConfig, output_dir: Path, *, do_train: bool) -> TrainingArguments:
    params: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": task_cfg.learning_rate,
        "per_device_train_batch_size": task_cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": task_cfg.per_device_eval_batch_size,
        "num_train_epochs": task_cfg.num_train_epochs,
        "weight_decay": task_cfg.weight_decay,
        "warmup_ratio": task_cfg.warmup_ratio,
        "fp16": task_cfg.fp16,
        "bf16": task_cfg.bf16,
        "save_total_limit": task_cfg.save_total_limit,
        "report_to": task_cfg.report_to,
        "save_strategy": "no",
        "logging_strategy": "steps" if do_train else "no",
        "logging_steps": 25,
    }
    signature = inspect.signature(TrainingArguments)
    eval_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    params[eval_key] = "no"
    return TrainingArguments(**params)


def trainer_processing_kwargs(tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
    signature = inspect.signature(Trainer)
    if "processing_class" in signature.parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def classification_metrics(eval_pred: Any) -> dict[str, float]:
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    metrics = accuracy_metrics((predictions, labels))
    metrics["macro_f1"] = macro_f1(preds, labels)
    return metrics


def accuracy_metrics(eval_pred: Any) -> dict[str, float]:
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": float((preds == labels).astype(np.float32).mean().item())}


def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for label in sorted(set(labels.tolist()) | set(preds.tolist())):
        tp = int(((preds == label) & (labels == label)).sum())
        fp = int(((preds == label) & (labels != label)).sum())
        fn = int(((preds != label) & (labels == label)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def first_present(batch: dict[str, list[Any]], columns: tuple[str, ...]) -> str:
    for column in columns:
        if column in batch:
            return column
    raise KeyError(f"None of the expected columns are present: {columns}; got {sorted(batch)}")
