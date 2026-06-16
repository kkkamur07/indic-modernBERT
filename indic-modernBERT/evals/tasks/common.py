"""Shared helpers for supervised evaluation task adapters."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
from transformers import Trainer, TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from evals.config import EvalSuiteConfig, SupervisedDefaultsConfig
from evals.runtime import should_use_bf16


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
        "bf16": should_use_bf16(task_cfg.bf16),
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


def resolve_max_seq_length(cfg: EvalSuiteConfig, task_cfg: SupervisedDefaultsConfig) -> int:
    if cfg.model.context_mode == "common_128":
        return min(task_cfg.max_seq_length, 128)
    return min(task_cfg.max_seq_length, cfg.model.max_sequence_length)


def classification_metrics(eval_pred: Any) -> dict[str, float]:
    from sklearn.metrics import f1_score

    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_from_preds(preds, labels),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def accuracy_metrics(eval_pred: Any) -> dict[str, float]:
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_from_preds(preds, labels)}


def accuracy_from_preds(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).astype(np.float32).mean().item())


def first_present(batch: dict[str, list[Any]], columns: tuple[str, ...]) -> str:
    for column in columns:
        if column in batch:
            return column
    raise KeyError(f"None of the expected columns are present: {columns}; got {sorted(batch)}")
