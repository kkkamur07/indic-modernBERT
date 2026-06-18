"""Sequence classification supervised evaluation adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from transformers import AutoModelForSequenceClassification, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from evals.config import EvalSuiteConfig, SupervisedDefaultsConfig
from evals.registry import TaskSpec
from evals.tasks.common import (
    classification_metrics,
    first_present,
    resolve_max_seq_length,
    select_rows,
    trainer_processing_kwargs,
    training_args,
)


def run_sequence_classification(
    cfg: EvalSuiteConfig,
    task_cfg: SupervisedDefaultsConfig,
    spec: TaskSpec,
    raw_dataset: Any,
    train_split: str | None,
    eval_split: str,
    tokenizer: PreTrainedTokenizerBase,
    task_dir: Path,
) -> dict[str, float]:
    max_seq_length = resolve_max_seq_length(cfg, task_cfg)

    def normalize(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        text_col = first_present(batch, spec.text_columns)
        texts = [str(value) for value in batch[text_col]]
        label_col = spec.label_column
        if label_col in batch:
            raw = batch[label_col]
            labels = [1 if str(v).lower() == "positive" else 0 if str(v).lower() == "negative" else int(v) for v in raw]
        elif "stars" in batch:
            labels = [1 if int(value) > 3 else 0 for value in batch["stars"]]
        else:
            raw = batch.get("label", [])
            labels = [int(v) for v in raw]
        return {"text": texts, "labels": labels}

    def tokenize_text(batch: dict[str, list[Any]]) -> dict[str, Any]:
        tokenized = tokenizer(batch["text"], truncation=True, max_length=max_seq_length)
        tokenized["labels"] = batch["labels"]
        return tokenized

    train_dataset = None
    if task_cfg.do_train and train_split is not None:
        train_dataset = select_rows(raw_dataset[train_split], task_cfg.max_train_samples).map(normalize, batched=True)
        train_dataset = train_dataset.filter(lambda row: row["labels"] in (0, 1))
        train_dataset = train_dataset.map(
            tokenize_text,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
    else:
        logger.warning(
            "Task '{}' has no train split — running zero-shot with a randomly-initialised head. "
            "Metric reflects representation quality only, not fine-tuned accuracy.",
            spec.name,
        )

    eval_raw = select_rows(raw_dataset[eval_split], task_cfg.max_eval_samples).map(normalize, batched=True)
    eval_raw = eval_raw.filter(lambda row: row["labels"] in (0, 1))
    eval_dataset = eval_raw.map(
        tokenize_text,
        batched=True,
        remove_columns=eval_raw.column_names,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name_or_path,
        num_labels=spec.num_labels or 2,
        trust_remote_code=cfg.model.trust_remote_code,
        ignore_mismatched_sizes=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args(task_cfg, task_dir, do_train=train_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=classification_metrics,
        **trainer_processing_kwargs(tokenizer),
    )
    if train_dataset is not None:
        trainer.train()
    return trainer.evaluate(eval_dataset=eval_dataset)
