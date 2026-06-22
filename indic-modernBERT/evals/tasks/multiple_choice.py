"""Multiple-choice supervised evaluation adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMultipleChoice, Trainer
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

from evals.config import EvalSuiteConfig, SupervisedDefaultsConfig
from evals.registry import TaskSpec
from evals.tasks.common import accuracy_metrics, resolve_max_seq_length, select_rows, trainer_processing_kwargs, training_args


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: bool | str | PaddingStrategy = True
    max_length: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        label_name = "label" if "label" in features[0] else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened = [
            {key: value[choice_idx] for key, value in feature.items()}
            for feature in features
            for choice_idx in range(num_choices)
        ]
        batch = self.tokenizer.pad(
            flattened,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.view(batch_size, num_choices, -1) for key, value in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def run_multiple_choice(
    cfg: EvalSuiteConfig,
    task_cfg: SupervisedDefaultsConfig,
    spec: TaskSpec,
    train_raw_dataset: Any,
    train_split: str | None,
    eval_raw_dataset: Any,
    eval_split: str,
    tokenizer: PreTrainedTokenizerBase,
    task_dir: Path,
) -> dict[str, float]:
    max_seq_length = resolve_max_seq_length(cfg, task_cfg)

    def preprocess(examples: dict[str, list[Any]]) -> dict[str, Any]:
        ending_names = _choice_columns(examples)
        context_col = "premise" if "premise" in examples else "context"
        first_sentences = [[premise] * len(ending_names) for premise in examples[context_col]]
        second_sentences = [
            [f"{question} {examples[end][idx]}" for end in ending_names]
            for idx, question in enumerate(examples["question"])
        ]
        flat_first = sum(first_sentences, [])
        flat_second = sum(second_sentences, [])
        tokenized = tokenizer(
            flat_first,
            flat_second,
            truncation=True,
            max_length=max_seq_length,
        )
        features = {
            key: [value[i : i + len(ending_names)] for i in range(0, len(value), len(ending_names))]
            for key, value in tokenized.items()
        }
        features["labels"] = [_label_to_id(label, num_choices=len(ending_names)) for label in examples[spec.label_column]]
        return features

    train_dataset = None
    if task_cfg.do_train and train_split is not None:
        train_dataset = select_rows(train_raw_dataset[train_split], task_cfg.max_train_samples).map(
            preprocess,
            batched=True,
            remove_columns=train_raw_dataset[train_split].column_names,
        )
    eval_dataset = select_rows(eval_raw_dataset[eval_split], task_cfg.max_eval_samples).map(
        preprocess,
        batched=True,
        remove_columns=eval_raw_dataset[eval_split].column_names,
    )

    model = AutoModelForMultipleChoice.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
        ignore_mismatched_sizes=True,
    )
    _ensure_xlm_roberta_pooler(model)
    trainer = Trainer(
        model=model,
        args=training_args(task_cfg, task_dir, do_train=train_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=accuracy_metrics,
        **trainer_processing_kwargs(tokenizer),
    )
    if train_dataset is not None:
        trainer.train()
    return trainer.evaluate(eval_dataset=eval_dataset)


def _choice_columns(examples: dict[str, list[Any]]) -> list[str]:
    if {"choice1", "choice2"}.issubset(examples):
        return ["choice1", "choice2"]
    if {"answerA", "answerB", "answerC"}.issubset(examples):
        return ["answerA", "answerB", "answerC"]
    raise KeyError(f"Could not infer multiple-choice columns from {sorted(examples)}")


def _label_to_id(label: Any, *, num_choices: int) -> int:
    value = int(label)
    # Social IQa labels are 1-based strings; IndicCOPA labels are already 0/1.
    if num_choices == 3 and value in (1, 2, 3):
        return value - 1
    return value


def _ensure_xlm_roberta_pooler(model: torch.nn.Module) -> None:
    """Work around a Transformers XLM-R multiple-choice head that expects a pooler."""
    if model.__class__.__name__ != "XLMRobertaForMultipleChoice":
        return
        
    encoder = getattr(model, "roberta", None)
    if encoder is None or getattr(encoder, "pooler", None) is not None:
        return

    from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaPooler

    encoder.pooler = XLMRobertaPooler(model.config)
