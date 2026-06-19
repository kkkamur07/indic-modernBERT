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
    raw_dataset: Any,
    train_split: str | None,
    eval_split: str,
    tokenizer: PreTrainedTokenizerBase,
    task_dir: Path,
) -> dict[str, float]:
    max_seq_length = resolve_max_seq_length(cfg, task_cfg)

    def preprocess(examples: dict[str, list[Any]]) -> dict[str, Any]:
        ending_names = ["choice1", "choice2"]
        first_sentences = [[premise] * len(ending_names) for premise in examples["premise"]]
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
        features["labels"] = examples[spec.label_column]
        return features

    train_dataset = None
    if task_cfg.do_train and train_split is not None:
        train_dataset = select_rows(raw_dataset[train_split], task_cfg.max_train_samples).map(
            preprocess,
            batched=True,
            remove_columns=raw_dataset[train_split].column_names,
        )
    eval_dataset = select_rows(raw_dataset[eval_split], task_cfg.max_eval_samples).map(
        preprocess,
        batched=True,
        remove_columns=raw_dataset[eval_split].column_names,
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


def _ensure_xlm_roberta_pooler(model: torch.nn.Module) -> None:
    """Work around a Transformers XLM-R multiple-choice head that expects a pooler."""
    if model.__class__.__name__ != "XLMRobertaForMultipleChoice":
        return
        
    encoder = getattr(model, "roberta", None)
    if encoder is None or getattr(encoder, "pooler", None) is not None:
        return

    from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaPooler

    encoder.pooler = XLMRobertaPooler(model.config)
