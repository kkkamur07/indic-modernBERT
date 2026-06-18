"""Token classification supervised evaluation adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from evals.config import EvalSuiteConfig, SupervisedDefaultsConfig
from evals.registry import TaskSpec
from evals.tasks.common import resolve_max_seq_length, select_rows, trainer_processing_kwargs, training_args


def run_token_classification(
    cfg: EvalSuiteConfig,
    task_cfg: SupervisedDefaultsConfig,
    spec: TaskSpec,
    raw_dataset: Any,
    train_split: str | None,
    eval_split: str,
    tokenizer: PreTrainedTokenizerBase,
    task_dir: Path,
) -> dict[str, float]:
    label_list = raw_dataset[eval_split].features[spec.label_column].feature.names
    max_seq_length = resolve_max_seq_length(cfg, task_cfg)

    def tokenize_and_align_labels(examples: dict[str, list[Any]]) -> dict[str, Any]:
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_seq_length,
        )
        labels = []
        for row_idx, label in enumerate(examples[spec.label_column]):
            word_ids = tokenized.word_ids(batch_index=row_idx)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    # label_all_tokens=True matches IndicBERT reference (ner.py default)
                    # so F1 is comparable with published Naamapadam numbers.
                    label_ids.append(label[word_idx] if spec.label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized["labels"] = labels
        return tokenized

    train_dataset = None
    if task_cfg.do_train and train_split is not None:
        train_dataset = select_rows(raw_dataset[train_split], task_cfg.max_train_samples).map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_dataset[train_split].column_names,
        )

    eval_dataset = select_rows(raw_dataset[eval_split], task_cfg.max_eval_samples).map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_dataset[eval_split].column_names,
    )

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        import evaluate

        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_list[p] for p, label in zip(prediction, label_row) if label != -100]
            for prediction, label_row in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[label] for _p, label in zip(prediction, label_row) if label != -100]
            for prediction, label_row in zip(predictions, labels)
        ]
        results = evaluate.load("seqeval").compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": float(results["overall_precision"]),
            "recall": float(results["overall_recall"]),
            "f1": float(results["overall_f1"]),
            "accuracy": float(results["overall_accuracy"]),
        }

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model.model_name_or_path,
        num_labels=len(label_list),
        trust_remote_code=cfg.model.trust_remote_code,
        ignore_mismatched_sizes=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args(task_cfg, task_dir, do_train=train_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        **trainer_processing_kwargs(tokenizer),
    )
    if train_dataset is not None:
        trainer.train()
    return trainer.evaluate(eval_dataset=eval_dataset)
