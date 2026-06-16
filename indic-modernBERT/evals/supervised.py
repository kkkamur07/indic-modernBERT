"""Supervised Hindi gate adapters."""

from __future__ import annotations

import collections
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from transformers import (
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

from evals.config import EvalSuiteConfig, SupervisedDefaultsConfig
from evals.registry import TaskSpec, get_task_spec
from evals.runtime import set_eval_seed


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


def run_supervised_task(cfg: EvalSuiteConfig, task_name: str, output_dir: Path) -> dict[str, Any]:
    spec = get_task_spec(task_name)
    task_cfg = cfg.task_config(task_name)
    set_eval_seed(cfg.seed)

    try:
        raw_dataset = _load_dataset(spec)
    except Exception as exc:  # pragma: no cover - depends on network/cache.
        return _blocked_result(spec, "dataset_load_failed", exc)

    train_split = _pick_split(raw_dataset, preferred=spec.train_split, fallback=("train",))
    eval_split = _pick_split(raw_dataset, preferred=spec.eval_split, fallback=("validation", "test", "train"))
    if eval_split is None:
        return _blocked_result(spec, "missing_eval_split", ValueError(f"No eval split in {list(raw_dataset)}"))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_source,
        trust_remote_code=cfg.model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.cls_token

    task_dir = output_dir / "supervised" / spec.name
    task_dir.mkdir(parents=True, exist_ok=True)

    try:
        if spec.task_type == "sequence_classification":
            metrics = _run_sequence_classification(cfg, task_cfg, spec, raw_dataset, train_split, eval_split, tokenizer, task_dir)
        elif spec.task_type == "token_classification":
            metrics = _run_token_classification(cfg, task_cfg, spec, raw_dataset, train_split, eval_split, tokenizer, task_dir)
        elif spec.task_type == "question_answering":
            metrics = _run_question_answering(cfg, task_cfg, spec, raw_dataset, train_split, eval_split, tokenizer, task_dir)
        elif spec.task_type == "multiple_choice":
            metrics = _run_multiple_choice(cfg, task_cfg, spec, raw_dataset, train_split, eval_split, tokenizer, task_dir)
        else:
            raise ValueError(f"Unsupported task type: {spec.task_type}")
    except Exception as exc:  # pragma: no cover - runtime/model/dataset dependent.
        return _blocked_result(spec, "task_run_failed", exc)

    return {
        "name": spec.name,
        "display_name": spec.display_name,
        "type": spec.task_type,
        "status": "completed",
        "metrics": _clean_metric_keys(metrics),
        "config": {
            "dataset": spec.dataset_name,
            "dataset_config": spec.dataset_config,
            "train_split": train_split,
            "eval_split": eval_split,
            "max_seq_length": task_cfg.max_seq_length,
            "max_train_samples": task_cfg.max_train_samples,
            "max_eval_samples": task_cfg.max_eval_samples,
        },
    }


def _load_dataset(spec: TaskSpec) -> Any:
    from datasets import load_dataset

    if spec.dataset_config is None:
        return load_dataset(spec.dataset_name, trust_remote_code=spec.trust_remote_code)
    return load_dataset(spec.dataset_name, spec.dataset_config, trust_remote_code=spec.trust_remote_code)


def _pick_split(dataset: Any, *, preferred: str, fallback: tuple[str, ...]) -> str | None:
    if preferred in dataset:
        return preferred
    for name in fallback:
        if name in dataset:
            return name
    return None


def _select(dataset: Any, max_samples: int | None) -> Any:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    return dataset.select(range(max_samples))


def _training_args(task_cfg: SupervisedDefaultsConfig, output_dir: Path, *, do_train: bool) -> TrainingArguments:
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


def _trainer_processing_kwargs(tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
    signature = inspect.signature(Trainer)
    if "processing_class" in signature.parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def _run_sequence_classification(
    cfg: EvalSuiteConfig,
    task_cfg: SupervisedDefaultsConfig,
    spec: TaskSpec,
    raw_dataset: Any,
    train_split: str | None,
    eval_split: str,
    tokenizer: PreTrainedTokenizerBase,
    task_dir: Path,
) -> dict[str, float]:
    def normalize(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        text_col = _first_present(batch, spec.text_columns)
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
        tokenized = tokenizer(batch["text"], truncation=True, max_length=task_cfg.max_seq_length)
        tokenized["labels"] = batch["labels"]
        return tokenized

    train_dataset = None
    if task_cfg.do_train and train_split is not None:
        train_dataset = _select(raw_dataset[train_split], task_cfg.max_train_samples).map(normalize, batched=True)
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

    eval_raw = _select(raw_dataset[eval_split], task_cfg.max_eval_samples).map(normalize, batched=True)
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
        args=_training_args(task_cfg, task_dir, do_train=train_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_classification_metrics,
        **_trainer_processing_kwargs(tokenizer),
    )
    if train_dataset is not None:
        trainer.train()
    return trainer.evaluate(eval_dataset=eval_dataset)


def _run_token_classification(
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

    def tokenize_and_align_labels(examples: dict[str, list[Any]]) -> dict[str, Any]:
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=task_cfg.max_seq_length,
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
        train_dataset = _select(raw_dataset[train_split], task_cfg.max_train_samples).map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_dataset[train_split].column_names,
        )

    eval_dataset = _select(raw_dataset[eval_split], task_cfg.max_eval_samples).map(
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
        args=_training_args(task_cfg, task_dir, do_train=train_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        **_trainer_processing_kwargs(tokenizer),
    )
    if train_dataset is not None:
        trainer.train()
    return trainer.evaluate(eval_dataset=eval_dataset)


def _run_question_answering(
    cfg: EvalSuiteConfig,
    task_cfg: SupervisedDefaultsConfig,
    spec: TaskSpec,
    raw_dataset: Any,
    train_split: str | None,
    eval_split: str,
    tokenizer: PreTrainedTokenizerBase,
    task_dir: Path,
) -> dict[str, float]:
    doc_stride = int(spec.extra["doc_stride"])
    max_answer_length = int(spec.extra["max_answer_length"])
    n_best = int(spec.extra["n_best"])

    def preprocess_training_examples(examples: dict[str, list[Any]]) -> dict[str, Any]:
        questions = [question.strip() for question in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=task_cfg.max_seq_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []
        for idx, offset in enumerate(offset_mapping):
            sample_idx = sample_map[idx]
            answer = answers[sample_idx]
            if not answer["answer_start"] or not answer["text"]:
                start_positions.append(0)
                end_positions.append(0)
                continue
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(idx)
            context_start = next(i for i, seq_id in enumerate(sequence_ids) if seq_id == 1)
            context_end = len(sequence_ids) - 1
            while sequence_ids[context_end] != 1:
                context_end -= 1
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                token_start = context_start
                while token_start <= context_end and offset[token_start][0] <= start_char:
                    token_start += 1
                token_end = context_end
                while token_end >= context_start and offset[token_end][1] >= end_char:
                    token_end -= 1
                start_positions.append(token_start - 1)
                end_positions.append(token_end + 1)
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_validation_examples(examples: dict[str, list[Any]]) -> dict[str, Any]:
        questions = [question.strip() for question in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=task_cfg.max_seq_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []
        for idx in range(len(inputs["input_ids"])):
            sample_idx = sample_map[idx]
            example_ids.append(examples["id"][sample_idx])
            sequence_ids = inputs.sequence_ids(idx)
            offset = inputs["offset_mapping"][idx]
            inputs["offset_mapping"][idx] = [
                item if sequence_ids[token_idx] == 1 else None for token_idx, item in enumerate(offset)
            ]
        inputs["example_id"] = example_ids
        return inputs

    train_dataset = None
    if task_cfg.do_train and train_split is not None:
        train_dataset = _select(raw_dataset[train_split], task_cfg.max_train_samples).map(
            preprocess_training_examples,
            batched=True,
            remove_columns=raw_dataset[train_split].column_names,
        )

    eval_examples = _select(raw_dataset[eval_split], task_cfg.max_eval_samples)
    eval_dataset = eval_examples.map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_dataset[eval_split].column_names,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
        ignore_mismatched_sizes=True,
    )
    trainer = Trainer(
        model=model,
        args=_training_args(task_cfg, task_dir, do_train=train_dataset is not None),
        train_dataset=train_dataset,
        **_trainer_processing_kwargs(tokenizer),
    )
    if train_dataset is not None:
        trainer.train()
    predictions, _, _ = trainer.predict(eval_dataset)
    start_logits, end_logits = predictions
    return _qa_metrics(start_logits, end_logits, eval_dataset, eval_examples, n_best, max_answer_length)


def _run_multiple_choice(
    cfg: EvalSuiteConfig,
    task_cfg: SupervisedDefaultsConfig,
    spec: TaskSpec,
    raw_dataset: Any,
    train_split: str | None,
    eval_split: str,
    tokenizer: PreTrainedTokenizerBase,
    task_dir: Path,
) -> dict[str, float]:
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
            max_length=task_cfg.max_seq_length,
        )
        return {
            key: [value[i : i + len(ending_names)] for i in range(0, len(value), len(ending_names))]
            for key, value in tokenized.items()
        }

    train_dataset = None
    if task_cfg.do_train and train_split is not None:
        train_dataset = _select(raw_dataset[train_split], task_cfg.max_train_samples).map(
            preprocess,
            batched=True,
            remove_columns=raw_dataset[train_split].column_names,
        )
    eval_dataset = _select(raw_dataset[eval_split], task_cfg.max_eval_samples).map(
        preprocess,
        batched=True,
        remove_columns=raw_dataset[eval_split].column_names,
    )

    model = AutoModelForMultipleChoice.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
        ignore_mismatched_sizes=True,
    )
    trainer = Trainer(
        model=model,
        args=_training_args(task_cfg, task_dir, do_train=train_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=_accuracy_metrics,
        **_trainer_processing_kwargs(tokenizer),
    )
    if train_dataset is not None:
        trainer.train()
    return trainer.evaluate(eval_dataset=eval_dataset)


def _qa_metrics(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    features: Any,
    examples: Any,
    n_best: int,
    max_answer_length: int,
) -> dict[str, float]:
    import evaluate

    example_to_features: dict[str, list[int]] = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in examples:
        answers = []
        for feature_index in example_to_features[example["id"]]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]
            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    answers.append(
                        {
                            "text": example["context"][offsets[start_index][0] : offsets[end_index][1]],
                            "score": start_logit[start_index] + end_logit[end_index],
                        }
                    )
        if answers:
            best = max(answers, key=lambda row: row["score"])
            predicted_answers.append({"id": example["id"], "prediction_text": best["text"]})
        else:
            predicted_answers.append({"id": example["id"], "prediction_text": ""})

    references = [{"id": example["id"], "answers": example["answers"]} for example in examples]
    return {key: float(value) for key, value in evaluate.load("squad").compute(predictions=predicted_answers, references=references).items()}


def _classification_metrics(eval_pred: Any) -> dict[str, float]:
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    metrics = _accuracy_metrics((predictions, labels))
    metrics["macro_f1"] = _macro_f1(preds, labels)
    return metrics


def _accuracy_metrics(eval_pred: Any) -> dict[str, float]:
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": float((preds == labels).astype(np.float32).mean().item())}


def _macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
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


def _first_present(batch: dict[str, list[Any]], columns: tuple[str, ...]) -> str:
    for column in columns:
        if column in batch:
            return column
    raise KeyError(f"None of the expected columns are present: {columns}; got {sorted(batch)}")


def _clean_metric_keys(metrics: dict[str, Any]) -> dict[str, float]:
    clean = {}
    for key, value in metrics.items():
        if key.startswith("eval_"):
            key = key.removeprefix("eval_")
        if isinstance(value, (int, float, np.floating)):
            clean[key] = float(value)
    return clean


def _blocked_result(spec: TaskSpec, status: str, exc: Exception) -> dict[str, Any]:
    return {
        "name": spec.name,
        "display_name": spec.display_name,
        "type": spec.task_type,
        "status": status,
        "metrics": {},
        "error": f"{type(exc).__name__}: {exc}",
        "config": {
            "dataset": spec.dataset_name,
            "dataset_config": spec.dataset_config,
        },
    }
