"""Question answering supervised evaluation adapter."""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from transformers import AutoModelForQuestionAnswering, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from evals.config import EvalSuiteConfig, SupervisedDefaultsConfig
from evals.registry import TaskSpec
from evals.tasks.common import resolve_max_seq_length, select_rows, trainer_processing_kwargs, training_args


def run_question_answering(
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
    max_answer_length = int(spec.extra["max_answer_length"])
    n_best = int(spec.extra["n_best"])
    max_seq_length = resolve_max_seq_length(cfg, task_cfg)
    doc_stride = _resolve_doc_stride(tokenizer, max_seq_length, int(spec.extra["doc_stride"]))

    def preprocess_training_examples(examples: dict[str, list[Any]]) -> dict[str, Any]:
        questions = [question.strip() for question in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_seq_length,
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
            max_length=max_seq_length,
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
        train_dataset = select_rows(train_raw_dataset[train_split], task_cfg.max_train_samples).map(
            preprocess_training_examples,
            batched=True,
            remove_columns=train_raw_dataset[train_split].column_names,
        )

    eval_examples = select_rows(eval_raw_dataset[eval_split], task_cfg.max_eval_samples)
    eval_dataset = eval_examples.map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=eval_raw_dataset[eval_split].column_names,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
        ignore_mismatched_sizes=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args(task_cfg, task_dir, do_train=train_dataset is not None),
        train_dataset=train_dataset,
        **trainer_processing_kwargs(tokenizer),
    )
    if train_dataset is not None:
        trainer.train()
    predictions, _, _ = trainer.predict(eval_dataset)
    start_logits, end_logits = predictions
    return _qa_metrics(start_logits, end_logits, eval_dataset, eval_examples, n_best, max_answer_length)

# Edit the context length by using effective doc size. 
def _resolve_doc_stride(
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    configured_doc_stride: int,
) -> int:
    effective_context_length = max(max_seq_length - tokenizer.num_special_tokens_to_add(pair=True), 1)

    if configured_doc_stride < effective_context_length:
        return configured_doc_stride

    doc_stride = max(effective_context_length // 2 - 1, 0)

    logger.info(
        "Reducing QA doc_stride from {} to {} for effective max context length {}",
        configured_doc_stride,
        doc_stride,
        effective_context_length,
    )

    return doc_stride


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
    return {
        key: float(value)
        for key, value in evaluate.load("squad").compute(predictions=predicted_answers, references=references).items()
    }
