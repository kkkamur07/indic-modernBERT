"""Task registry for the Hindi phase-1 supervised gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TaskType = Literal["sequence_classification", "token_classification", "question_answering", "multiple_choice"]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    display_name: str
    task_type: TaskType
    dataset_name: str
    dataset_config: str | None
    metric_names: tuple[str, ...]
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str | None = "test"
    num_labels: int | None = None
    label_column: str = "label"
    text_columns: tuple[str, ...] = ()
    remove_columns: tuple[str, ...] = ()
    max_seq_length: int = 128
    trust_remote_code: bool = True
    # When True, continuation subword tokens get the same NER label as their first
    # subword — matching the IndicBERT reference (label_all_tokens=True). Required for
    # F1 scores to be comparable with published IndicBERT Naamapadam numbers.
    label_all_tokens: bool = False
    extra: dict[str, str | int | bool] = field(default_factory=dict)


TASK_REGISTRY: dict[str, TaskSpec] = {
    "sentiment": TaskSpec(
        name="sentiment",
        display_name="IndicSentiment Hindi",
        task_type="sequence_classification",
        dataset_name="ai4bharat/IndicSentiment",
        dataset_config="translation-hi",
        metric_names=("accuracy", "macro_f1"),
        # Actual HF column name is "LABEL", not "label".
        label_column="LABEL",
        text_columns=("INDIC REVIEW", "sentence1", "text", "review_body"),
        max_seq_length=128,
        num_labels=2,
        # IndicSentiment has no train split — only a test set.
        # Fine-tuning cannot happen; the model is evaluated zero-shot with a fresh
        # randomly-initialised classification head. Treat this metric as a
        # representation-quality signal, not a fine-tuned accuracy number.
        train_split="train",
    ),
    "ner": TaskSpec(
        name="ner",
        display_name="Naamapadam Hindi NER",
        task_type="token_classification",
        dataset_name="ai4bharat/naamapadam",
        dataset_config="hi",
        metric_names=("precision", "recall", "f1", "accuracy"),
        label_column="ner_tags",
        text_columns=("tokens",),
        max_seq_length=128,
        # Match IndicBERT reference (ner.py label_all_tokens=True default) so that
        # F1 scores are comparable with published Naamapadam results.
        label_all_tokens=True,
    ),
    "qa": TaskSpec(
        name="qa",
        display_name="IndicQA Hindi",
        task_type="question_answering",
        dataset_name="ai4bharat/IndicQA",
        dataset_config="indicqa.hi",
        metric_names=("exact_match", "f1"),
        label_column="answers",
        text_columns=("question", "context"),
        max_seq_length=384,
        extra={"doc_stride": 128, "n_best": 20, "max_answer_length": 30},
    ),
    "copa": TaskSpec(
        name="copa",
        display_name="IndicCOPA Hindi",
        task_type="multiple_choice",
        dataset_name="ai4bharat/IndicCOPA",
        dataset_config="translation-hi",
        metric_names=("accuracy",),
        label_column="label",
        text_columns=("premise", "question", "choice1", "choice2"),
        max_seq_length=512,
    ),
}


def get_task_spec(name: str) -> TaskSpec:
    try:
        return TASK_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown eval task '{name}'. Available tasks: {sorted(TASK_REGISTRY)}") from exc
