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
    train_dataset_name: str | None = None
    train_dataset_config: str | None = None
    train_dataset_trust_remote_code: bool | None = None
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str | None = "test"
    num_labels: int | None = None
    label_column: str = "label"
    text_columns: tuple[str, ...] = ()
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
        # IndicXTREME follows cross-lingual transfer: train on English Amazon
        # reviews, then evaluate zero-shot on IndicSentiment target splits.
        # The original HF dataset is defunct; this is a loadable mirror.
        train_dataset_name="buruzaemon/amazon_reviews_multi",
        train_dataset_config="en",
        metric_names=("accuracy", "macro_f1"),
        # Actual HF column name is "LABEL", not "label".
        label_column="LABEL",
        text_columns=("INDIC REVIEW", "sentence1", "text", "review_body"),
        max_seq_length=128,
        num_labels=2,
        # Keep training enabled when the configured dataset exposes a train split.
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
    "massive_intent": TaskSpec(
        name="massive_intent",
        display_name="MASSIVE Hindi Intent",
        task_type="sequence_classification",
        dataset_name="AmazonScience/massive",
        dataset_config="hi-IN",
        metric_names=("accuracy", "macro_f1"),
        label_column="intent",
        text_columns=("utt",),
        max_seq_length=128,
        num_labels=60,
    ),
    "qa": TaskSpec(
        name="qa",
        display_name="IndicQA Hindi",
        task_type="question_answering",
        dataset_name="ai4bharat/IndicQA",
        dataset_config="indicqa.hi",
        # Official IndicBERT fine-tunes QA on English SQuAD via XTREME.
        train_dataset_name="google/xtreme",
        train_dataset_config="SQuAD",
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
        # Official IndicBERT fine-tunes XCOPA transfer on English Social IQa.
        train_dataset_name="allenai/social_i_qa",
        train_dataset_config=None,
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
