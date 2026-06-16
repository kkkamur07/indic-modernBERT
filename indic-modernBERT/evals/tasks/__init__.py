"""Supervised task adapters."""

from evals.tasks.multiple_choice import run_multiple_choice
from evals.tasks.question_answering import run_question_answering
from evals.tasks.sequence_classification import run_sequence_classification
from evals.tasks.token_classification import run_token_classification

__all__ = [
    "run_multiple_choice",
    "run_question_answering",
    "run_sequence_classification",
    "run_token_classification",
]
