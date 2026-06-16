"""Supervised Hindi gate orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from transformers import AutoTokenizer

from evals.config import EvalSuiteConfig, SupervisedDefaultsConfig
from evals.registry import TaskSpec, get_task_spec
from evals.runtime import set_eval_seed
from evals.tasks import (
    run_multiple_choice,
    run_question_answering,
    run_sequence_classification,
    run_token_classification,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

TaskRunner = Callable[
    [EvalSuiteConfig, SupervisedDefaultsConfig, TaskSpec, Any, str | None, str, PreTrainedTokenizerBase, Path],
    dict[str, float],
]

TASK_RUNNERS: dict[str, TaskRunner] = {
    "sequence_classification": run_sequence_classification,
    "token_classification": run_token_classification,
    "question_answering": run_question_answering,
    "multiple_choice": run_multiple_choice,
}


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
        runner = TASK_RUNNERS[spec.task_type]
        metrics = runner(cfg, task_cfg, spec, raw_dataset, train_split, eval_split, tokenizer, task_dir)
    except KeyError as exc:
        return _blocked_result(spec, "unsupported_task_type", ValueError(f"Unsupported task type: {spec.task_type}"))
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
