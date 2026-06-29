"""Orchestrate the configured Hindi evaluation suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from evals.config import EvalSuiteConfig
from evals.efficiency import run_efficiency_sweep
from evals.mlm import run_mlm_eval
from evals.registry import get_task_spec
from evals.reporting import write_reports
from evals.retrieval import run_retrieval_eval
from evals.runtime import checkpoint_output_dir
from evals.supervised import run_supervised_task


def run_eval_suite(cfg: EvalSuiteConfig) -> dict[str, Any]:
    output_dir = checkpoint_output_dir(cfg.output_dir, cfg.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Evaluation output directory: {}", output_dir)

    results: list[dict[str, Any]] = []
    if cfg.mlm.enabled:
        logger.info("Running MLM holdout")
        results.append(_safe_layer("mlm_holdout", lambda: run_mlm_eval(cfg, output_dir)))

    for task_name in cfg.tasks:
        def _run_task(name: str = task_name) -> dict[str, Any]:
            spec = get_task_spec(name)
            logger.info("Running supervised task: {} ({})", spec.name, spec.display_name)
            task_result = run_supervised_task(cfg, name, output_dir)
            _write_result(output_dir / "supervised" / name / "metrics.json", task_result)
            return task_result

        results.append(_safe_layer(task_name, _run_task))

    if cfg.retrieval.enabled:
        logger.info("Running retrieval evaluation")
        results.append(_safe_layer("retrieval", lambda: run_retrieval_eval(cfg, output_dir)))

    if cfg.efficiency.enabled:
        logger.info("Running efficiency sweep")
        results.append(_safe_layer("efficiency_sweep", lambda: run_efficiency_sweep(cfg, output_dir)))

    report_paths = write_reports(cfg, output_dir, results)
    return {"output_dir": str(output_dir), "report_paths": report_paths, "results": results}


def _safe_layer(name: str, fn: Any) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - hardware/network/runtime dependent.
        return {
            "name": name,
            "type": name,
            "status": "layer_failed",
            "metrics": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
