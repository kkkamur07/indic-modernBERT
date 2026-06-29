"""Hydra entrypoint for the Hindi evaluation suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "indic-modernBERT"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import hydra
from loguru import logger
from omegaconf import DictConfig

from evals.config import config_to_jsonable, load_eval_suite_config
from evals.runtime import active_context_length, model_run_slug
from evals.runner import run_eval_suite
from utils.log_helpers import setup_run_log


@hydra.main(version_base=None, config_path="../configs/hi/evals", config_name="hindi_phase1")
def main(cfg: DictConfig) -> None:
    eval_cfg = load_eval_suite_config(cfg)
    summaries = []
    failures = []

    for index, model_cfg in enumerate(eval_cfg.models, start=1):
        run_cfg = eval_cfg.for_model(model_cfg)
        setup_run_log(f"evals__model-{model_run_slug(run_cfg.model)}.log")
        logger.info(
            "Evaluating model {}/{}: {} | context_mode={} | active_context_length={} | max_sequence_length={}",
            index,
            len(eval_cfg.models),
            run_cfg.model.model_name_or_path,
            run_cfg.model.context_mode,
            active_context_length(run_cfg.model),
            run_cfg.model.max_sequence_length,
        )
        logger.info("Selected supervised tasks: {}", run_cfg.tasks)

        summary = run_eval_suite(run_cfg)
        summary["model_name_or_path"] = run_cfg.model.model_name_or_path
        summary["context_mode"] = run_cfg.model.context_mode
        summary["active_context_length"] = active_context_length(run_cfg.model)
        summary["max_sequence_length"] = run_cfg.model.max_sequence_length
        summaries.append(summary)

        output_dir = Path(summary["output_dir"])
        (output_dir / "resolved_config.json").write_text(
            json.dumps(config_to_jsonable(cfg), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote reports: {}", summary["report_paths"])

        failures.extend(row for row in summary["results"] if row.get("status") != "completed")

    if len(summaries) > 1:
        aggregate_path = eval_cfg.output_dir / "multi_model_summary.json"
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote multi-model summary: {}", aggregate_path)

    sys.exit(len(failures))


if __name__ == "__main__":
    main()
