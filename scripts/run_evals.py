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
from evals.runner import run_eval_suite
from utils.log_helpers import setup_run_log, slug


@hydra.main(version_base=None, config_path="../configs/evals", config_name="hindi_phase1")
def main(cfg: DictConfig) -> float:
    eval_cfg = load_eval_suite_config(cfg)
    setup_run_log(f"evals__model-{slug(eval_cfg.model.model_name_or_path)}.log")
    logger.info("Evaluating model: {}", eval_cfg.model.model_name_or_path)
    logger.info("Selected supervised tasks: {}", eval_cfg.tasks)

    summary = run_eval_suite(eval_cfg)
    output_dir = Path(summary["output_dir"])
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config_to_jsonable(cfg), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote reports: {}", summary["report_paths"])

    failures = [row for row in summary["results"] if row.get("status") != "completed"]
    return float(len(failures))


if __name__ == "__main__":
    main()
