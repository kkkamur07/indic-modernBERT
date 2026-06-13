"""Evaluate bytes-per-token compression efficiency on Hindi text."""

from __future__ import annotations

from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from .common import collect_stats, get_baseline_names, setup_eval_run_log


@hydra.main(version_base=None, config_path="../../../configs", config_name="tokenizer")
def main(cfg: DictConfig) -> None:
    eval_cfg = cfg.tokenizer.evals.bytes_per_token
    run_log = setup_eval_run_log(eval_cfg, prefix="bytes_per_token")

    candidate_tokenizer = Tokenizer.from_file(str(Path(eval_cfg.tokenizer_path)))
    candidate = collect_stats(
        tokenize_len=lambda text: len(
            candidate_tokenizer.encode(text, add_special_tokens=False).ids
        ),
        data_root=Path(eval_cfg.data_root),
        text_column=eval_cfg.text_column,
    )

    cand = candidate["bytes_per_token"]
    logger.info("Candidate Hindi bytes/token: {:.6f}", cand)

    baseline_names = get_baseline_names(eval_cfg)

    for baseline_name in baseline_names:
        baseline_tokenizer = AutoTokenizer.from_pretrained(
            baseline_name,
            use_fast=True,
        )

        baseline = collect_stats(
            tokenize_len=lambda text: len(
                baseline_tokenizer(text, add_special_tokens=False)["input_ids"]
            ),
            data_root=Path(eval_cfg.data_root),
            text_column=eval_cfg.text_column,
        )

        base = baseline["bytes_per_token"]
        logger.info("Baseline [{}] Hindi bytes/token: {:.6f}", baseline_name, base)

    logger.info("Higher bytes/token is better.")
    logger.info("Hydra experiment log: {}", run_log.resolve())


if __name__ == "__main__":
    main()
