"""Evaluate cross-language parity for fertility and bytes-per-token."""

from __future__ import annotations

from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from common import collect_stats, get_baseline_names, parity_from_metric, setup_eval_run_log


def build_parity_block(report: dict[str, object]) -> dict[str, dict[str, float]]:
    per_language = report["per_language"]

    return {
        "fertility": parity_from_metric(per_language, "fertility"),
        "bytes_per_token": parity_from_metric(per_language, "bytes_per_token"),
    }


@hydra.main(version_base=None, config_path="../../../configs", config_name="tokenizer")
def main(cfg: DictConfig) -> None:
    eval_cfg = cfg.tokenizer.evals.parity
    run_log = setup_eval_run_log(eval_cfg, prefix="parity")
    
    candidate_tokenizer = Tokenizer.from_file(str(Path(eval_cfg.tokenizer_path)))
    
    candidate = collect_stats(
        tokenize_len=lambda text: len(
            candidate_tokenizer.encode(text, add_special_tokens=False).ids
        ),
        data_root=Path(eval_cfg.data_root),
        text_column=eval_cfg.text_column,
    )

    candidate_parity = build_parity_block(candidate)

    logger.info(
        "Candidate parity ratio | fertility={:.6f} | bytes_per_token={:.6f}",
        candidate_parity["fertility"]["parity_ratio"],
        candidate_parity["bytes_per_token"]["parity_ratio"],
    )

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

        baseline_parity = build_parity_block(baseline)

        logger.info(
            "Baseline [{}] parity ratio | fertility={:.6f} | bytes_per_token={:.6f}",
            baseline_name,
            baseline_parity["fertility"]["parity_ratio"],
            baseline_parity["bytes_per_token"]["parity_ratio"],
        )
        logger.info(
            "fertility: {:.4f} (baseline {:.4f})",
            candidate_parity["fertility"]["parity_ratio"],
            baseline_parity["fertility"]["parity_ratio"],
        )
        logger.info(
            "bytes/token: {:.4f} (baseline {:.4f})",
            candidate_parity["bytes_per_token"]["parity_ratio"],
            baseline_parity["bytes_per_token"]["parity_ratio"],
        )
    logger.info("Hydra experiment log: {}", run_log.resolve())


if __name__ == "__main__":
    main()
