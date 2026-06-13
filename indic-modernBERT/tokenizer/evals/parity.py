"""Evaluate cross-lingual parity ratio (Hindi vs reference language)."""

from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig

from config import load_eval_config
from utils.log_helpers import log_hydra_run_log, setup_eval_run_log

from .common import (
    collect_cross_lingual_parity,
    fast_encode_fns,
    hf_encode_fns,
    load_candidate_tokenizer,
    load_hf_tokenizer,
)


@hydra.main(version_base=None, config_path="../../../configs", config_name="tokenizer")
def main(cfg: DictConfig) -> None:
    eval_cfg = load_eval_config(cfg, "parity")
    run_log = setup_eval_run_log(eval_cfg, "parity")

    reference_name = eval_cfg.reference_tokenizer_name
    parallel_path = eval_cfg.parallel_data_path
    assert reference_name is not None
    assert parallel_path is not None

    candidate = load_candidate_tokenizer(eval_cfg.tokenizer_path)
    reference = load_hf_tokenizer(reference_name)
    cand_len, _ = fast_encode_fns(candidate)
    ref_len, _ = hf_encode_fns(reference)

    parity = collect_cross_lingual_parity(
        hindi_tokenize_len=cand_len,
        reference_lang_tokenize_len=ref_len,
        parallel_path=parallel_path,
        hindi_column=eval_cfg.parallel_hindi_column,
        reference_column=eval_cfg.parallel_reference_column,
    )

    logger.info(
        "Candidate parity ratio | line-avg={:.6f} | micro={:.6f} | rows={}",
        parity["parity_ratio"],
        parity["parity_ratio_micro"],
        parity["rows"],
    )
    
    logger.info("Closer to 1.0 is better (Petrov et al., 2023).")
    log_hydra_run_log(run_log)


if __name__ == "__main__":
    main()
