"""Run intrinsic tokenizer metrics (fertility, bytes/token, NSL, Rényi)."""

from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig

from config import load_eval_config
from utils.log_helpers import log_hydra_run_log, setup_eval_run_log

from .common import (
    collect_cross_lingual_parity,
    collect_intrinsic_metrics,
    fast_encode_fns,
    hf_encode_fns,
    load_candidate_tokenizer,
    load_hf_tokenizer,
)


def _log_intrinsic(label: str, metrics: dict[str, float | int]) -> None:
    logger.info(
        "{} | fertility={:.6f} | bytes/token={:.6f} | NSL={:.6f} | "
        "Rényi entropy={:.6f} | Rényi efficiency={:.6f}",
        label,
        metrics["fertility"],
        metrics["bytes_per_token"],
        metrics["nsl"],
        metrics["renyi_entropy"],
        metrics["renyi_efficiency"],
    )


@hydra.main(version_base=None, config_path="../../../configs", config_name="tokenizer")
def main(cfg: DictConfig) -> None:
    eval_cfg = load_eval_config(cfg, "intrinsic")
    run_log = setup_eval_run_log(eval_cfg, "intrinsic")

    candidate = load_candidate_tokenizer(eval_cfg.tokenizer_path)
    cand_len, cand_tokens = fast_encode_fns(candidate)

    reference = (
        load_hf_tokenizer(eval_cfg.reference_tokenizer_name)
        if eval_cfg.reference_tokenizer_name is not None
        else None
    )

    ref_len = hf_encode_fns(reference)[0] if reference is not None else None

    candidate_metrics = collect_intrinsic_metrics(
        tokenize_len=cand_len,
        tokenize_tokens=cand_tokens,
        data_root=eval_cfg.data_root,
        text_column=eval_cfg.text_column,
        reference_tokenize_len=ref_len,
        vocab_size=candidate.get_vocab_size(),
        renyi_alpha=eval_cfg.renyi_alpha,
    )

    _log_intrinsic("Candidate", candidate_metrics)

    if eval_cfg.parallel_data_path is not None and reference is not None and ref_len is not None:

        parity = collect_cross_lingual_parity(
            hindi_tokenize_len=cand_len,
            reference_lang_tokenize_len=ref_len,
            parallel_path=eval_cfg.parallel_data_path,
            hindi_column=eval_cfg.parallel_hindi_column,
            reference_column=eval_cfg.parallel_reference_column,
        )

        logger.info(
            "Candidate parity | line-avg={:.6f} | micro={:.6f} | rows={}",
            parity["parity_ratio"],
            parity["parity_ratio_micro"],
            parity["rows"],
        )

    for baseline_name in eval_cfg.baseline_names:
        baseline = load_hf_tokenizer(baseline_name)
        base_len, base_tokens = hf_encode_fns(baseline)

        baseline_metrics = collect_intrinsic_metrics(
            tokenize_len=base_len,
            tokenize_tokens=base_tokens,
            data_root=eval_cfg.data_root,
            text_column=eval_cfg.text_column,
            vocab_size=baseline.vocab_size,
            renyi_alpha=eval_cfg.renyi_alpha,
        )

        _log_intrinsic(f"Baseline [{baseline_name}]", baseline_metrics)

    logger.info(
        "Metric guide | fertility ↓ | bytes/token ↑ | NSL ↓ | Rényi efficiency ↑ | parity ≈ 1"
    )
    
    log_hydra_run_log(run_log)


if __name__ == "__main__":
    main()
