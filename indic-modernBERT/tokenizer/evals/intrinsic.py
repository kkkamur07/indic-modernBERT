"""Run intrinsic tokenizer metrics (fertility, bytes/token, NSL, Rényi)."""

from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig

from config import load_eval_config, load_tokenizer_config
from utils.log_helpers import log_hydra_run_log, setup_eval_run_log

from .common import (
    collect_intrinsic_metrics,
    fast_encode_fns,
    hf_encode_fns,
    load_candidate_tokenizer,
    load_hf_tokenizer,
    log_fertility_comparison,
    log_intrinsic_metrics,
    short_tokenizer_name,
)


@hydra.main(version_base=None, config_path="../../../configs/hi", config_name="tokenizer")
def main(cfg: DictConfig) -> None:
    eval_cfg = load_eval_config(cfg, "intrinsic")
    tokenizer_cfg = load_tokenizer_config(cfg)
    pretok_cfg = tokenizer_cfg.pretokenization
    use_script_norm = pretok_cfg.use_script_norm
    use_nfkc = pretok_cfg.use_nfkc
    if eval_cfg.tokenizer_path is None:
        raise ValueError(
            "intrinsic eval requires tokenizer.evals.intrinsic.tokenizer_path "
            "or use make eval-bpe to compare all trained vocabs."
        )
    run_log = setup_eval_run_log(eval_cfg, "intrinsic")
    fertility_by_label: dict[str, float] = {}

    preprocess_steps = [
        name
        for enabled, name in (
            (use_script_norm, "script norm"),
            (use_nfkc, "NFKC"),
        )
        if enabled
    ]
    if preprocess_steps:
        logger.info(
            "Applying preprocess_for_eval ({}) to all tokenizers before encode",
            " + ".join(preprocess_steps),
        )

    candidate = load_candidate_tokenizer(eval_cfg.tokenizer_path)
    cand_len, cand_tokens = fast_encode_fns(candidate)

    reference = (
        load_hf_tokenizer(eval_cfg.reference_tokenizer_name)
        if eval_cfg.reference_tokenizer_name is not None
        else None
    )

    ref_len = None
    if reference is not None:
        assert eval_cfg.reference_tokenizer_name is not None
        ref_name = eval_cfg.reference_tokenizer_name
        logger.info("Loading reference tokenizer: {}", ref_name)
        ref_len, ref_tokens = hf_encode_fns(reference)

        reference_metrics = collect_intrinsic_metrics(
            tokenize_len=ref_len,
            tokenize_tokens=ref_tokens,
            data_root=eval_cfg.data_root,
            text_column=eval_cfg.text_column,
            vocab_size=reference.vocab_size,
            renyi_alpha=eval_cfg.renyi_alpha,
            use_script_norm=use_script_norm,
            use_nfkc=use_nfkc,
            max_shards=eval_cfg.max_shards,
            progress_desc=f"Reference [{short_tokenizer_name(ref_name)}]",
        )
        reference_label = f"Reference [{short_tokenizer_name(ref_name)}]"
        log_intrinsic_metrics(reference_label, reference_metrics)
        fertility_by_label[reference_label] = float(reference_metrics["fertility"])

    candidate_metrics = collect_intrinsic_metrics(
        tokenize_len=cand_len,
        tokenize_tokens=cand_tokens,
        data_root=eval_cfg.data_root,
        text_column=eval_cfg.text_column,
        reference_tokenize_len=ref_len,
        vocab_size=candidate.get_vocab_size(),
        renyi_alpha=eval_cfg.renyi_alpha,
        use_script_norm=use_script_norm,
        use_nfkc=use_nfkc,
        max_shards=eval_cfg.max_shards,
        progress_desc="Candidate",
    )

    log_intrinsic_metrics("Candidate", candidate_metrics)
    fertility_by_label["Candidate"] = float(candidate_metrics["fertility"])

    for baseline_name in eval_cfg.baseline_names:
        if baseline_name == eval_cfg.reference_tokenizer_name:
            continue

        logger.info("Loading baseline tokenizer: {}", baseline_name)
        baseline = load_hf_tokenizer(baseline_name)
        base_len, base_tokens = hf_encode_fns(baseline)
        baseline_label = f"Baseline [{short_tokenizer_name(baseline_name)}]"

        baseline_metrics = collect_intrinsic_metrics(
            tokenize_len=base_len,
            tokenize_tokens=base_tokens,
            data_root=eval_cfg.data_root,
            text_column=eval_cfg.text_column,
            vocab_size=baseline.vocab_size,
            renyi_alpha=eval_cfg.renyi_alpha,
            use_script_norm=use_script_norm,
            use_nfkc=use_nfkc,
            max_shards=eval_cfg.max_shards,
            progress_desc=baseline_label,
        )

        log_intrinsic_metrics(baseline_label, baseline_metrics)
        fertility_by_label[baseline_label] = float(baseline_metrics["fertility"])

    log_fertility_comparison(fertility_by_label)

    logger.info(
        "Metric guide | fertility ↓ | bytes/token ↑ | NSL ↓ | Rényi efficiency ↑"
    )

    log_hydra_run_log(run_log)


if __name__ == "__main__":
    main()
