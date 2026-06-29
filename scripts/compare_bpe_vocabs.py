"""Compare intrinsic metrics across all trained BPE vocab sizes on eval holdout."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "indic-modernBERT"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import hydra
from loguru import logger
from omegaconf import DictConfig

from config import load_eval_config, load_tokenizer_config
from tokenizer.evals.common import (
    collect_intrinsic_metrics,
    fast_encode_fns,
    hf_encode_fns,
    load_candidate_tokenizer,
    load_hf_tokenizer,
    log_fertility_comparison,
    log_intrinsic_metrics,
    short_tokenizer_name,
)
from utils.log_helpers import log_hydra_run_log, setup_eval_run_log


def _log_comparison_table(rows: list[tuple[str, dict]]) -> None:
    logger.info("--- vocab comparison (eval holdout) ---")
    logger.info("label | fertility | bytes/token | NSL | Rényi eff | vocab")
    for label, m in rows:
        logger.info(
            "{} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {}",
            label,
            m["fertility"],
            m["bytes_per_token"],
            m["nsl"],
            m["renyi_efficiency"],
            m.get("vocab_size", "-"),
        )


@hydra.main(version_base=None, config_path="../configs/hi", config_name="tokenizer")
def main(cfg: DictConfig) -> None:
    tokenizer_cfg = load_tokenizer_config(cfg)
    eval_cfg = load_eval_config(cfg, "intrinsic")
    bpe_cfg = tokenizer_cfg.trainer.bpe
    use_script_norm = tokenizer_cfg.pretokenization.use_script_norm
    run_log = setup_eval_run_log(eval_cfg, "intrinsic_all_vocabs")

    fertility_by_label: dict[str, float] = {}
    table_rows: list[tuple[str, dict]] = []

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
            progress_desc=f"Reference [{short_tokenizer_name(ref_name)}]",
        )
        reference_label = f"Reference [{short_tokenizer_name(ref_name)}]"
        log_intrinsic_metrics(reference_label, reference_metrics)

        fertility_by_label[reference_label] = float(reference_metrics["fertility"])
        row = dict(reference_metrics)
        row["vocab_size"] = reference.vocab_size
        table_rows.append((reference_label, row))

    for run in bpe_cfg.iter_runs():
        tokenizer_path = run.output_dir / "tokenizer.json"

        if not tokenizer_path.is_file():
            logger.warning("Skipping vocab_size={} — missing {}", run.vocab_size, tokenizer_path)
            continue

        logger.info("Evaluating vocab_size={} | {}", run.vocab_size, tokenizer_path)
        candidate = load_candidate_tokenizer(tokenizer_path)
        cand_len, cand_tokens = fast_encode_fns(candidate)
        label = f"BPE vs{run.vocab_size}"

        metrics = collect_intrinsic_metrics(
            tokenize_len=cand_len,
            tokenize_tokens=cand_tokens,
            data_root=eval_cfg.data_root,
            text_column=eval_cfg.text_column,
            reference_tokenize_len=ref_len,
            vocab_size=candidate.get_vocab_size(),
            renyi_alpha=eval_cfg.renyi_alpha,
            use_script_norm=use_script_norm,
            progress_desc=label,
        )
        log_intrinsic_metrics(label, metrics)

        fertility_by_label[label] = float(metrics["fertility"])
        row = dict(metrics)
        row["vocab_size"] = candidate.get_vocab_size()
        table_rows.append((label, row))

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
            progress_desc=baseline_label,
        )

        log_intrinsic_metrics(baseline_label, baseline_metrics)
        fertility_by_label[baseline_label] = float(baseline_metrics["fertility"])
        row = dict(baseline_metrics)
        row["vocab_size"] = baseline.vocab_size
        table_rows.append((baseline_label, row))

    if table_rows:
        _log_comparison_table(table_rows)

    if fertility_by_label:
        log_fertility_comparison(fertility_by_label)

    logger.info("Metric guide | fertility ↓ | bytes/token ↑ | NSL ↓ | Rényi efficiency ↑")

    log_hydra_run_log(run_log)


if __name__ == "__main__":
    main()
