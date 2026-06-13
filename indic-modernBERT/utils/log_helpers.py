"""Shared logging helpers for tokenizer training and evaluations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from hydra.core.hydra_config import HydraConfig
from loguru import logger

if TYPE_CHECKING:
    from config.schema import BpeTrainerConfig, EvalConfig, PretokenizationConfig, SuperBpeTrainerConfig
    from tokenizer.pretokenization import PretokenizationStage


def slug(value: str) -> str:
    keep = []
    for ch in value:
        keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(keep).strip("_")[:80]


def setup_training_run_log(
    data_root: Path,
    vocab_sizes: list[int],
    trainer_name: str,
) -> Path:
    data_tag = slug(data_root.name)
    vocab_tag = "-".join(str(vocab_size) for vocab_size in vocab_sizes)
    log_name = f"train_{trainer_name}__vocabs-{vocab_tag}__data-{data_tag}.log"
    return setup_run_log(log_name)


def setup_run_log(log_name: str) -> Path:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    log_path = out_dir / log_name
    logger.remove()
    logger.add(
        log_path,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        enqueue=True,
    )
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}",
    )
    return log_path


def setup_eval_run_log(eval_cfg: EvalConfig, prefix: str) -> Path:
    cand = slug(eval_cfg.tokenizer_path.parent.name or eval_cfg.tokenizer_path.stem)
    base = (
        "multi"
        if len(eval_cfg.baseline_names) > 1
        else slug(eval_cfg.baseline_names[0].split("/")[-1])
    )
    data = slug(eval_cfg.data_root.name)
    log_name = f"{prefix}__cand-{cand}__base-{base}__data-{data}.log"
    return setup_run_log(log_name)


def log_bpe_training_start(
    bpe_cfg: BpeTrainerConfig,
    pretok_cfg: PretokenizationConfig,
) -> None:
    logger.info(
        "Starting BPE training | data_root={} | vocab_sizes={} | min_freq={} | "
        "use_script_norm={} | use_nfkc={}",
        bpe_cfg.data_root,
        bpe_cfg.vocab_sizes,
        bpe_cfg.min_frequency,
        pretok_cfg.use_script_norm,
        pretok_cfg.use_nfkc,
    )


def log_superbpe_training_start(
    superbpe_cfg: SuperBpeTrainerConfig,
    pretok_cfg: PretokenizationConfig,
) -> None:
    logger.info(
        "Starting SuperBPE training | data_root={} | vocab_sizes={} | "
        "transition_fraction={} | min_freq={} | use_script_norm={} | use_nfkc={}",
        superbpe_cfg.data_root,
        superbpe_cfg.vocab_sizes,
        superbpe_cfg.transition_fraction,
        superbpe_cfg.min_frequency,
        pretok_cfg.use_script_norm,
        pretok_cfg.use_nfkc,
    )


def log_vocab_training_complete(vocab_size: int, tokenizer_path: Path) -> None:
    logger.info(
        "Completed vocab_size={} | tokenizer_path={}",
        vocab_size,
        tokenizer_path.resolve(),
    )


def log_hydra_run_log(run_log: Path) -> None:
    logger.info("Hydra experiment log: {}", run_log.resolve())


def log_training_stage(
    *,
    stage_name: str,
    pretokenization_stage: PretokenizationStage,
    target_vocab_size: int,
    current_vocab_size: int | None = None,
) -> None:
    message = (
        "Completed {stage_name} | pretokenization={pretokenization_stage} | "
        "target_vocab_size={target_vocab_size}"
    )

    if current_vocab_size is not None:
        message += " | vocab_size={current_vocab_size}"

    logger.info(
        message,
        stage_name=stage_name,
        pretokenization_stage=pretokenization_stage,
        target_vocab_size=target_vocab_size,
        current_vocab_size=current_vocab_size,
    )
