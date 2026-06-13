"""Train a Hindi SuperBPE tokenizer (MUTANT two-stage subword + multiword)."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from config import load_tokenizer_config
from utils.log_helpers import (
    log_hydra_run_log,
    log_superbpe_training_start,
    log_training_stage,
    log_vocab_training_complete,
    setup_training_run_log,
)
from constants import validate_vocab_size
from .common import (
    attach_cls_sep_processor,
    create_bpe_tokenizer,
    make_corpus_iterator,
    resolve_transition_vocab_size,
    save_bpe_checkpoint,
    save_tokenizer,
    train_bpe_stage,
    train_superbpe_stage2,
)


def train_superbpe(
    data_root: Path,
    output_dir: Path,
    text_column: str,
    vocab_size: int,
    min_frequency: int,
    *,
    transition_vocab_size: int | None = None,
    transition_fraction: float = 0.9,
    use_script_norm: bool = True,
    use_nfkc: bool = True,
) -> Path:

    validate_vocab_size(vocab_size)

    stage1_vocab_size = resolve_transition_vocab_size(
        vocab_size,
        transition_vocab_size=transition_vocab_size,
        transition_fraction=transition_fraction,
    )

    tokenizer = create_bpe_tokenizer(use_nfkc=use_nfkc)

    stage1_factory = make_corpus_iterator(
        data_root,
        text_column,
        use_script_norm=use_script_norm,
        progress_desc=f"SuperBPE S1 vocab={vocab_size}",
    )
    stage2_factory = make_corpus_iterator(
        data_root,
        text_column,
        use_script_norm=use_script_norm,
        progress_desc=f"SuperBPE S2 vocab={vocab_size}",
    )
    stage1_checkpoint = output_dir / ".stage1_checkpoint"

    # Stage 1: full regex pretokenization (words, digits, punctuation, whitespace).
    train_bpe_stage(
        tokenizer,
        stage1_factory,
        pretokenization_stage="subword",
        vocab_size=stage1_vocab_size,
        min_frequency=min_frequency,
        stage_name="SuperBPE stage 1 (subword)",
    )

    save_bpe_checkpoint(tokenizer, stage1_checkpoint)

    # Stage 2: no pretokenization; extend from merges.txt (reference SuperBPE cwd flow).
    tokenizer = train_superbpe_stage2(
        stage1_checkpoint,
        stage2_factory,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        use_nfkc=use_nfkc,
    )

    log_training_stage(
        stage_name="SuperBPE stage 2 (multiword)",
        pretokenization_stage="superword",
        target_vocab_size=vocab_size,
        current_vocab_size=tokenizer.get_vocab_size(),
    )

    attach_cls_sep_processor(tokenizer)
    return save_tokenizer(tokenizer, output_dir)


@hydra.main(version_base=None, config_path="../../../configs", config_name="tokenizer")
def main(cfg: DictConfig) -> None:
    config = load_tokenizer_config(cfg)
    superbpe_cfg = config.trainer.superbpe
    pretok_cfg = config.pretokenization

    run_log = setup_training_run_log(
        superbpe_cfg.data_root,
        superbpe_cfg.vocab_sizes,
        "superbpe",
    )

    log_superbpe_training_start(superbpe_cfg, pretok_cfg)

    for run in superbpe_cfg.iter_runs():
        tokenizer_path = train_superbpe(
            data_root=superbpe_cfg.data_root,
            output_dir=run.output_dir,
            text_column=superbpe_cfg.text_column,
            vocab_size=run.vocab_size,
            min_frequency=superbpe_cfg.min_frequency,
            transition_vocab_size=run.transition_vocab_size,
            transition_fraction=superbpe_cfg.transition_fraction,
            use_script_norm=pretok_cfg.use_script_norm,
            use_nfkc=pretok_cfg.use_nfkc,
        )
        
        log_vocab_training_complete(run.vocab_size, tokenizer_path)

    log_hydra_run_log(run_log)


if __name__ == "__main__":
    main()
