"""Train a Hindi BPE tokenizer."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from config import load_tokenizer_config
from utils.log_helpers import (
    log_bpe_training_start,
    log_hydra_run_log,
    log_vocab_training_complete,
    setup_training_run_log,
)

from tokenizer.pretokenization import PretokenizationStage

from .common import (
    attach_cls_sep_processor,
    create_bpe_tokenizer,
    iter_corpus_texts,
    save_tokenizer,
    train_bpe_stage,
)


def train_bpe(
    data_root: Path,
    output_dir: Path,
    text_column: str,
    vocab_size: int,
    min_frequency: int,
    *,
    pretokenization_stage: PretokenizationStage = "subword",
    use_script_norm: bool = True,
    use_nfkc: bool = True,
) -> Path:
    tokenizer = create_bpe_tokenizer(use_nfkc=use_nfkc)

    train_bpe_stage(
        tokenizer,
        lambda: iter_corpus_texts(
            data_root,
            text_column,
            use_script_norm=use_script_norm,
            progress_desc=f"BPE vocab={vocab_size}",
        ),
        pretokenization_stage=pretokenization_stage,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        stage_name="BPE",
    )

    attach_cls_sep_processor(tokenizer)
    return save_tokenizer(tokenizer, output_dir)


@hydra.main(version_base=None, config_path="../../../configs/hi", config_name="tokenizer")
def main(cfg: DictConfig) -> None:

    config = load_tokenizer_config(cfg)
    bpe_cfg = config.trainer.bpe
    pretok_cfg = config.pretokenization
    run_log = setup_training_run_log(bpe_cfg.data_root, bpe_cfg.vocab_sizes, "bpe")
    
    log_bpe_training_start(bpe_cfg, pretok_cfg)

    for run in bpe_cfg.iter_runs():
        tokenizer_path = train_bpe(
            data_root=bpe_cfg.data_root,
            output_dir=run.output_dir,
            text_column=bpe_cfg.text_column,
            vocab_size=run.vocab_size,
            min_frequency=bpe_cfg.min_frequency,
            pretokenization_stage=pretok_cfg.stage,
            use_script_norm=pretok_cfg.use_script_norm,
            use_nfkc=pretok_cfg.use_nfkc,
        )
        log_vocab_training_complete(run.vocab_size, tokenizer_path)

    log_hydra_run_log(run_log)


if __name__ == "__main__":
    main()
