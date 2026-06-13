"""Export a Composer FlexBERT checkpoint to Hugging Face format."""

from __future__ import annotations

from pathlib import Path

import transformers
from loguru import logger

from config import load_modernbert_arch_config
from model.factory import build_modernbert_config
from model.modernbert.model import FlexBertForMaskedLM


def export_composer_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    arch_config_path: Path | None = None,
    pretrained_model_name: str = "bert-base-uncased",
    tokenizer_path: Path | None = None,
) -> Path:
    """Load a Composer `.pt` checkpoint and write HF `save_pretrained` artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if arch_config_path is None:
        raise ValueError("`arch_config_path` is required.")

    arch = load_modernbert_arch_config(arch_config_path)
    config = build_modernbert_config(
        pretrained_model_name=pretrained_model_name,
        model_config=arch,
    )

    model = FlexBertForMaskedLM.from_composer(pretrained_checkpoint=str(checkpoint_path), config=config)
    model.save_pretrained(output_dir)
    logger.info("Saved model weights to {}", output_dir)

    if tokenizer_path is not None:
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        tokenizer.save_pretrained(output_dir)
        logger.info("Saved tokenizer from {}", tokenizer_path)

    return output_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Composer checkpoint (.pt)")
    parser.add_argument("output_dir", type=Path, help="HF export directory")
    parser.add_argument("--config", type=Path, default=Path("configs/model/modernbert_base.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=None)
    args = parser.parse_args()

    export_composer_checkpoint(
        args.checkpoint,
        args.output_dir,
        arch_config_path=args.config,
        tokenizer_path=args.tokenizer,
    )


if __name__ == "__main__":
    main()
