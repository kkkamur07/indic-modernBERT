"""Validate a trained SuperBPE tokenizer loads and encodes Hindi text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger
from tokenizers import Tokenizer


def validate_tokenizer(tokenizer_path: Path, sample: str) -> None:
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encoded = tokenizer.encode(sample)
    logger.info("Tokenizer: {}", tokenizer_path)
    logger.info("Sample: {!r}", sample)
    logger.info("Tokens: {} | ids: {}", encoded.tokens, encoded.ids)
    logger.info("Vocab size: {}", tokenizer.get_vocab_size(with_added_tokens=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizer/superbpe_vs50368/tokenizer.json"),
    )
    parser.add_argument(
        "--text",
        default="यह एक छोटा हिंदी वाक्य है।",
        help="Hindi sample string to encode",
    )
    args = parser.parse_args()

    tokenizer_path = args.tokenizer if args.tokenizer.is_absolute() else Path.cwd() / args.tokenizer
    try:
        validate_tokenizer(tokenizer_path, args.text)
    except FileNotFoundError as exc:
        logger.error("{}", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
