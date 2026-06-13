"""Validate SuperBPE pretokenization and merge-extension behavior."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer

from tokenizer.pretokenization import build_pre_tokenizer, describe_splits
from tokenizer.trainer.common import (
    create_bpe_tokenizer,
    save_bpe_checkpoint,
    superbpe_extend_available,
    train_bpe_stage,
    train_superbpe_stage2,
)

CHECKPOINT_DIR = Path(".validate_superbpe_ckpt")
SAMPLE_TEXT = "भारत में लोग रहते हैं।"
TRAIN_PHRASE = "भारत में लोग रहते हैं"


def token_count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def read_merge_lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def split_merge_pair(line: str) -> tuple[str, str]:
    left, _, right = line.partition(" ")
    return left, right


def is_cross_word_merge(line: str) -> bool:
    """True when a merge rule touches a whitespace word boundary."""
    left, right = split_merge_pair(line)
    if left == " " or right == " ":
        return True
    return " " in left or " " in right


def cross_word_vocab_tokens(stage1_vocab: set[str], stage2_vocab: set[str]) -> list[str]:
    return sorted(token for token in stage2_vocab - stage1_vocab if " " in token)


def validate_pretokenization() -> None:
    sub = [p for p, _ in describe_splits(SAMPLE_TEXT, stage="subword", use_script_norm=False)]
    sup = [p for p, _ in describe_splits(SAMPLE_TEXT, stage="superword", use_script_norm=False)]

    print("=== Pretokenization ===")
    print(f"  stage1 chunks ({len(sub)}): {sub}")
    print(f"  stage2 chunks ({len(sup)}): {sup}")

    assert len(sub) > 1, "stage1 should split words"
    assert len(sup) == 1 and sup[0] == SAMPLE_TEXT, "stage2 should not split words"
    assert build_pre_tokenizer("superword") is None
    print("  PASS")


def validate_merge_extension() -> None:
    print("\n=== Merge extension (patched tokenizers library) ===")

    if not superbpe_extend_available():
        print("  FAIL — stock huggingface/tokenizers detected")
        print("  Run: uv sync  (pyproject.toml pins the vendored patch)")
        raise SystemExit(1)

    corpus = [TRAIN_PHRASE] * 200
    factory = lambda: iter(corpus)

    stage1_tokenizer = create_bpe_tokenizer(use_nfkc=False)
    train_bpe_stage(
        stage1_tokenizer,
        factory,
        pretokenization_stage="subword",
        vocab_size=128,
        min_frequency=1,
        stage_name="validate-stage1",
    )

    stage1_vocab = set(stage1_tokenizer.get_vocab())
    stage1_merges = read_merge_lines(save_bpe_checkpoint(stage1_tokenizer, CHECKPOINT_DIR) / "merges.txt")

    stage1_counts = {text: token_count(stage1_tokenizer, text) for text in {TRAIN_PHRASE, SAMPLE_TEXT}}
    print(f"  stage1 token counts: {stage1_counts}")

    stage2_tokenizer = train_superbpe_stage2(
        CHECKPOINT_DIR,
        factory,
        vocab_size=256,
        min_frequency=1,
        use_nfkc=False,
    )

    stage2_vocab = set(stage2_tokenizer.get_vocab())
    stage2_ckpt = CHECKPOINT_DIR / "stage2"
    stage2_merges = read_merge_lines(save_bpe_checkpoint(stage2_tokenizer, stage2_ckpt) / "merges.txt")
    new_merge_lines = stage2_merges[len(stage1_merges) :]

    stage2_counts = {text: token_count(stage2_tokenizer, text) for text in {TRAIN_PHRASE, SAMPLE_TEXT}}
    print(f"  stage2 token counts: {stage2_counts}")

    assert stage1_vocab.issubset(stage2_vocab), "stage1 tokens must survive stage2"
    assert len(stage2_vocab) > len(stage1_vocab), "stage2 should grow the vocabulary"

    for text, before in stage1_counts.items():
        after = stage2_counts[text]
        assert after < before, (
            f"stage2 should compress {text!r} ({before} tokens -> {after} tokens)"
        )

    cross_word_merges = [line for line in new_merge_lines if is_cross_word_merge(line)]
    assert cross_word_merges, (
        "stage2 should append merge rules that cross whitespace boundaries; "
        f"got {len(new_merge_lines)} new merges, none cross-word"
    )

    superwords = cross_word_vocab_tokens(stage1_vocab, stage2_vocab)
    assert superwords, "stage2 should add multi-word vocabulary entries"

    train_reduction = 1.0 - (stage2_counts[TRAIN_PHRASE] / stage1_counts[TRAIN_PHRASE])
    sample_reduction = 1.0 - (stage2_counts[SAMPLE_TEXT] / stage1_counts[SAMPLE_TEXT])

    print(f"  vocab growth: {len(stage1_vocab)} -> {len(stage2_vocab)}")
    print(
        "  compression: "
        f"train phrase {stage1_counts[TRAIN_PHRASE]} -> {stage2_counts[TRAIN_PHRASE]} "
        f"({train_reduction:.1%}), "
        f"sample {stage1_counts[SAMPLE_TEXT]} -> {stage2_counts[SAMPLE_TEXT]} "
        f"({sample_reduction:.1%})"
    )
    print(f"  new cross-word merges: {len(cross_word_merges)} / {len(new_merge_lines)}")
    print(f"  merge examples: {cross_word_merges[:5]}")
    print(f"  superword examples: {superwords[:5]}")
    print("  PASS")


def main() -> None:
    validate_pretokenization()
    validate_merge_extension()


if __name__ == "__main__":
    main()
