"""Demonstrate Hindi pre-tokenization splits."""

from __future__ import annotations

from .pipeline import PretokenizationStage, apply_script_normalization, describe_splits

"""

Goals : 
1. We need the punctuation and symbols to be separate from the words. 
2. We need the script normalization to be applied to the text, so that similar script characters are merged together. 
3. Numeric characters should be split into separate tokens. 
4. Semantic units should stay together. 
"""

_EXAMPLES = [
    "मूल्य ₹1,234.50 है।",
    "भारत में १२३ लोग रहते हैं।",
    "Hello, world 123!",
]

_SCRIPT_NORM_EXAMPLE = "क\u200dष"


def _print_example(text: str, stage: PretokenizationStage) -> None:
    splits = describe_splits(text, stage=stage)
    print(f"  input : {text!r}")
    print(f"  chunks: {[piece for piece, _ in splits]}")

    for piece, (start, end) in splits:
        print(f"[{start}:{end}] {piece!r}")


def main() -> None:
    print("=== script normalization example ===")
    print(f"  input : {_SCRIPT_NORM_EXAMPLE!r}")
    print(f"  output: {apply_script_normalization(_SCRIPT_NORM_EXAMPLE)!r}")
    print()

    for stage in ("subword", "superword"):
        print(f"=== {stage} stage ===")
        for text in _EXAMPLES:
            _print_example(text, stage)
            print()


if __name__ == "__main__":
    main()
