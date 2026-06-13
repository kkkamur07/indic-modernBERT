"""Shared language constants for Hindi-only tokenizer work."""

HINDI_LANG3 = "hin"
HINDI_LANG2 = "hi"


def validate_vocab_size(vocab_size: int) -> None:
    if vocab_size % 64 != 0:
        raise ValueError(f"vocab_size={vocab_size} is invalid. Use a value divisible by 64.")
