"""Regex patterns for Hindi-aware pre-tokenization."""

SUBWORD_SPLIT_PATTERN = (
    # English contractions; low impact for Hindi, but keeps GPT-style behavior.
    r"(?:'s|'t|'re|'ve|'m|'ll|'d)"
    # Letter runs with combining marks, so Devanagari matras/halant stay attached.
    r"|[^\r\n\p{L}\p{M}\p{N}]?[\p{L}\p{M}]+"
    # Digit chunks capped at three characters.
    r"|\p{N}{1,3}"
    # Punctuation and symbols, optionally with one leading space.
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    # Newline runs.
    r"|\s*[\r\n]+"
    # Trailing whitespace.
    r"|\s+(?!\S)"
    # Other whitespace.
    r"|\s+"
)

SUPERWORD_SPLIT_PATTERN = (
    # Relaxed word chunks for SuperBPE phase 2.
    r" ?[\p{L}\p{M}]+"
    # Symbol runs remain separate from words.
    r"| ?[^\s\p{L}\p{M}\p{N}]+"
    # Trailing whitespace.
    r"|\s+(?!\S)"
    # Other whitespace.
    r"|\s+"
)
