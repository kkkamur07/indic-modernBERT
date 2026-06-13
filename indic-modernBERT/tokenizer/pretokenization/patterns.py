"""Regex patterns for Hindi-aware pre-tokenization."""

# Stage 1 (subword): full semantic splitting — words, digits, punctuation, whitespace.
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

# Stage 2 (superword): no pattern — BPE sees the full normalized string and learns
# merges across former stage-1 boundaries (spaces, punctuation gaps, etc.).
# Reference SuperBPE uses only a Western thousands-separator lookahead; we omit
# even that for Hindi-first training so nothing re-splits the text before BPE.
