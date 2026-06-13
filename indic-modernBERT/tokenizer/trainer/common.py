"""Shared utilities for BPE and SuperBPE tokenizer training."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from constants import HINDI_LANG3, validate_vocab_size
from tokenizer.pretokenization import (
    PretokenizationStage,
    build_pre_tokenizer,
    normalize_text,
)
from utils.log_helpers import log_training_stage

CJK_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
EMOJI_AND_SYMBOL_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]"
)

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
]
UNK_TOKEN = "[UNK]"

SUPERBPE_FORK_INSTALL = (
    "SuperBPE stage 2 needs the vendored tokenizers patch (merge extension in Rust). "
    "Stock huggingface/tokenizers rebuilds the vocabulary when pretokenization changes "
    "and drops stage-1 merges. Re-sync the env: uv sync"
)

_extend_support_cache: bool | None = None


def _tokenizers_package_path() -> Path:
    import tokenizers

    return Path(tokenizers.__file__).resolve()


def _uses_vendored_superbpe_tokenizers() -> bool:
    path = _tokenizers_package_path().as_posix()
    return "tokenizers_superbpe" in path or "_support_repo/superbpe" in path


def _count_merge_lines(merges_path: Path) -> int:
    return sum(
        1
        for line in merges_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )


def _probe_merge_extension() -> bool:
    """Stage 2 should add merges and compress a repeated multi-word Hindi phrase."""
    import tempfile

    phrase = "भारत में लोग रहते हैं"
    corpus = [phrase] * 200

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp)
        stage1 = create_bpe_tokenizer(use_nfkc=False)
        configure_pre_tokenizer(stage1, "subword")
        train_bpe_on_corpus(
            stage1,
            iter(corpus),
            vocab_size=128,
            min_frequency=1,
            show_progress=False,
        )
        stage1_tokens = len(stage1.encode(phrase, add_special_tokens=False).ids)
        merges_before = _count_merge_lines(
            save_bpe_checkpoint(stage1, checkpoint) / "merges.txt"
        )

        stage2 = create_bpe_tokenizer(use_nfkc=False)
        configure_pre_tokenizer(stage2, "superword")
        with _training_cwd(checkpoint):
            train_bpe_on_corpus(
                stage2,
                iter(corpus),
                vocab_size=256,
                min_frequency=1,
                show_progress=False,
            )

        stage2_tokens = len(stage2.encode(phrase, add_special_tokens=False).ids)
        save_bpe_checkpoint(stage2, checkpoint / "stage2")
        merges_after = _count_merge_lines(checkpoint / "stage2" / "merges.txt")

        return merges_after > merges_before and stage2_tokens < stage1_tokens


@contextmanager
def _training_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def superbpe_extend_available() -> bool:
    """True when the vendored tokenizers build can extend ``merges.txt`` in stage 2."""
    global _extend_support_cache
    if _extend_support_cache is not None:
        return _extend_support_cache

    if not _uses_vendored_superbpe_tokenizers():
        _extend_support_cache = False
        return _extend_support_cache

    _extend_support_cache = _probe_merge_extension()
    return _extend_support_cache


def require_superbpe_extend() -> None:
    if not superbpe_extend_available():
        raise RuntimeError(SUPERBPE_FORK_INSTALL)


def save_bpe_checkpoint(tokenizer: Tokenizer, checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.model.save(str(checkpoint_dir))
    merges_path = checkpoint_dir / "merges.txt"
    if not merges_path.exists():
        raise RuntimeError(f"BPE checkpoint missing merges.txt at {merges_path}")
    return checkpoint_dir


def train_superbpe_stage2(
    checkpoint_dir: Path,
    corpus_factory: Callable[[], Iterator[str]],
    *,
    vocab_size: int,
    min_frequency: int,
    use_nfkc: bool = True,
) -> Tokenizer:
    """Extend a stage-1 checkpoint using ``merges.txt`` (SuperBPE stage 2).

    Depends on the vendored ``tokenizers`` fork. Stage 2 changes cwd to the
    checkpoint dir (where ``merges.txt`` lives); config paths must be absolute
    (see ``resolve_from_cwd`` in ``config/schema.py``).

    The fork must parse word-initial merge lines like ``"  क"`` correctly — see
    ``LEARNINGS.md`` (*merges.txt leading-space parsing*) if stage 2 panics or
    logs ``not found in word_to_id`` for blank tokens.
    """
    require_superbpe_extend()
    validate_vocab_size(vocab_size)

    tokenizer = create_bpe_tokenizer(use_nfkc=use_nfkc)
    configure_pre_tokenizer(tokenizer, "superword")
    with _training_cwd(checkpoint_dir):
        train_bpe_on_corpus(
            tokenizer,
            corpus_factory(),
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )
    return tokenizer


def resolve_transition_vocab_size(
    vocab_size: int,
    *,
    transition_vocab_size: int | None,
    transition_fraction: float,
) -> int:
    if transition_vocab_size is not None:
        if transition_vocab_size >= vocab_size:
            raise ValueError(
                f"transition_vocab_size={transition_vocab_size} must be < vocab_size={vocab_size}."
            )
        return transition_vocab_size

    return compute_transition_vocab_size(vocab_size, transition_fraction)


# So that we train on a multiple of 64 - for compute efficiency.
def compute_transition_vocab_size(
    vocab_size: int,
    transition_fraction: float,
) -> int:

    if not 0.0 < transition_fraction < 1.0:

        raise ValueError(
            f"transition_fraction={transition_fraction} must be between 0 and 1."
        )

    transition_vocab_size = int(vocab_size * transition_fraction)
    transition_vocab_size = (transition_vocab_size // 64) * 64

    if transition_vocab_size <= 0:
        transition_vocab_size = 64

    if transition_vocab_size >= vocab_size:
        transition_vocab_size = vocab_size - 64

    if transition_vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError(
            f"transition_vocab_size={transition_vocab_size} is too small for vocab_size={vocab_size}."
        )

    return transition_vocab_size


def make_corpus_iterator(
    data_root: Path,
    text_column: str,
    *,
    use_script_norm: bool = True,
) -> Callable[[], Iterator[str]]:
    """Return a factory that yields a fresh corpus iterator for each training pass."""

    def fresh_iterator() -> Iterator[str]:
        return iter_corpus_texts(
            data_root,
            text_column,
            use_script_norm=use_script_norm,
        )

    return fresh_iterator


def iter_corpus_texts(
    data_root: Path,
    text_column: str,
    *,
    use_script_norm: bool = True,
    filter_cjk_emoji: bool = True,
) -> Iterator[str]:

    parquet_files = sorted(data_root.glob(f"verified/{HINDI_LANG3}/*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Hindi parquet files found under: {data_root}/verified/{HINDI_LANG3}"
        )

    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path, columns=[text_column])

        #! Check this for efficiency, because some methods are faster. 
        for value in table[text_column].to_pylist():
            if value is None:
                continue

            # Script norm runs here (Python/indic-nlp). NFKC is applied by
            # tokenizer.normalizer during train_from_iterator, not in this loop.
            text = normalize_text(
                str(value),
                use_script_norm=use_script_norm,
                use_nfkc=False,
            ).strip()

            if not text:
                continue

            if filter_cjk_emoji and (
                CJK_RE.search(text) or EMOJI_AND_SYMBOL_RE.search(text)
            ):
                continue

            yield text


def create_bpe_tokenizer(*, use_nfkc: bool = True) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

    # For unicode normalization.
    if use_nfkc:
        tokenizer.normalizer = NFKC()

    return tokenizer


def train_bpe_on_corpus(
    tokenizer: Tokenizer,
    texts: Iterator[str],
    *,
    vocab_size: int,
    min_frequency: int,
    show_progress: bool = True,
) -> None:
    validate_vocab_size(vocab_size)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=show_progress,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)


def train_bpe_stage(
    tokenizer: Tokenizer,
    corpus_factory: Callable[[], Iterator[str]],
    *,
    pretokenization_stage: PretokenizationStage,
    vocab_size: int,
    min_frequency: int,
    stage_name: str,
) -> None:
    configure_pre_tokenizer(tokenizer, pretokenization_stage)
    train_bpe_on_corpus(
        tokenizer,
        corpus_factory(),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )
    log_training_stage(
        stage_name=stage_name,
        pretokenization_stage=pretokenization_stage,
        target_vocab_size=vocab_size,
        current_vocab_size=tokenizer.get_vocab_size(),
    )


def configure_pre_tokenizer(
    tokenizer: Tokenizer,
    stage: PretokenizationStage,
) -> None:

    tokenizer.pre_tokenizer = build_pre_tokenizer(stage)


def attach_cls_sep_processor(tokenizer: Tokenizer) -> None:
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")

    if cls_id is None or sep_id is None:
        return

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )


def construct_hf_tokenizer(tokenizer_dir: Path):
    """Export a HuggingFace ``PreTrainedTokenizerFast`` alongside ``tokenizer.json``."""
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    tokenizer_dir = Path(tokenizer_dir)
    base_tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base_tokenizer,
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[CLS]",
        eos_token="[SEP]",
    )
    hf_tokenizer.save_pretrained(tokenizer_dir)
    return hf_tokenizer


def save_tokenizer(
    tokenizer: Tokenizer,
    output_dir: Path,
    *,
    export_hf: bool = True,
) -> Path:

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    if export_hf:
        construct_hf_tokenizer(output_dir)

    return tokenizer_path
