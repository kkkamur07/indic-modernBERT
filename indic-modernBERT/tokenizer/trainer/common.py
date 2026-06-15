"""Shared utilities for Hindi BPE tokenizer training."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
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
    preprocess_for_tokenizer,
)
from utils.log_helpers import log_training_stage
from utils.progress import iter_with_progress

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


def iter_corpus_texts(
    data_root: Path,
    text_column: str,
    *,
    use_script_norm: bool = True,
    filter_cjk_emoji: bool = True,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> Iterator[str]:
    parquet_files = sorted(data_root.glob(f"verified/{HINDI_LANG3}/*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Hindi parquet files found under: {data_root}/verified/{HINDI_LANG3}"
        )

    label = progress_desc or "Corpus"
    shard_total = len(parquet_files)

    for shard_idx, parquet_path in enumerate(parquet_files, start=1):
        table = pq.read_table(parquet_path, columns=[text_column])
        column = table[text_column]
        values = column.to_pylist()
        desc = f"{label} | {shard_idx}/{shard_total} {parquet_path.name}"
        row_iter = iter_with_progress(
            values,
            total=len(column),
            desc=desc,
            show_progress=show_progress,
        )

        for value in row_iter:
            if value is None:
                continue

            text = preprocess_for_tokenizer(
                str(value),
                use_script_norm=use_script_norm,
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

    tokenizer_config_path = tokenizer_dir / "tokenizer_config.json"
    if tokenizer_config_path.is_file():
        import json

        with tokenizer_config_path.open(encoding="utf-8") as f:
            tokenizer_config = json.load(f)
        tokenizer_config["preprocess_script_normalization"] = True
        tokenizer_config["preprocess_script_normalization_note"] = (
            "Apply tokenizer.pretokenization.preprocess_for_tokenizer() before "
            "encode; NFKC is already in tokenizer.json."
        )
        with tokenizer_config_path.open("w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
            f.write("\n")

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
