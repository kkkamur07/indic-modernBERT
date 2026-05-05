"""Train a small BPE tokenizer baseline on local Sangraha parquet shards."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterator

import hydra
import pyarrow.parquet as pq
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from loguru import logger
from omegaconf import DictConfig
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

try:
    from ..utils.log_helpers import setup_run_log, slug
except ImportError:  # pragma: no cover - allows direct script execution.
    from tokenizer.utils.log_helpers import setup_run_log, slug

LANG_CODE_MAP = {
    "asm": "as",
    "ben": "bn",
    "eng": "en",
    "guj": "gu",
    "hin": "hi",
    "kan": "kn",
    "mal": "ml",
    "mar": "mr",
    "nep": "ne",
    "ori": "or",
    "pan": "pa",
    "san": "sa",
    "tam": "ta",
    "tel": "te",
    "urd": "ur",
}

CJK_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
EMOJI_AND_SYMBOL_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]"
)

# Helper functions

def _build_text_normalizer(mode: str) -> Callable[[str, str], str]:
    if mode == "none":
        return lambda _lang, text: text

    if mode == "hf_basic":
        return lambda _lang, text: text

    # TODO : Verify from indic_nlp library implementation. 
    if mode == "indic_nlp":
        factory = IndicNormalizerFactory()
        normalizers: dict[str, object] = {}

        for lang3, lang2 in LANG_CODE_MAP.items():
            if lang2 == "en":
                continue
            normalizers[lang3] = factory.get_normalizer(lang2, remove_nuktas=False)

        def _normalize(lang: str, text: str) -> str:
            normalizer = normalizers.get(lang)
            if normalizer is None:
                return text
            return normalizer.normalize(text)

        return _normalize

    raise ValueError(f"Unsupported normalization mode: {mode}")


def _iter_texts(
    data_root: Path,
    text_column: str,
    text_normalizer: Callable[[str, str], str],
) -> Iterator[str]:

    parquet_files = sorted(data_root.glob("verified/*/*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {data_root}")

    for parquet_path in parquet_files:
        lang = parquet_path.parent.name
        table = pq.read_table(parquet_path, columns=[text_column])

        for value in table[text_column].to_pylist():
            if value is None:
                continue
            text = text_normalizer(lang, str(value)).strip()

            if not text:
                continue

            if CJK_RE.search(text) or EMOJI_AND_SYMBOL_RE.search(text):
                continue
            
            yield text


def train_bpe(
    data_root: Path,
    output_dir: Path,
    text_column: str,
    vocab_size: int,
    min_frequency: int, # Minimum frequency of a token to be included in the vocabulary
    normalization: str,

) -> Path:

    if vocab_size % 64 != 0:
        raise ValueError(f"vocab_size={vocab_size} is invalid. Use a value divisible by 64.")

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"

    text_normalizer = _build_text_normalizer(normalization)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    if normalization == "hf_basic":
        tokenizer.normalizer = NFKC()

    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    tokenizer.train_from_iterator(
        _iter_texts(
            data_root,
            text_column=text_column,
            text_normalizer=text_normalizer,
        ),
        trainer=trainer,
    )

    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")

    if cls_id is not None and sep_id is not None:
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
        )

    tokenizer.save(str(tokenizer_path))
    return tokenizer_path


def _setup_run_log(data_root: Path, normalization: str, vocab_sizes: list[int]) -> Path:
    data_tag = slug(data_root.name)
    vocab_tag = "-".join(str(v) for v in vocab_sizes)
    log_name = f"train_bpe__norm-{normalization}__vocabs-{vocab_tag}__data-{data_tag}.log"
    return setup_run_log(log_name)


@hydra.main(version_base=None, config_path="../../../configs", config_name="tokenizer")
def main(cfg: DictConfig) -> None:

    bpe_cfg = cfg.tokenizer.trainer.bpe
    vocab_sizes = [int(v) for v in bpe_cfg.vocab_sizes]
    base_output_dir = Path(bpe_cfg.output_dir)
    data_root = Path(bpe_cfg.data_root)
    normalization = str(bpe_cfg.normalization)
    run_log = _setup_run_log(data_root, normalization, vocab_sizes)

    logger.info(
        "Starting BPE training | data_root={} | normalization={} | vocab_sizes={} | min_freq={}",
        data_root,
        normalization,
        vocab_sizes,
        int(bpe_cfg.min_frequency),
    )

    for vocab_size in vocab_sizes:
        
        output_dir = (
            base_output_dir
            if len(vocab_sizes) == 1
            else base_output_dir.parent / f"{base_output_dir.name}_vs{vocab_size}"
        )

        tokenizer_path = train_bpe(
            data_root=data_root,
            output_dir=output_dir,
            text_column=bpe_cfg.text_column,
            vocab_size=vocab_size,
            min_frequency=int(bpe_cfg.min_frequency),
            normalization=normalization,
        )

        logger.info(
            "Completed vocab_size={} | tokenizer_path={}",
            vocab_size,
            tokenizer_path.resolve(),
        )

    logger.info("Hydra experiment log: {}", run_log.resolve())


if __name__ == "__main__":
    main()
