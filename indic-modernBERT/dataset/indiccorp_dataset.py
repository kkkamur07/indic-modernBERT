"""Convert IndicCorp V2 Hindi ``.txt`` dumps into Sangraha-compatible parquet.

IndicCorp V2 (``ai4bharat/IndicCorpV2``) ships Hindi as plain text files
(``hi-1.txt``, ``hi-2.txt``, ``hi-3.txt``). Each document is a single
non-empty line; blank lines are pure separators (verified: max run of
consecutive non-empty lines is 1). The MLM/BPE pipeline here consumes parquet
shards with the same schema as Sangraha ``verified/hin``:

    doc_id: large_string   # sha1 of the text (matches Sangraha)
    text:   large_string   # the document (one source line)
    type:   large_string   # provenance tag, "indiccorp_v2"

The files are streamed line by line, so a full ~25 GB file never lands in
memory. The only buffer is the current shard's rows (``ROWS_PER_SHARD``
documents), which is flushed to disk and cleared.

Edit the config block below, then run from the repo root (PYTHONPATH includes
``indic-modernBERT`` via the Makefile / ``PYTHONPATH=indic-modernBERT``):

    uv run python -m dataset.indiccorp_dataset
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from constants import HINDI_LANG2


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Conversion config — set values here and run; no CLI flags.
# --------------------------------------------------------------------------- #
# Inputs: .txt files and/or directories holding hi-*.txt.
INPUT_PATHS: list[Path] = [repo_root() / "data" / "indiccorp_v2" / "raw" / "data"]
OUT_DIR: Path = repo_root() / "data" / "indiccorp_v2" / "parquet" / HINDI_LANG2

# Documents per shard. Hindi docs average ~751 utf-8 bytes (median 549, p99
# 4334), so 200k docs ≈ 150 MB uncompressed text per shard — in line with the
# Sangraha verified shards (~175k rows) and ~50-70 MB on disk after zstd.
ROWS_PER_SHARD: int = 200_000

DOC_TYPE: str = "indiccorp_v2"
COMPRESSION: str = "zstd"
OVERWRITE: bool = True
# Set to an int to stop early per file (smoke test); None converts everything.
LIMIT_DOCS: int | None = None

# Match the Sangraha verified/hin schema exactly so downstream readers that
# select the "text" column behave identically across corpora.
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.large_string()),
        pa.field("text", pa.large_string()),
        pa.field("type", pa.large_string()),
    ]
)


def doc_id_for(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def resolve_inputs(input_paths: list[Path]) -> list[Path]:
    """Expand files/directories into a sorted list of ``.txt`` files."""
    files: list[Path] = []
    for path in input_paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.txt")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")

    if not files:
        raise FileNotFoundError(f"No .txt files found under: {input_paths}")
    return files


def iter_documents(txt_path: Path) -> Iterator[tuple[str, int]]:
    """Yield ``(document, raw_byte_len)`` per non-empty line.

    O(1) memory: holds only the current line, never a list of lines. The byte
    length is the encoded size of the raw line (incl. newline) so the caller
    can drive a byte-accurate progress bar against the file size.
    """
    with txt_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            nbytes = len(raw.encode("utf-8"))
            text = raw.strip()
            if text:
                yield text, nbytes


def write_shard(rows: list[dict[str, str]], out_path: Path, *, compression: str) -> None:
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    pq.write_table(table, out_path, compression=compression)


def convert(
    *,
    input_paths: list[Path] = INPUT_PATHS,
    out_dir: Path = OUT_DIR,
    rows_per_shard: int = ROWS_PER_SHARD,
    doc_type: str = DOC_TYPE,
    compression: str = COMPRESSION,
    limit_docs: int | None = LIMIT_DOCS,
    overwrite: bool = OVERWRITE,
) -> list[Path]:
    files = resolve_inputs(input_paths)

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.parquet"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{len(existing)} parquet shard(s) already in {out_dir.resolve()}. "
            "Set OVERWRITE = True to replace them."
        )
    for stale in existing:
        stale.unlink()

    written: list[Path] = []
    total_docs = 0

    for txt_path in files:
        stem = txt_path.stem  # e.g. "hi-1"
        buffer: list[dict[str, str]] = []
        shard_idx = 0
        seen_docs = 0

        progress = tqdm(
            total=txt_path.stat().st_size,
            desc=stem,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )
        for text, nbytes in iter_documents(txt_path):
            progress.update(nbytes)
            buffer.append({"doc_id": doc_id_for(text), "text": text, "type": doc_type})
            seen_docs += 1
            total_docs += 1

            # Flush on whole-document boundaries: a document is never split
            # across shards, so words/text are never broken.
            if len(buffer) >= rows_per_shard:
                out_path = out_dir / f"{stem}-{shard_idx:05d}.parquet"
                write_shard(buffer, out_path, compression=compression)
                written.append(out_path)
                buffer.clear()
                shard_idx += 1
                progress.set_postfix(shards=shard_idx, docs=seen_docs)

            if limit_docs is not None and seen_docs >= limit_docs:
                break

        if buffer:
            out_path = out_dir / f"{stem}-{shard_idx:05d}.parquet"
            write_shard(buffer, out_path, compression=compression)
            written.append(out_path)
            buffer.clear()

        progress.close()

    print(f"\nWrote {len(written)} shard(s), {total_docs} document(s) -> {out_dir.resolve()}")
    return written


def main() -> None:
    convert()


if __name__ == "__main__":
    main()
