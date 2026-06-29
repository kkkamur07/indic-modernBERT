"""Prepare fixed local Hindi mMARCO triplet splits.

The Hugging Face ``unicamp-dl/mmarco`` builder loads the full collection/query
maps before yielding its first text triplet. For fixed 100k Optuna subsets and
1.25M full-budget DPR runs, that is unnecessarily slow and can make each run see
different streamed rows. This script downloads the raw TSV files once, samples
triplet IDs locally, then resolves only the IDs needed for the local JSONL split.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASET_REPO = "unicamp-dl/mmarco"
COLLECTION_PATH = "data/google/collections/hindi_collection.tsv"
QUERIES_PATH = "data/google/queries/train/hindi_queries.train.tsv"
TRIPLES_PATH = "data/triples.train.ids.small.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=100_000)
    parser.add_argument("--eval-samples", type=int, default=1_000)
    # Reservoir-sample from the first N ID triples. The ID triple file is much
    # smaller to scan than the translated collection, and this gives enough
    # randomness without scanning all 39.8M rows for every regeneration.
    parser.add_argument("--candidate-triples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument(
        "--download-dir",
        default="artifacts/retrieval_finetune/hi/raw/unicamp-dl_mmarco",
        help="Directory for the downloaded raw mMARCO TSV files.",
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/retrieval_finetune/hi/subsets/"
            "mmarco_hindi_train100k_eval1k_seed17.jsonl"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    total = args.train_samples + args.eval_samples
    rng = random.Random(args.seed)

    if output.exists() and not args.overwrite:
        rows = _count_lines(output)
        if rows == total:
            print(f"Subset already exists: {output} ({rows} rows)")
            print("Use --overwrite to regenerate it.")
            return
        print(
            f"Existing subset is incomplete ({rows}/{total} rows); regenerating."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    download_dir = Path(args.download_dir)
    triples_path = _download_dataset_file(TRIPLES_PATH, download_dir)
    queries_path = _download_dataset_file(QUERIES_PATH, download_dir)
    collection_path = _download_dataset_file(COLLECTION_PATH, download_dir)

    print(
        f"Sampling {total} ID triples from first {args.candidate_triples} rows",
        flush=True,
    )
    triples = _sample_triples(
        triples_path,
        total=total,
        candidate_triples=args.candidate_triples,
        rng=rng,
        progress_every=args.progress_every,
    )
    qids = {qid for qid, _, _ in triples}
    pids = {pid for _, pos_id, neg_id in triples for pid in (pos_id, neg_id)}

    print(f"Resolving {len(qids)} queries", flush=True)
    queries = _resolve_tsv_ids(
        queries_path,
        wanted_ids=qids,
        label="queries",
        progress_every=args.progress_every,
    )

    print(f"Resolving {len(pids)} passages from Hindi collection", flush=True)
    collection = _resolve_tsv_ids(
        collection_path,
        wanted_ids=pids,
        label="collection",
        progress_every=args.progress_every,
    )

    written = 0
    with tmp_output.open("w", encoding="utf-8") as handle:
        for qid, pos_id, neg_id in triples:
            payload = {
                "query": queries[qid],
                "positive": collection[pos_id],
                "negative": collection[neg_id],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    if written != total:
        raise RuntimeError(f"Expected to write {total} rows, wrote {written}")

    tmp_output.replace(output)
    print(f"Wrote {written} rows to {output}")
    print(f"Train rows: {args.train_samples}")
    print(f"Eval rows: {args.eval_samples}")


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _download_dataset_file(filename: str, download_dir: Path) -> Path:
    path = download_dir / filename
    if path.exists():
        print(f"Using downloaded file: {path}", flush=True)
        return path

    print(f"Downloading {DATASET_REPO}/{filename} to {download_dir}", flush=True)
    downloaded = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=filename,
        repo_type="dataset",
        local_dir=download_dir,
    )
    return Path(downloaded)


def _iter_file_lines(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line:
                yield line.rstrip("\n")


def _sample_triples(
    path: Path,
    *,
    total: int,
    candidate_triples: int,
    rng: random.Random,
    progress_every: int,
) -> list[tuple[str, str, str]]:
    if candidate_triples < total:
        raise ValueError("candidate_triples must be >= train_samples + eval_samples")

    reservoir: list[tuple[str, str, str]] = []
    seen = 0
    for line in _iter_file_lines(path):
        if seen >= candidate_triples:
            break
        parts = line.rstrip().split("\t")
        if len(parts) != 3:
            raise ValueError(f"Malformed triple line: {line[:200]}")
        triple = (parts[0], parts[1], parts[2])

        if len(reservoir) < total:
            reservoir.append(triple)
        else:
            replacement_idx = rng.randint(0, seen)
            if replacement_idx < total:
                reservoir[replacement_idx] = triple

        seen += 1
        if progress_every > 0 and seen % progress_every == 0:
            print(f"Scanned {seen}/{candidate_triples} ID triples", flush=True)

    if len(reservoir) != total:
        raise RuntimeError(f"Expected {total} sampled triples, got {len(reservoir)}")

    rng.shuffle(reservoir)
    return reservoir


def _resolve_tsv_ids(
    path: Path,
    *,
    wanted_ids: set[str],
    label: str,
    progress_every: int,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    scanned = 0

    for line in _iter_file_lines(path):
        scanned += 1
        if "\t" not in line:
            continue
        row_id, text = line.rstrip().split("\t", 1)
        if row_id in wanted_ids:
            resolved[row_id] = text
            if len(resolved) == len(wanted_ids):
                print(
                    f"Resolved all {label}: {len(resolved)}/{len(wanted_ids)} "
                    f"after scanning {scanned} rows",
                    flush=True,
                )
                return resolved

        if progress_every > 0 and scanned % progress_every == 0:
            print(
                f"Scanned {scanned} {label} rows; "
                f"resolved {len(resolved)}/{len(wanted_ids)}",
                flush=True,
            )

    missing = sorted(wanted_ids - resolved.keys())
    raise RuntimeError(
        f"Could not resolve {len(missing)} {label} IDs; first missing: {missing[:5]}"
    )


if __name__ == "__main__":
    main()
