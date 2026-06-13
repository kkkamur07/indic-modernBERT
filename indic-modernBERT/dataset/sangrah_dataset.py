"""Download random Hindi parquet shards from ai4bharat/sangraha (verified/hin/)."""

from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from constants import HINDI_LANG3

DEFAULT_REPO = "ai4bharat/sangraha"
DEFAULT_REV = "main"
VERIFIED_DATA_RE = re.compile(
    rf"^verified/{re.escape(HINDI_LANG3)}/data-.+\.parquet$"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def list_hindi_parquet_paths(repo_id: str, revision: str) -> list[str]:
    api = HfApi()
    tree = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        path_in_repo=f"verified/{HINDI_LANG3}",
        recursive=True,
    )
    return [entry.path for entry in tree if VERIFIED_DATA_RE.match(entry.path)]


def download_shard(
    *,
    rel_path: str,
    repo_id: str,
    revision: str,
    out_dir: Path,
) -> str:
    return hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        filename=rel_path,
        local_dir=str(out_dir),
    )


def download_hindi_shards(
    *,
    count: int = 20,
    eval_count: int = 2,
    seed: int = 42,
    repo_id: str = DEFAULT_REPO,
    revision: str = DEFAULT_REV,
    train_dir: Path | None = None,
    eval_dir: Path | None = None,
) -> tuple[list[str], list[str]]:

    if eval_count >= count:
        raise ValueError(f"eval_count={eval_count} must be less than count={count}")

    root = repo_root()
    train_dir = train_dir or root / "data" / "sangrah_dataset"
    eval_dir = eval_dir or root / "data" / "eval" / "hi"

    paths = list_hindi_parquet_paths(repo_id, revision)
    if len(paths) < count:
        raise ValueError(
            f"Requested {count} shards but only {len(paths)} available under "
            f"verified/{HINDI_LANG3}/"
        )

    rng = random.Random(seed)
    chosen = rng.sample(sorted(paths), count)
    eval_paths = chosen[:eval_count]
    train_paths = chosen[eval_count:]

    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    for rel_path in train_paths:
        download_shard(
            rel_path=rel_path,
            repo_id=repo_id,
            revision=revision,
            out_dir=train_dir,
        )

    for rel_path in eval_paths:
        download_shard(
            rel_path=rel_path,
            repo_id=repo_id,
            revision=revision,
            out_dir=eval_dir,
        )

    return train_paths, eval_paths


def clear_data_dirs() -> None:
    root = repo_root() / "data"
    for name in ("sangrah_dataset", "eval"):
        target = root / name
        if target.exists():
            shutil.rmtree(target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download random Hindi Sangrah shards for training and eval holdout.",
    )
    parser.add_argument("--count", type=int, default=20, help="Total shards to download.")
    parser.add_argument(
        "--eval-count",
        type=int,
        default=2,
        help="Shards reserved for eval holdout (copied to data/eval/hi/).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for shard selection.")
    parser.add_argument("--no-clean", action="store_true", help="Skip deleting existing data/.")
    args = parser.parse_args()

    if not args.no_clean:
        clear_data_dirs()

    train_paths, eval_paths = download_hindi_shards(
        count=args.count,
        eval_count=args.eval_count,
        seed=args.seed,
    )

    root = repo_root()
    train_dir = root / "data" / "sangrah_dataset"
    eval_dir = root / "data" / "eval" / "hi"

    print(f"Training ({len(train_paths)} shards) -> {train_dir.resolve()}")
    for rel_path in train_paths:
        print(f"  {rel_path}")

    print(f"\nEval holdout ({len(eval_paths)} shards) -> {eval_dir.resolve()}")
    for rel_path in eval_paths:
        print(f"  {rel_path}")


if __name__ == "__main__":
    main()
