"""Download Hindi Sangrah parquet shards from ai4bharat/sangraha."""

from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from constants import HINDI_LANG3

DEFAULT_REPO = "ai4bharat/sangraha"
DEFAULT_REV = "main"
VERIFIED_DATA_RE = re.compile(
    rf"^verified/{re.escape(HINDI_LANG3)}/data-.+\.parquet$"
)
PARQUET_RE = re.compile(r"^.+\.parquet$")

# HuggingFace uses script tags for synthetic Hindi (not hin/).
HINDI_SPLIT_PATHS: dict[str, list[str]] = {
    "verified": [f"verified/{HINDI_LANG3}"],
    "unverified": [f"unverified/{HINDI_LANG3}"],
    "synthetic": ["synthetic/hin_Deva", "synthetic/hin_Latn"],
}
ALL_SPLITS = tuple(HINDI_SPLIT_PATHS)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def list_hindi_parquet_paths(repo_id: str, revision: str) -> list[str]:
    return list_parquet_paths(repo_id, revision, f"verified/{HINDI_LANG3}")


def list_parquet_paths(repo_id: str, revision: str, path_in_repo: str) -> list[str]:
    api = HfApi()
    tree = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        path_in_repo=path_in_repo,
        recursive=True,
    )
    return sorted(
        entry.path
        for entry in tree
        if PARQUET_RE.match(entry.path)
    )


def estimate_hindi_bytes(
    *,
    splits: tuple[str, ...] = ALL_SPLITS,
    repo_id: str = DEFAULT_REPO,
    revision: str = DEFAULT_REV,
) -> dict[str, int]:
    api = HfApi()
    sizes: dict[str, int] = {}
    for split in splits:
        split_bytes = 0
        for path_in_repo in HINDI_SPLIT_PATHS[split]:
            tree = api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                path_in_repo=path_in_repo,
                recursive=True,
            )
            for entry in tree:
                if PARQUET_RE.match(entry.path):
                    split_bytes += getattr(entry, "size", 0) or 0
        sizes[split] = split_bytes
    return sizes


def format_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1e9:.2f} GB"


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


def download_paths(
    paths: list[str],
    *,
    repo_id: str,
    revision: str,
    out_dir: Path,
    desc: str,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for rel_path in tqdm(paths, desc=desc, unit="shard"):
        download_shard(
            rel_path=rel_path,
            repo_id=repo_id,
            revision=revision,
            out_dir=out_dir,
        )
        downloaded.append(rel_path)
    return downloaded


def download_hindi_splits(
    *,
    splits: tuple[str, ...] = ALL_SPLITS,
    repo_id: str = DEFAULT_REPO,
    revision: str = DEFAULT_REV,
    out_dir: Path | None = None,
) -> dict[str, list[str]]:
    out_dir = out_dir or repo_root() / "data" / "sangrah_dataset"
    downloaded: dict[str, list[str]] = {}

    for split in splits:
        split_paths: list[str] = []
        for path_in_repo in HINDI_SPLIT_PATHS[split]:
            paths = list_parquet_paths(repo_id, revision, path_in_repo)
            if not paths:
                raise ValueError(f"No parquet shards found under {path_in_repo}/")
            split_paths.extend(paths)

        downloaded[split] = download_paths(
            split_paths,
            repo_id=repo_id,
            revision=revision,
            out_dir=out_dir,
            desc=f"{split}/hin",
        )

    return downloaded


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


def check_free_space(path: Path, required_bytes: int) -> int:
    path.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(path).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Need at least {format_gb(required_bytes)} free at {path.resolve()}, "
            f"but only {format_gb(free_bytes)} available."
        )
    return free_bytes


def iter_local_shards(train_dir: Path, split: str) -> list[Path]:
    shards: list[Path] = []
    for path_in_repo in HINDI_SPLIT_PATHS[split]:
        rel = Path(path_in_repo)
        shard_dir = train_dir / rel
        if shard_dir.is_dir():
            shards.extend(sorted(shard_dir.glob("*.parquet")))
    return shards


def holdout_eval_shards(
    *,
    train_dir: Path | None = None,
    eval_dir: Path | None = None,
    eval_count: int = 2,
    seed: int = 42,
    splits: tuple[str, ...] = ("verified",),
) -> dict[str, list[Path]]:
    """Move random shards from train_dir to eval_dir (no train/eval leakage)."""
    if eval_count < 1:
        raise ValueError(f"eval_count={eval_count} must be at least 1")

    root = repo_root()
    train_dir = train_dir or root / "data" / "sangrah_dataset"
    eval_dir = eval_dir or root / "data" / "eval" / "hi"

    rng = random.Random(seed)
    moved: dict[str, list[Path]] = {}

    for split in splits:
        shards = iter_local_shards(train_dir, split)
        if not shards:
            raise FileNotFoundError(
                f"No local parquet shards for split={split!r} under {train_dir.resolve()}"
            )
        if len(shards) <= eval_count:
            raise ValueError(
                f"split={split!r} has {len(shards)} shard(s) but eval_count={eval_count}; "
                "leave at least one shard for training."
            )

        chosen = sorted(rng.sample(shards, eval_count))
        moved[split] = []
        for src in chosen:
            rel = src.relative_to(train_dir)
            dst = eval_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                raise FileExistsError(f"Eval shard already exists: {dst}")
            shutil.move(src, dst)
            moved[split].append(dst)

    manifest = eval_dir / "holdout_manifest.txt"
    lines = [
        f"seed={seed}",
        f"eval_count_per_split={eval_count}",
        f"splits={' '.join(splits)}",
        "",
    ]
    for split, paths in moved.items():
        lines.append(f"[{split}]")
        lines.extend(f"  {path.relative_to(eval_dir)}" for path in paths)
        lines.append("")
    manifest.write_text("\n".join(lines), encoding="utf-8")

    return moved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Hindi Sangrah parquet shards from ai4bharat/sangraha.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download every Hindi shard for the selected splits (verified/unverified/synthetic).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=ALL_SPLITS,
        default=list(ALL_SPLITS),
        help="Splits to download when --full is set.",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Print shard counts and sizes, then exit.",
    )
    parser.add_argument(
        "--check-space",
        action="store_true",
        help="Verify free disk space for the selected splits before downloading.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root (default: <repo>/data/sangrah_dataset).",
    )
    parser.add_argument(
        "--headroom-gb",
        type=float,
        default=10.0,
        help="Extra GB to require beyond the estimated download size.",
    )
    parser.add_argument("--count", type=int, default=20, help="Total shards to download.")
    parser.add_argument(
        "--eval-count",
        type=int,
        default=2,
        help="Shards reserved for eval holdout (moved to data/eval/hi/ per split).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for shard selection.")
    parser.add_argument("--no-clean", action="store_true", help="Skip deleting existing data/.")
    parser.add_argument(
        "--holdout-eval",
        action="store_true",
        help="Move random shards from train data to data/eval/hi/ (run after --full).",
    )
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        choices=ALL_SPLITS,
        default=["verified"],
        help="Splits to draw eval holdout shards from (default: verified only).",
    )
    args = parser.parse_args()

    root = repo_root()
    out_dir = args.out_dir or root / "data" / "sangrah_dataset"
    eval_dir = root / "data" / "eval" / "hi"
    splits = tuple(dict.fromkeys(args.splits))
    eval_splits = tuple(dict.fromkeys(args.eval_splits))

    if args.holdout_eval and not args.full and not args.estimate:
        moved = holdout_eval_shards(
            train_dir=out_dir,
            eval_dir=eval_dir,
            eval_count=args.eval_count,
            seed=args.seed,
            splits=eval_splits,
        )
        print(f"Eval holdout -> {eval_dir.resolve()}")
        for split, paths in moved.items():
            print(f"  {split}: {len(paths)} shard(s)")
            for path in paths:
                print(f"    {path.relative_to(eval_dir)}")
        print(f"\nManifest: {eval_dir / 'holdout_manifest.txt'}")
        return

    if args.estimate or args.check_space:
        sizes = estimate_hindi_bytes(splits=splits)
        total = sum(sizes.values())
        print("Hindi Sangrah size estimate (ai4bharat/sangraha):")
        for split, nbytes in sizes.items():
            paths: list[str] = []
            for path_in_repo in HINDI_SPLIT_PATHS[split]:
                paths.extend(list_parquet_paths(DEFAULT_REPO, DEFAULT_REV, path_in_repo))
            print(f"  {split:11s}: {len(paths):3d} shards, {format_gb(nbytes)}")
            for path_in_repo in HINDI_SPLIT_PATHS[split]:
                subpaths = list_parquet_paths(DEFAULT_REPO, DEFAULT_REV, path_in_repo)
                if subpaths:
                    print(f"    {path_in_repo}: {len(subpaths)} shards")
        print(f"  {'total':11s}: {format_gb(total)} (+ {args.headroom_gb:.0f} GB headroom recommended)")

        if args.check_space:
            required = int(total + args.headroom_gb * 1e9)
            free = check_free_space(out_dir.parent if out_dir.name == "sangrah_dataset" else out_dir, required)
            print(f"\nOK: {format_gb(free)} free at {out_dir.resolve().parent}")

        if args.estimate and not args.full:
            return

    if args.full:
        sizes = estimate_hindi_bytes(splits=splits)
        required = int(sum(sizes.values()) + args.headroom_gb * 1e9)
        check_free_space(out_dir.parent if out_dir.name == "sangrah_dataset" else out_dir, required)

        downloaded = download_hindi_splits(splits=splits, out_dir=out_dir)
        print(f"\nDownloaded Hindi Sangrah -> {out_dir.resolve()}")
        for split, paths in downloaded.items():
            print(f"  {split}: {len(paths)} shards")

        moved = holdout_eval_shards(
            train_dir=out_dir,
            eval_dir=eval_dir,
            eval_count=args.eval_count,
            seed=args.seed,
            splits=eval_splits,
        )
        print(f"\nEval holdout -> {eval_dir.resolve()}")
        for split, paths in moved.items():
            print(f"  {split}: {len(paths)} shard(s)")
            for path in paths:
                print(f"    {path.relative_to(eval_dir)}")
        return

    if not args.no_clean:
        clear_data_dirs()

    train_paths, eval_paths = download_hindi_shards(
        count=args.count,
        eval_count=args.eval_count,
        seed=args.seed,
        train_dir=out_dir,
        eval_dir=root / "data" / "eval" / "hi",
    )

    train_dir = out_dir
    eval_dir = root / "data" / "eval" / "hi"

    print(f"Training ({len(train_paths)} shards) -> {train_dir.resolve()}")
    for rel_path in train_paths:
        print(f"  {rel_path}")

    print(f"\nEval holdout ({len(eval_paths)} shards) -> {eval_dir.resolve()}")
    for rel_path in eval_paths:
        print(f"  {rel_path}")


if __name__ == "__main__":
    main()
