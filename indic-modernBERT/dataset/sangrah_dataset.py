"""Download one random parquet shard per language from ai4bharat/sangraha (verified/)."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

DEFAULT_REPO = "ai4bharat/sangraha"
DEFAULT_REV = "main"
VERIFIED_DATA_RE = re.compile(r"^verified/(?P<lang>[^/]+)/data-.+\.parquet$")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def list_verified_parquet_paths(
    repo_id: str,
    revision: str,
) -> list[str]:

    api = HfApi()
    tree = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        path_in_repo="verified",
        recursive=True,
    )
    return [entry.path for entry in tree]


def filter_paths_by_min_shards(
    paths: list[str],
    min_data_files: int,
) -> tuple[list[str], Counter[str], list[str]]:

    counts_all: Counter[str] = Counter()
    for p in paths:
        m = VERIFIED_DATA_RE.match(p)
        if m:
            counts_all[m["lang"]] += 1

    langs_kept = {lang for lang, n in counts_all.items() if n >= min_data_files}
    dropped = sorted(lang for lang, n in counts_all.items() if n < min_data_files)
    paths_kept = [
        p for p in paths if (m := VERIFIED_DATA_RE.match(p)) and m["lang"] in langs_kept
    ]
    return paths_kept, counts_all, dropped


def pick_one_shard_per_lang(
    paths_kept: list[str],
    rng: random.Random,
) -> dict[str, str]:

    by_lang: dict[str, list[str]] = defaultdict(list)

    for p in paths_kept:
        m = VERIFIED_DATA_RE.match(p)
        assert m is not None
        by_lang[m["lang"]].append(p)

    return {lang: rng.choice(files) for lang, files in sorted(by_lang.items())}


def download_shards(
    shard_pick: dict[str, str],
    *,
    repo_id: str,
    revision: str,
    out_dir: Path,
) -> dict[str, str]:

    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, str] = {}

    for lang, rel_path in sorted(shard_pick.items()):
        local = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=rel_path,
            local_dir=str(out_dir),
        )
        downloaded[lang] = local
        
    return downloaded


def main() -> None:
    repo = DEFAULT_REPO
    revision = DEFAULT_REV
    out_dir = repo_root() / "data" / "sangrah_dataset"
    min_data_files = 3
    rng_seed = 42

    paths = list_verified_parquet_paths(repo, revision)
    paths_kept, _, dropped = filter_paths_by_min_shards(paths, min_data_files)

    if dropped:
        print(
            f"Skipping {len(dropped)} languages with <{min_data_files} shards: "
            f"{', '.join(dropped)}"
        )

    rng = random.Random(rng_seed)
    shard_pick = pick_one_shard_per_lang(paths_kept, rng)

    downloaded = download_shards(
        shard_pick,
        repo_id=repo,
        revision=revision,
        out_dir=out_dir,
    )
    
    for lang, rel in sorted(shard_pick.items()):
        print(f"  {lang}: {rel} -> {downloaded[lang]}")

    print(f"\n{len(downloaded)} files under {out_dir.resolve()}")


if __name__ == "__main__":
    main()
