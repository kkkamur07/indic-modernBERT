"""Download one random Hindi parquet shard from ai4bharat/sangraha (verified/hin/)."""

from __future__ import annotations

import random
import re
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
    
    # First get all the paths and then filter out the ones that match the regex
    return [entry.path for entry in tree if VERIFIED_DATA_RE.match(entry.path)]


def download_random_shard(
    *,
    repo_id: str = DEFAULT_REPO,
    revision: str = DEFAULT_REV,
    out_dir: Path,
    rng: random.Random | None = None,

) -> tuple[str, str]:

    paths = list_hindi_parquet_paths(repo_id, revision)

    if not paths:
        raise FileNotFoundError(f"No Hindi parquet shards found under verified/{HINDI_LANG3}/")

    rng = rng or random.Random()
    rel_path = rng.choice(sorted(paths))

    # Download the random shard from the hub
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        filename=rel_path,
        local_dir=str(out_dir),
    )

    return rel_path, local_path


def main() -> None:
    out_dir = repo_root() / "data" / "sangrah_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)

    rel_path, local_path = download_random_shard(out_dir=out_dir, rng=rng)
    
    print(f"  {HINDI_LANG3}: {rel_path} -> {local_path}")
    print(f"\n1 file under {out_dir.resolve()}")


if __name__ == "__main__":
    main()
