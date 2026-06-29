#!/usr/bin/env python3
"""Upload hindi-modernBERT (ba1157) artifacts to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MLM_DIR = _REPO_ROOT / "artifacts/model/modernbert/hi/hf_export/phase2_latest_ba1157"
DEFAULT_RETRIEVER_DIR = (
    _REPO_ROOT
    / "artifacts/retrieval_finetune/hi/full_local_jsonl_train_eval_runs"
    / "phase2_latest_ba1157/phase2_latest_ba1157-DPR-0.00010972521281842244/final"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_id",
        nargs="?",
        default=os.environ.get("HF_REPO_ID"),
        help="Target Hub repo, e.g. org/hindi-modernbert (or set HF_REPO_ID)",
    )
    parser.add_argument(
        "--variant",
        choices=("mlm", "retriever"),
        default="mlm",
        help="Upload base MLM export (ba1157) or DPR retriever checkpoint",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Override local model directory",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Target git revision on the Hub repo",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update the Hub repo as private",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print upload plan without pushing",
    )
    return parser.parse_args()


def _resolve_model_dir(args: argparse.Namespace) -> Path:
    if args.model_dir is not None:
        return args.model_dir.resolve()
    if args.variant == "retriever":
        return DEFAULT_RETRIEVER_DIR.resolve()
    return DEFAULT_MLM_DIR.resolve()


def _required_files(model_dir: Path, variant: str) -> list[Path]:
    common = ["README.md"]
    if variant == "mlm":
        names = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    else:
        names = ["config.json", "model.safetensors", "modules.json", "tokenizer.json", "tokenizer_config.json"]
    return [model_dir / name for name in [*names, *common]]


def main() -> None:
    args = _parse_args()
    if not args.repo_id:
        print("error: repo_id is required (positional arg or HF_REPO_ID env var)", file=sys.stderr)
        sys.exit(1)

    model_dir = _resolve_model_dir(args)
    if not model_dir.is_dir():
        print(f"error: model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    missing = [path for path in _required_files(model_dir, args.variant) if not path.is_file()]
    if missing:
        print("error: missing required files:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        sys.exit(1)

    print(f"variant:   {args.variant}")
    print(f"repo_id:   {args.repo_id}")
    print(f"model_dir: {model_dir}")
    print(f"revision:  {args.revision}")
    print(f"private:   {args.private}")

    if args.dry_run:
        print("dry-run: no upload performed")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("error: huggingface_hub is not installed", file=sys.stderr)
        sys.exit(1)

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        exist_ok=True,
        private=args.private,
    )
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        commit_message=f"Upload hindi-modernBERT ba1157 ({args.variant})",
    )
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
