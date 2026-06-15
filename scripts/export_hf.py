"""Export a Composer FlexBERT checkpoint to Hugging Face format."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "indic-modernBERT"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import torch
import transformers
from loguru import logger
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from config import load_modernbert_arch_config
from model.factory import build_modernbert_config
from model.modernbert.model import FlexBertForMaskedLM
from utils.paths import resolve_hf_tokenizer_dir


def _extract_state_dict(checkpoint_path: Path) -> dict:
    """Load model weights from a Composer or raw state-dict checkpoint."""
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"Unrecognized checkpoint format: {checkpoint_path}")

    if "state" in raw and isinstance(raw["state"], dict):
        state = raw["state"]
        if "model" in state:
            state_dict = state["model"]
        else:
            state_dict = state
    else:
        state_dict = raw

    consume_prefix_in_state_dict_if_present(state_dict, prefix="model.")
    return state_dict


def export_composer_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    arch_config_path: Path | None = None,
    pretrained_model_name: str = "bert-base-uncased",
    tokenizer_path: Path | None = None,
) -> Path:
    """Load a Composer ``.pt`` checkpoint and write HF ``save_pretrained`` artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if arch_config_path is None:
        raise ValueError("`arch_config_path` is required.")

    arch = load_modernbert_arch_config(arch_config_path)
    config = build_modernbert_config(
        pretrained_model_name=pretrained_model_name,
        model_config=arch,
    )

    model = FlexBertForMaskedLM(config)
    state_dict = _extract_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading checkpoint: {}", ", ".join(missing[:8]))
        if len(missing) > 8:
            logger.warning("... and {} more missing keys", len(missing) - 8)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: {}", ", ".join(unexpected[:8]))
        if len(unexpected) > 8:
            logger.warning("... and {} more unexpected keys", len(unexpected) - 8)

    model.save_pretrained(output_dir)
    logger.info("Saved model weights to {}", output_dir)

    if tokenizer_path is not None:
        tokenizer_dir = resolve_hf_tokenizer_dir(tokenizer_path)
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
        tokenizer.save_pretrained(output_dir)
        logger.info("Saved tokenizer from {}", tokenizer_dir)

    return output_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Composer checkpoint (.pt)")
    parser.add_argument("output_dir", type=Path, help="HF export directory")
    parser.add_argument("--config", type=Path, default=Path("configs/model/modernbert_base.yaml"))
    parser.add_argument(
        "--pretrained-model-name",
        default="bert-base-uncased",
        help="HF config base for FlexBertConfig.from_pretrained",
    )
    parser.add_argument("--tokenizer", type=Path, default=None, help="tokenizer dir or tokenizer.json")
    args = parser.parse_args()

    export_composer_checkpoint(
        args.checkpoint,
        args.output_dir,
        arch_config_path=args.config,
        pretrained_model_name=args.pretrained_model_name,
        tokenizer_path=args.tokenizer,
    )


if __name__ == "__main__":
    main()
