"""Export a Composer checkpoint to HF-native ModernBERT format."""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "indic-modernBERT"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import torch
import transformers
from loguru import logger
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from transformers import ModernBertConfig, ModernBertForMaskedLM

from config import load_modernbert_arch_config
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


def _layer_types(num_hidden_layers: int, global_attn_every_n_layers: int) -> list[str]:
    if global_attn_every_n_layers <= 0:
        return ["full_attention"] * num_hidden_layers
    return [
        "full_attention" if layer_idx % global_attn_every_n_layers == 0 else "sliding_attention"
        for layer_idx in range(num_hidden_layers)
    ]


def _special_token_id(tokenizer: transformers.PreTrainedTokenizerBase | None, token_name: str, default: int | None) -> int | None:
    if tokenizer is None:
        return default
    token_id = getattr(tokenizer, f"{token_name}_token_id", None)
    return default if token_id is None else int(token_id)


def _build_hf_modernbert_config(
    arch_config_path: Path,
    *,
    tokenizer: transformers.PreTrainedTokenizerBase | None,
    max_position_embeddings: int,
) -> ModernBertConfig:
    arch = load_modernbert_arch_config(arch_config_path)
    if arch.attention_layer != "rope":
        raise ValueError("HF-native ModernBERT export requires `attention_layer: rope`.")
    if arch.mlp_layer != "glu":
        raise ValueError("HF-native ModernBERT export requires `mlp_layer: glu`.")
    if not arch.final_norm:
        raise ValueError("HF-native ModernBERT export requires `final_norm: true`.")

    local_rope_theta = arch.local_attn_rotary_emb_base
    if local_rope_theta <= 0:
        local_rope_theta = arch.rotary_emb_base

    rope_parameters = {
        "full_attention": {
            "rope_type": "default",
            "rope_theta": float(arch.rotary_emb_base),
        },
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": float(local_rope_theta),
        },
    }

    return ModernBertConfig(
        vocab_size=arch.vocab_size,
        hidden_size=arch.hidden_size,
        intermediate_size=arch.intermediate_size,
        num_hidden_layers=arch.num_hidden_layers,
        num_attention_heads=arch.num_attention_heads,
        hidden_activation=arch.hidden_act,
        max_position_embeddings=max_position_embeddings,
        norm_eps=arch.norm_kwargs.eps,
        norm_bias=arch.norm_kwargs.bias,
        attention_bias=arch.attn_qkv_bias,
        attention_dropout=arch.attention_probs_dropout_prob,
        layer_types=_layer_types(arch.num_hidden_layers, arch.global_attn_every_n_layers),
        rope_parameters=rope_parameters,
        local_attention=arch.sliding_window if arch.sliding_window > 0 else max_position_embeddings,
        embedding_dropout=arch.embed_dropout_prob,
        mlp_bias=arch.mlp_in_bias or arch.mlp_out_bias,
        mlp_dropout=arch.mlp_dropout_prob,
        decoder_bias=True,
        sparse_prediction=False,
        pad_token_id=_special_token_id(tokenizer, "pad", 50283),
        bos_token_id=_special_token_id(tokenizer, "bos", 50281),
        eos_token_id=_special_token_id(tokenizer, "eos", 50282),
        cls_token_id=_special_token_id(tokenizer, "cls", 50281),
        sep_token_id=_special_token_id(tokenizer, "sep", 50282),
    )


def _to_hf_modernbert_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    converted = {}
    for key, value in state_dict.items():
        if key.startswith("bert.embeddings."):
            key = key.replace("bert.embeddings.", "model.embeddings.", 1)
        elif key.startswith("bert.encoder.layers."):
            key = key.replace("bert.encoder.layers.", "model.layers.", 1)
        elif key.startswith("bert.final_norm."):
            key = key.replace("bert.final_norm.", "model.final_norm.", 1)
        elif key.startswith("bert."):
            key = key.replace("bert.", "model.", 1)
        converted[key] = value
    return converted


def _assert_hf_native_modernbert(output_dir: Path) -> None:
    config_path = output_dir / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("model_type") != "modernbert":
        raise ValueError(f"{config_path} is not HF-native ModernBERT: model_type={payload.get('model_type')!r}")


def export_composer_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    arch_config_path: Path | None = None,
    tokenizer_path: Path | None = None,
    max_position_embeddings: int = 8192,
) -> Path:
    """Load a Composer ``.pt`` checkpoint and write HF-native ModernBERT artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if arch_config_path is None:
        raise ValueError("`arch_config_path` is required.")

    tokenizer = None
    if tokenizer_path is not None:
        tokenizer_dir = resolve_hf_tokenizer_dir(tokenizer_path)
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))

    config = _build_hf_modernbert_config(
        arch_config_path,
        tokenizer=tokenizer,
        max_position_embeddings=max_position_embeddings,
    )
    model = ModernBertForMaskedLM(config)
    state_dict = _to_hf_modernbert_state_dict(_extract_state_dict(checkpoint_path))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading checkpoint: {}", ", ".join(missing[:8]))
        if len(missing) > 8:
            logger.warning("... and {} more missing keys", len(missing) - 8)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: {}", ", ".join(unexpected[:8]))
        if len(unexpected) > 8:
            logger.warning("... and {} more unexpected keys", len(unexpected) - 8)
    if missing or unexpected:
        raise ValueError(
            "Checkpoint did not match HF-native ModernBERT exactly; refusing to export partial weights. "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    model.save_pretrained(output_dir)
    _assert_hf_native_modernbert(output_dir)
    logger.info("Saved HF-native ModernBERT model weights to {}", output_dir)

    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
        logger.info("Saved tokenizer from {}", tokenizer_path)

    return output_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Composer checkpoint (.pt)")
    parser.add_argument("output_dir", type=Path, help="HF-native ModernBERT export directory")
    parser.add_argument("--config", type=Path, default=Path("configs/model/modernbert_base.yaml"))
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=8192,
        help="ModernBERT max_position_embeddings to write into config.json",
    )
    parser.add_argument("--tokenizer", type=Path, default=None, help="tokenizer dir or tokenizer.json")
    args = parser.parse_args()

    export_composer_checkpoint(
        args.checkpoint,
        args.output_dir,
        arch_config_path=args.config,
        tokenizer_path=args.tokenizer,
        max_position_embeddings=args.max_position_embeddings,
    )


if __name__ == "__main__":
    main()
