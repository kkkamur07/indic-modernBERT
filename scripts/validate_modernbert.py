"""Smoke-test ModernBERT forward pass and MLM loss."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger

from config import load_modernbert_arch_config
from model import ModernBertConfig, ModernBertForMaskedLM, build_modernbert_config


def load_model_config(path: Path) -> ModernBertConfig:
    arch = load_modernbert_arch_config(path)
    return build_modernbert_config(model_config=arch)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_smoke(config: ModernBertConfig, batch_size: int, seq_len: int, device: str) -> None:
    model = ModernBertForMaskedLM(config).to(device)
    model.eval()

    vocab = config.vocab_size
    input_ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    with torch.no_grad():
        if config.padding == "unpadded":
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            probe = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        else:
            probe = model(input_ids=input_ids, attention_mask=attention_mask)

    assert probe.logits.shape == (batch_size, seq_len, vocab)

    labels = input_ids.clone()
    labels[:, :2] = -100

    with torch.no_grad():
        if config.padding == "unpadded":
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
            )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

    assert outputs.loss is not None
    if config.masked_prediction:
        num_masked = int((labels != -100).sum())
        assert outputs.logits.shape == (num_masked, vocab)
    else:
        assert outputs.logits.shape == (batch_size, seq_len, vocab)

    logger.info(
        "ModernBERT smoke OK | padding={} layers={} hidden={} params={:,} loss={:.4f} logits={}",
        config.padding,
        config.num_hidden_layers,
        config.hidden_size,
        count_parameters(model),
        float(outputs.loss),
        tuple(outputs.logits.shape),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/modernbert_smoke.yaml"),
        help="YAML with a top-level model_config dict",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    config = load_model_config(config_path)
    logger.info("Loaded config from {}", config_path)
    run_smoke(config, args.batch_size, args.seq_len, args.device)


if __name__ == "__main__":
    main()
