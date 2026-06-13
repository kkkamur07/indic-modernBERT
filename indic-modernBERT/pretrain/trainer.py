"""Composer Trainer entry point for Hindi ModernBERT MLM pretraining."""

from __future__ import annotations

import json

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from config import load_pretrain_config
from pretrain.data import describe_data_root
from utils.log_helpers import setup_training_run_log


@hydra.main(version_base=None, config_path="../../../configs/pretrain", config_name="hindi_mlm")
def main(cfg: DictConfig) -> None:
    pretrain_cfg = load_pretrain_config(cfg)
    setup_training_run_log("pretrain")

    arch = pretrain_cfg.load_arch()
    logger.info("Architecture config: {}", pretrain_cfg.arch_config_path)
    logger.info("Training data: {}", describe_data_root(pretrain_cfg.data_root))
    logger.info(
        "Tokenizer: {} | checkpoints: {}",
        pretrain_cfg.tokenizer_path,
        pretrain_cfg.output_dir,
    )

    try:
        from composer import Trainer
    except ImportError as exc:
        raise ImportError(
            "Composer is required for pretraining. Install with: uv sync --extra pretrain"
        ) from exc

    from model.factory import create_modernbert_mlm

    composer_model = create_modernbert_mlm(
        pretrained_model_name=pretrain_cfg.pretrained_model_name,
        model_config=arch,
        tokenizer_path=str(pretrain_cfg.tokenizer_path),
        gradient_checkpointing=pretrain_cfg.gradient_checkpointing,
        disable_train_metrics=pretrain_cfg.disable_train_metrics,
    )

    pretrain_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (pretrain_cfg.output_dir / "resolved_config.json").write_text(
        json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2)
    )

    raise NotImplementedError(
        "Composer dataloaders and Trainer wiring are not ported yet. "
        "Model factory is ready; next step is porting text_data.py and main.py."
    )


if __name__ == "__main__":
    main()
