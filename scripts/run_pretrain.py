"""Hydra entrypoint for full Composer MLM pretraining."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "indic-modernBERT"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from config import load_pretrain_config
from pretrain.parquet_mlm import describe_data_root
from pretrain.train import run_mlm_pretrain
from utils.log_helpers import setup_run_log, slug


def _suffix_multirun_paths(cfg: DictConfig) -> DictConfig:
    """Give each Hydra multirun job unique artifact dirs (hydra.job.num is not in OmegaConf tree)."""
    try:
        from hydra.core.hydra_config import HydraConfig
        from hydra.types import RunMode
    except ImportError:
        return cfg

    try:
        hc = HydraConfig.get()
    except ValueError:
        return cfg

    if hc.mode != RunMode.MULTIRUN:
        return cfg

    lr = OmegaConf.select(cfg, "pretrain.optimizer.lr")
    suffix = f"trial{hc.job.num}"
    if lr is not None:
        suffix = f"{suffix}_lr{lr}"

    for key in ("output_dir", "save_folder"):
        base = OmegaConf.select(cfg, f"pretrain.{key}")
        if base is not None:
            cfg.pretrain[key] = f"{base}/{suffix}"

    tb_dir = OmegaConf.select(cfg, "pretrain.loggers.tensorboard.log_dir")
    if tb_dir is not None:
        cfg.pretrain.loggers.tensorboard.log_dir = f"{tb_dir}/{suffix}"

    return cfg


@hydra.main(version_base=None, config_path="../configs/pretrain", config_name="hindi_mlm")
def main(cfg: DictConfig) -> float:
    cfg = _suffix_multirun_paths(cfg)
    pretrain_cfg = load_pretrain_config(cfg)
    setup_run_log(f"pretrain__data-{slug(pretrain_cfg.data_root.name)}.log")

    logger.info("Architecture config: {}", pretrain_cfg.arch_config_path)
    logger.info("Training data: {}", describe_data_root(pretrain_cfg.data_root))
    logger.info(
        "Pretrain | global_batch={} | microbatch={} | grad_accum={} | max_duration={} | "
        "checkpoints={} | save_interval={} | save_keep={} | autoresume={} | "
        "eval_interval={} | eval_batch={} | save_best={} | loggers={} | "
        "load_path={} | reset_time={} | restart_override={}",
        pretrain_cfg.global_train_batch_size,
        pretrain_cfg.device_train_microbatch_size,
        pretrain_cfg.grad_accum_steps,
        pretrain_cfg.max_duration,
        pretrain_cfg.save_folder,
        pretrain_cfg.save_interval,
        pretrain_cfg.save_num_checkpoints_to_keep,
        pretrain_cfg.autoresume,
        pretrain_cfg.eval_interval,
        pretrain_cfg.global_eval_batch_size,
        pretrain_cfg.callbacks.get("save_best_checkpoints"),
        list(pretrain_cfg.loggers),
        pretrain_cfg.load_path,
        pretrain_cfg.reset_time,
        pretrain_cfg.restart_override,
    )

    pretrain_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (pretrain_cfg.output_dir / "resolved_config.json").write_text(
        json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2) + "\n"
    )

    eval_loss = run_mlm_pretrain(pretrain_cfg)
    summary = {
        "eval_loss": eval_loss,
        "lr": pretrain_cfg.optimizer.lr,
        "max_duration": pretrain_cfg.max_duration,
        "global_train_batch_size": pretrain_cfg.global_train_batch_size,
    }
    (pretrain_cfg.output_dir / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return eval_loss


if __name__ == "__main__":
    main()
