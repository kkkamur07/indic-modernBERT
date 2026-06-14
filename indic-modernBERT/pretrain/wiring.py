"""Upstream ModernBERT main.py training wiring (optimizer, scheduler, callbacks, DataSpec)."""

from __future__ import annotations

from typing import Any

import torch
from composer.callbacks import LRMonitor, MemoryMonitor, OptimizerMonitor, RuntimeEstimator, SpeedMonitor
from composer.core import DataSpec
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (
    ConstantWithWarmupScheduler,
    CosineAnnealingWithWarmupScheduler,
    LinearWithWarmupScheduler,
)
from composer.utils import dist
from torch import nn
from torch.optim import AdamW
from transformers import PreTrainedTokenizerBase

from config import OptimizerConfig, PretrainConfig, SchedulerConfig
from pretrain.callbacks.dataloader_speed import DataloaderSpeedMonitor
from pretrain.callbacks.log_grad_norm import LogGradNorm
from pretrain.callbacks.packing_efficiency import PackingEfficency
from pretrain.callbacks.save_best_checkpoints import SaveBestCheckpoints
from pretrain.callbacks.train_step_logger import TrainStepLogger
from pretrain.dataloader import (
    build_eval_dataloader,
    build_parquet_train_dataloader,
    build_padded_mlm_dataloader,
)
from pretrain.scheduler import CosineInverseSqrtScheduler, OneMinusSqrtScheduler, WarmupStableDecayScheduler
from pretrain.sequence_packer import get_num_samples_in_packed_batch, split_packed_batch


def param_groups_weight_decay(model: nn.Module, weight_decay: float = 1e-5, no_weight_decay_list=()):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}]


def build_callback(name: str, kwargs: dict[str, Any]):
    if name == "lr_monitor":
        return LRMonitor()
    if name == "memory_monitor":
        return MemoryMonitor()
    if name == "speed_monitor":
        return SpeedMonitor(
            window_size=kwargs.get("window_size", 1),
            gpu_flops_available=kwargs.get("gpu_flops_available", None),
        )
    if name == "runtime_estimator":
        return RuntimeEstimator()
    if name == "optimizer_monitor":
        return OptimizerMonitor(log_optimizer_metrics=kwargs.get("log_optimizer_metrics", True))
    if name == "log_grad_norm":
        return LogGradNorm(
            log_optimizer_metrics=kwargs.get("log_optimizer_metrics", True),
            batch_log_interval=kwargs.get("batch_log_interval", 10),
        )
    if name == "dataloader_speed":
        return DataloaderSpeedMonitor()
    if name == "packing_efficiency":
        return PackingEfficency(log_interval=kwargs.get("log_interval", 10))
    if name == "save_best_checkpoints":
        return SaveBestCheckpoints(**kwargs)
    if name == "train_step_logger":
        return TrainStepLogger(
            log_microbatches=kwargs.get("log_microbatches", True),
            log_every_micro=kwargs.get("log_every_micro", True),
            micro_log_interval=kwargs.get("micro_log_interval", 1),
            log_eval_batches=kwargs.get("log_eval_batches", True),
        )
    raise ValueError(f"Not sure how to build callback: {name}")


def build_logger(name: str, kwargs: dict[str, Any]):
    if name == "tensorboard":
        from composer.loggers import TensorboardLogger

        from utils.paths import resolve_from_cwd

        logger_kwargs = dict(kwargs)
        log_dir = logger_kwargs.get("log_dir")
        if log_dir is not None:
            logger_kwargs["log_dir"] = str(resolve_from_cwd(log_dir))
        return TensorboardLogger(**logger_kwargs)
    raise ValueError(f"Not sure how to build logger: {name}")


def build_scheduler(cfg: SchedulerConfig):
    if cfg.name == "constant_with_warmup":
        return ConstantWithWarmupScheduler(t_warmup=cfg.t_warmup)
    if cfg.name == "cosine_with_warmup":
        return CosineAnnealingWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    if cfg.name == "linear_decay_with_warmup":
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    if cfg.name == "warmup_stable_decay":
        return WarmupStableDecayScheduler(
            t_warmup=cfg.t_warmup,
            alpha_f=cfg.alpha_f,
            t_decay=cfg.t_decay,
        )
    if cfg.name == "cosine_inverse_sqrt":
        return CosineInverseSqrtScheduler(
            t_warmup=cfg.t_warmup,
            t_cooldown=cfg.t_cooldown,
            t_cosine=cfg.t_cosine,
            alpha_f=cfg.alpha_f,
            alpha_s=cfg.alpha_s,
            warmup_schedule=cfg.warmup_schedule,
            cooldown_schedule=cfg.cooldown_schedule,
        )
    if cfg.name == "one_minus_sqrt":
        return OneMinusSqrtScheduler(t_decay=cfg.t_decay, t_max=cfg.t_max, alpha_f=cfg.alpha_f)
    raise ValueError(f"Not sure how to build scheduler: {cfg.name}")


def build_optimizer(cfg: OptimizerConfig, model: nn.Module):
    if cfg.filter_bias_norm_wd:
        params = param_groups_weight_decay(model, weight_decay=cfg.weight_decay)
    else:
        params = model.parameters()

    betas = [cfg.beta1, cfg.beta2]
    if cfg.name == "decoupled_adamw":
        return DecoupledAdamW(params, lr=cfg.lr, betas=betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
    if cfg.name == "adamw":
        return AdamW(params, lr=cfg.lr, betas=betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
    if cfg.name in {"stableadamw", "decoupled_stableadamw"}:
        try:
            if cfg.log_grad_norm:
                from pretrain.optimizer import StableAdamW
            else:
                from optimi import StableAdamW
        except ImportError as exc:
            raise ImportError("Install torch-optimi for StableAdamW: uv sync --extra pretrain") from exc

        decouple_lr = cfg.name == "decoupled_stableadamw"
        return StableAdamW(
            params,
            lr=cfg.lr,
            betas=tuple(betas),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            decouple_lr=decouple_lr,
        )
    raise ValueError(f"Not sure how to build optimizer: {cfg.name}")


def get_num_tokens_in_batch_unpadded(batch: dict) -> int:
    return batch["attention_mask"].sum().item()


def build_train_dataloader(
    pretrain_cfg: PretrainConfig,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> DataSpec | Any:
    device_batch_size = pretrain_cfg.global_train_batch_size // dist.get_world_size()
    if pretrain_cfg.sequence_packing:
        data_loader = build_parquet_train_dataloader(
            pretrain_cfg,
            tokenizer,
            device,
            device_batch_size=device_batch_size,
        )
    else:
        data_loader = build_padded_mlm_dataloader(
            pretrain_cfg,
            tokenizer,
            device,
            device_batch_size=device_batch_size,
        )

    split_batch_fn = None
    num_samples_in_batch_fn = None
    num_tokens_in_batch_fn = None
    if not pretrain_cfg.count_padding_tokens:
        num_tokens_in_batch_fn = get_num_tokens_in_batch_unpadded
    if pretrain_cfg.sequence_packing:
        split_batch_fn = split_packed_batch
        num_samples_in_batch_fn = get_num_samples_in_packed_batch

    return DataSpec(
        data_loader,
        get_num_tokens_in_batch=num_tokens_in_batch_fn,
        split_batch=split_batch_fn,
        get_num_samples_in_batch=num_samples_in_batch_fn,
    )


def build_eval_evaluator(pretrain_cfg: PretrainConfig, tokenizer: PreTrainedTokenizerBase, device: torch.device):
    from composer import Evaluator

    if pretrain_cfg.eval_data_root is None or pretrain_cfg.eval_interval is None:
        return None

    eval_loader = build_eval_dataloader(pretrain_cfg, tokenizer, device)
    device_eval_microbatch_size = pretrain_cfg.device_eval_microbatch_size
    if device_eval_microbatch_size is None and pretrain_cfg.global_eval_batch_size is not None:
        device_eval_microbatch_size = pretrain_cfg.global_eval_batch_size // dist.get_world_size()

    return Evaluator(
        label="eval",
        dataloader=eval_loader,
        device_eval_microbatch_size=device_eval_microbatch_size,
    )


def apply_restart_override(pretrain_cfg: PretrainConfig, trainer) -> None:
    """Mirror upstream main.py restart_override before trainer.fit()."""
    if not pretrain_cfg.restart_override:
        return

    from loguru import logger

    logger.info(
        "restart_override: applying optimizer LR/WD and scheduler base_lrs from config "
        "(scheduler={})",
        pretrain_cfg.scheduler.name,
    )
    optimizer = trainer.state.optimizers[0]
    scheduler_cfg = pretrain_cfg.scheduler
    opt_cfg = pretrain_cfg.optimizer

    if scheduler_cfg.name not in {"constant_with_warmup", "warmup_stable_decay"}:
        lr_ratio = opt_cfg.lr / optimizer.param_groups[0]["lr"]
        for param_group in optimizer.param_groups:
            param_group["lr"] = opt_cfg.lr
            param_group["weight_decay"] = (
                opt_cfg.weight_decay if param_group.get("weight_decay", 0.0) > 0 else 0.0
            )
        for scheduler in trainer.state.schedulers:
            for i in range(len(scheduler.base_lrs)):
                scheduler.base_lrs[i] *= lr_ratio
            for i in range(len(scheduler._last_lr)):
                scheduler._last_lr[i] *= lr_ratio
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = opt_cfg.lr
            param_group["weight_decay"] = (
                opt_cfg.weight_decay if param_group.get("weight_decay", 0.0) > 0 else 0.0
            )
        for scheduler in trainer.state.schedulers:
            for i in range(len(scheduler.base_lrs)):
                scheduler.base_lrs[i] = opt_cfg.lr
            for i in range(len(scheduler._last_lr)):
                scheduler._last_lr[i] = opt_cfg.lr
