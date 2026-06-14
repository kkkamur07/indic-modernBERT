"""Composer MLM pretraining — thin entry over upstream wiring."""

from __future__ import annotations

import json
import sys

import torch
from loguru import logger

from config import PretrainConfig
from model.factory import create_modernbert_mlm
from pretrain.gpu_batch import log_device_summary, resolve_device
from pretrain.wiring import (
    apply_restart_override,
    build_callback,
    build_eval_evaluator,
    build_logger,
    build_optimizer,
    build_scheduler,
    build_train_dataloader,
)


def _validate_production_kernels(pretrain_cfg: PretrainConfig, model) -> None:
    """Require FA2 + torch.compile when the arch yaml requests production kernels."""
    from model.modernbert.attention import IMPL_USE_FLASH2
    from model.modernbert.embeddings import FlexBertCompiledSansPositionEmbeddings
    from model.modernbert.layers import FlexBertCompileUnpadPreNormLayer

    arch = pretrain_cfg.load_arch()
    if not (arch.use_fa2 and arch.compile_model):
        return

    if not IMPL_USE_FLASH2:
        raise SystemExit(
            "Production arch requires flash-attn (FA2). Install: uv sync --extra pretrain"
        )

    hf_model = model.model
    config = hf_model.config
    if not config.use_fa2:
        raise ValueError(f"Model config use_fa2={config.use_fa2}; expected True")
    if not config.compile_model:
        raise ValueError(f"Model config compile_model={config.compile_model}; expected True")

    emb = hf_model.bert.embeddings
    layer0 = hf_model.bert.encoder.layers[0]
    if not isinstance(emb, FlexBertCompiledSansPositionEmbeddings):
        raise ValueError(
            f"compile_model=true requires FlexBertCompiledSansPositionEmbeddings, got {type(emb).__name__}"
        )
    if not isinstance(layer0, FlexBertCompileUnpadPreNormLayer):
        raise ValueError(
            f"compile_model=true requires FlexBertCompileUnpadPreNormLayer, got {type(layer0).__name__}"
        )
    if not hasattr(hf_model, "compiled_head"):
        raise ValueError("FlexBertForMaskedLM missing compiled_head")

    n_fa3 = sum(1 for layer in hf_model.bert.encoder.layers if getattr(layer.attn, "use_fa3", False))
    n_local_fa2 = sum(
        1
        for layer in hf_model.bert.encoder.layers
        if getattr(layer.attn, "use_fa2", False) and not getattr(layer.attn, "use_fa3", False)
    )
    logger.info(
        "Production kernels OK | FA2 installed | compile_model=true | "
        "attention layers FA3={} FA2={} | loss={}",
        n_fa3,
        n_local_fa2,
        arch.loss_function,
    )


def _best_eval_loss(pretrain_cfg: PretrainConfig, trainer) -> float:
    manifest_path = pretrain_cfg.save_folder / "best" / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest:
            return float(manifest[0]["eval_loss"])

    metrics = trainer.state.eval_metrics.get("eval", {})
    for name, metric in metrics.items():
        if "CrossEntropy" in name or name.endswith("Loss"):
            if hasattr(metric, "compute_final"):
                result = metric.compute_final()
                if isinstance(result, dict):
                    return float(next(iter(result.values())))
                return float(result)
            return float(metric.compute())

    raise RuntimeError(
        "No eval loss found after training — set eval_interval and save_best_checkpoints, "
        "or ensure eval_data_root is configured."
    )


def run_mlm_pretrain(pretrain_cfg: PretrainConfig) -> float:
    try:
        from composer import Trainer
    except ImportError as exc:
        raise ImportError(
            "Composer is required for full pretraining. Install with: uv sync --extra pretrain"
        ) from exc

    device = resolve_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    grad_accum_steps = pretrain_cfg.grad_accum_steps
    save_best = pretrain_cfg.callbacks.get("save_best_checkpoints")
    logger.info(
        "MLM pretrain | global_batch={} | microbatch={} | grad_accum={} | "
        "train_packing={} | eval_packing={} | max_duration={} | seq={} | arch={} | "
        "eval_interval={} | save_interval={} | save_best={} | load_path={} | {}",
        pretrain_cfg.global_train_batch_size,
        pretrain_cfg.device_train_microbatch_size,
        grad_accum_steps,
        pretrain_cfg.sequence_packing,
        pretrain_cfg.eval_sequence_packing,
        pretrain_cfg.max_duration,
        pretrain_cfg.max_seq_len,
        pretrain_cfg.arch_config_path,
        pretrain_cfg.eval_interval,
        pretrain_cfg.save_interval,
        save_best,
        pretrain_cfg.load_path,
        log_device_summary(device),
    )

    model = create_modernbert_mlm(
        pretrained_model_name=pretrain_cfg.pretrained_model_name,
        model_config=pretrain_cfg.load_arch(),
        tokenizer_path=str(pretrain_cfg.tokenizer_path),
        gradient_checkpointing=pretrain_cfg.gradient_checkpointing,
        disable_train_metrics=pretrain_cfg.disable_train_metrics,
    )
    _validate_production_kernels(pretrain_cfg, model)

    train_loader = build_train_dataloader(pretrain_cfg, model.tokenizer, device)
    eval_evaluator = build_eval_evaluator(pretrain_cfg, model.tokenizer, device)
    optimizer = build_optimizer(pretrain_cfg.optimizer, model)
    scheduler = build_scheduler(pretrain_cfg.scheduler)
    callbacks = []
    for name, kwargs in pretrain_cfg.callbacks.items():
        if name == "packing_efficiency" and not pretrain_cfg.sequence_packing:
            continue
        cb_kwargs = dict(kwargs)
        if name == "save_best_checkpoints" and "save_folder" not in cb_kwargs:
            cb_kwargs["save_folder"] = pretrain_cfg.save_folder
        callbacks.append(build_callback(name, cb_kwargs))

    loggers = [build_logger(name, kwargs) for name, kwargs in pretrain_cfg.loggers.items()]

    pretrain_cfg.save_folder.mkdir(parents=True, exist_ok=True)

    # Composer tqdm needs a TTY; `script` in the Makefile provides one when logging to a file.
    # Piping to `tee` without `script` disables the bar (BrokenPipeError on cursor moves).
    progress_bar = pretrain_cfg.progress_bar and (
        sys.stdout.isatty() or __import__("os").environ.get("TRAIN_PROGRESS_BAR") == "1"
    )
    if pretrain_cfg.progress_bar and not progress_bar:
        logger.info(
            "Disabling Composer progress bar (stdout is not a TTY). "
            "Use `make train-smoke-50ba` (script wrapper) or set log_to_console=true."
        )

    trainer_kwargs: dict = {
        "model": model,
        "train_dataloader": train_loader,
        "optimizers": optimizer,
        "schedulers": scheduler,
        "callbacks": callbacks,
        "max_duration": pretrain_cfg.max_duration,
        "device_train_microbatch_size": pretrain_cfg.device_train_microbatch_size,
        "save_folder": str(pretrain_cfg.save_folder),
        "save_interval": pretrain_cfg.save_interval,
        "save_num_checkpoints_to_keep": pretrain_cfg.save_num_checkpoints_to_keep,
        "save_overwrite": pretrain_cfg.save_overwrite,
        "progress_bar": progress_bar,
        "precision": pretrain_cfg.precision,
        "seed": pretrain_cfg.seed,
        "log_to_console": pretrain_cfg.log_to_console,
        "console_log_interval": pretrain_cfg.console_log_interval,
    }
    if loggers:
        trainer_kwargs["loggers"] = loggers
    if pretrain_cfg.autoresume is not None:
        trainer_kwargs["autoresume"] = pretrain_cfg.autoresume
    if eval_evaluator is not None:
        trainer_kwargs["eval_dataloader"] = eval_evaluator
        trainer_kwargs["eval_interval"] = pretrain_cfg.eval_interval
        if pretrain_cfg.eval_subset_num_batches >= 0:
            trainer_kwargs["eval_subset_num_batches"] = pretrain_cfg.eval_subset_num_batches
    if pretrain_cfg.load_path is not None:
        trainer_kwargs["load_path"] = str(pretrain_cfg.load_path)

    trainer = Trainer(**trainer_kwargs)
    apply_restart_override(pretrain_cfg, trainer)

    fit_kwargs: dict = {"reset_time": pretrain_cfg.reset_time}
    if pretrain_cfg.restart_override:
        fit_kwargs["device_train_microbatch_size"] = pretrain_cfg.device_train_microbatch_size

    logger.info(
        "Starting trainer.fit() | progress_bar={} | log_to_console={} | "
        "console_log_interval={} — first batch may take 1–3 min (torch.compile + grad accum)",
        progress_bar,
        pretrain_cfg.log_to_console,
        pretrain_cfg.console_log_interval,
    )
    trainer.fit(**fit_kwargs)
    eval_loss = _best_eval_loss(pretrain_cfg, trainer)
    logger.info("Pretrain complete | best_eval_loss={:.4f}", eval_loss)
    return eval_loss
