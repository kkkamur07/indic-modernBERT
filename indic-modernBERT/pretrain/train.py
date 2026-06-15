"""Composer MLM pretraining — thin entry over upstream wiring."""

from __future__ import annotations

import gc
import json
import sys

import torch
from loguru import logger
from torch.utils.data import DataLoader

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


# Wrapper attributes that hold a reference to the next loader in the chain.
# DataSpec.dataloader -> BufferedIterable.iterable (packer) -> packer.src_iterable
# (the torch DataLoader); Evaluator.dataloader -> DataSpec -> ... . The live
# DataLoader *iterator* (which owns the worker subprocesses) is held by the buffer
# (_active_iterator.iterator) and, crucially, by the packer (src_iterator) — with
# persistent_workers=False, torch does NOT store it on DataLoader._iterator, so we
# must reach it through src_iterator or the workers leak.
_LOADER_GRAPH_ATTRS = (
    "dataloader",
    "iterable",
    "src_iterable",
    "src_iterator",
    "_active_iterator",
    "iterator",
    "_iterator",
)


def _walk_loader_graph(root) -> list:
    """Collect every object reachable from a (possibly wrapped) loader so we can
    stop its background threads and reap its DataLoader worker subprocesses."""
    seen: set[int] = set()
    order: list = []
    stack = [root]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        order.append(cur)
        for attr in _LOADER_GRAPH_ATTRS:
            stack.append(getattr(cur, attr, None))
    return order


def _shutdown_worker_iterator(obj) -> None:
    """Shut down a torch ``_MultiProcessingDataLoaderIter``'s worker subprocesses.

    Works for any object exposing ``_shutdown_workers`` — both the iterator stored
    on a persistent DataLoader and the iterator held by the packer/buffer when
    persistent_workers=False (where DataLoader._iterator stays None).
    """
    shutdown = getattr(obj, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception as exc:
            logger.warning("DataLoader worker shutdown failed during teardown: {}", exc)


def _release_loader(loader) -> None:
    """Stop background fill threads and reap DataLoader worker subprocesses for a
    train or eval loader (unwrapping DataSpec/Evaluator/BufferedIterable)."""
    if loader is None:
        return
    graph = _walk_loader_graph(loader)
    # Stop packer/buffer fill threads first (e.g. BufferedIterable.close()) so they
    # stop pulling from the workers we are about to shut down.
    for obj in graph:
        if isinstance(obj, DataLoader):
            continue
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # cleanup must not mask the training result
                logger.warning("loader.close() failed during teardown: {}", exc)
    # Then reap the worker subprocesses. Shut down every reachable DataLoader
    # iterator (persistent path) AND any standalone _MultiProcessingDataLoaderIter
    # held by the packer (non-persistent path).
    for obj in graph:
        if isinstance(obj, DataLoader):
            try:
                obj.persistent_workers = False
            except Exception:
                pass
            _shutdown_worker_iterator(getattr(obj, "_iterator", None))
            try:
                obj._iterator = None
            except Exception:
                pass
        else:
            _shutdown_worker_iterator(obj)


def _release_training_resources(trainer, train_loader, eval_loader=None) -> None:
    """Tear down the Trainer and dataloaders so a multi-run process (e.g. Optuna
    sweep) does not leak DataLoader workers / packer threads across trials.

    Both the train loader (DataSpec -> BufferedIterable -> packer -> DataLoader)
    and the eval loader (Evaluator -> DataSpec -> DataLoader) must be released:
    the eval DataLoader is otherwise never closed, and DataSpec has no `.close()`,
    so persistent workers from every trial would accumulate until OOM.
    """
    _release_loader(train_loader)
    _release_loader(eval_loader)

    trainer_close = getattr(trainer, "close", None)
    if callable(trainer_close):
        try:
            trainer_close()
        except Exception as exc:
            logger.warning("trainer.close() failed during teardown: {}", exc)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    arch = pretrain_cfg.load_arch()
    if arch.compile_model:
        from pretrain.step_log import step_log

        n_layers = arch.num_hidden_layers
        step_log(
            "model",
            f"compile_model=true | {n_layers} compiled layers + compiled embeddings/head | "
            f"torch.compile runs on first forward",
            always=True,
        )

    train_loader = build_train_dataloader(pretrain_cfg, model.tokenizer, device)
    from pretrain.step_log import step_log

    if pretrain_cfg.sequence_packing:
        step_log(
            "data",
            "train dataloader ready | ParquetMLMDataset(text) -> TokenizeCollator -> "
            "GreedyBestFitSequencePacker -> BufferedIterable",
            always=True,
        )
    else:
        step_log(
            "data",
            "train dataloader ready | ParquetMLMDataset -> MLMCollator (tokenize+pad+mlm)",
            always=True,
        )
    eval_evaluator = build_eval_evaluator(pretrain_cfg, model.tokenizer, device)
    optimizer = build_optimizer(pretrain_cfg.optimizer, model)
    scheduler = build_scheduler(pretrain_cfg.scheduler)
    callbacks = []
    for name, kwargs in pretrain_cfg.callbacks.items():
        if kwargs.get("enabled") is False:
            continue
        if name == "packing_efficiency" and not pretrain_cfg.sequence_packing:
            continue
        cb_kwargs = dict(kwargs)
        cb_kwargs.pop("enabled", None)
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
    if pretrain_cfg.run_name is not None:
        trainer_kwargs["run_name"] = pretrain_cfg.run_name
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
        "console_log_interval={} | train_step_logger={} — "
        "first batch may take minutes (torch.compile + grad accum)",
        progress_bar,
        pretrain_cfg.log_to_console,
        pretrain_cfg.console_log_interval,
        "train_step_logger" in pretrain_cfg.callbacks,
    )
    try:
        trainer.fit(**fit_kwargs)
        eval_loss = _best_eval_loss(pretrain_cfg, trainer)
        logger.info("Pretrain complete | best_eval_loss={:.4f}", eval_loss)
        return eval_loss
    finally:
        _release_training_resources(trainer, train_loader, eval_evaluator)
