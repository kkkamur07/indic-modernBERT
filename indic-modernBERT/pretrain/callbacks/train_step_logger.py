"""Per-batch and per-microbatch training logs (Composer callback)."""

from __future__ import annotations

import time
from typing import Any

import torch
from composer.core import Callback, State
from composer.loggers import Logger

from pretrain.sequence_packer import get_num_samples_in_packed_batch, split_packed_batch
from pretrain.step_log import step_log

__all__ = ["TrainStepLogger"]


def _tensor_shapes(batch: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(batch.keys()):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            parts.append(f"{key}={tuple(value.shape)}")
    return ", ".join(parts) or "(no tensors)"


def _loss_scalar(loss: Any) -> float | None:
    if loss is None:
        return None
    if isinstance(loss, dict):
        if "total" in loss:
            loss = loss["total"]
        else:
            total = None
            for value in loss.values():
                if isinstance(value, torch.Tensor):
                    total = value if total is None else total + value
                elif isinstance(value, (int, float)):
                    total = torch.tensor(float(value)) if total is None else total + float(value)
            loss = total
    if isinstance(loss, torch.Tensor):
        detached = loss.detach()
        if detached.numel() == 0:
            return None
        if detached.numel() > 1:
            detached = detached.float().mean()
        return float(detached.cpu())
    try:
        return float(loss)
    except (TypeError, ValueError):
        return None


class TrainStepLogger(Callback):
    """Log dataloader fetch, each micro forward/backward, and optimizer steps.

    Uses loguru (same as ``pretrain/train.py``) and flushes stderr so ``make
    train-smoke-50ba`` / ``nohup.log`` show progress while the tqdm bar sits at 0/50.
    """

    def __init__(
        self,
        *,
        log_microbatches: bool = True,
        log_every_micro: bool = True,
        micro_log_interval: int = 1,
        log_eval_batches: bool = True,
    ) -> None:
        self.log_microbatches = log_microbatches
        self.log_every_micro = log_every_micro
        self.micro_log_interval = max(1, micro_log_interval)
        self.log_eval_batches = log_eval_batches
        self._micro_idx = 0
        self._n_micros = 0
        self._batch_t0 = 0.0
        self._micro_t0 = 0.0
        self._fit_t0 = 0.0
        self._eval_batch_idx = 0
        self._compile_logged = False
        self._compile_done = False

    def _emit(self, message: str) -> None:
        step_log("train", message)

    def _should_log_micro(self, micro_idx: int) -> bool:
        if not self.log_microbatches:
            return False
        if self.log_every_micro:
            return True
        if micro_idx == 0:
            return True
        if self._n_micros > 0 and micro_idx + 1 == self._n_micros:
            return True
        return (micro_idx + 1) % self.micro_log_interval == 0

    def fit_start(self, state: State, logger_cb: Logger) -> None:
        del logger_cb
        self._fit_t0 = time.perf_counter()
        micro = state.device_train_microbatch_size
        self._emit(
            f"fit_start | microbatch_size={micro} | "
            f"pipeline: parquet -> TokenizeCollator -> packer -> MLM mask -> model"
        )

    def before_dataloader(self, state: State, logger_cb: Logger) -> None:
        del logger_cb
        batch_no = int(state.timestamp.batch.value) + 1
        self._emit(
            f"batch {batch_no} | waiting for packed batch "
            f"(BufferedIterable; may block on parquet read / tokenize / pack)"
        )

    def after_dataloader(self, state: State, logger_cb: Logger) -> None:
        del logger_cb
        batch_no = int(state.timestamp.batch.value) + 1
        batch = state.batch
        micro = state.device_train_microbatch_size
        assert micro is not None

        microbatches = split_packed_batch(batch, int(micro))
        self._n_micros = len(microbatches)
        self._micro_idx = 0
        self._batch_t0 = time.perf_counter()

        n_seqs = get_num_samples_in_packed_batch(batch)
        n_tokens = 0
        if isinstance(batch.get("attention_mask"), torch.Tensor):
            n_tokens = int(batch["attention_mask"].sum())

        self._emit(
            f"batch {batch_no} | batch loaded | "
            f"microbatches={self._n_micros} packed_seqs={n_seqs} tokens={n_tokens} | "
            f"{_tensor_shapes(batch)}"
        )

    def batch_start(self, state: State, logger_cb: Logger) -> None:
        del logger_cb, state

    def before_forward(self, state: State, logger_cb: Logger) -> None:
        del logger_cb
        if not self._should_log_micro(self._micro_idx):
            self._micro_t0 = time.perf_counter()
            return
        batch_no = int(state.timestamp.batch.value) + 1
        self._micro_t0 = time.perf_counter()
        if batch_no == 1 and self._micro_idx == 0 and not self._compile_logged:
            self._compile_logged = True
            step_log(
                "model",
                "torch.compile start | first forward compiles embeddings + layers + head "
                "(may take several minutes; CPU-heavy)",
                always=True,
            )
        self._emit(
            f"batch {batch_no} | micro {self._micro_idx + 1}/{self._n_micros} | forward..."
        )

    def after_forward(self, state: State, logger_cb: Logger) -> None:
        del logger_cb
        if not self._should_log_micro(self._micro_idx):
            return
        batch_no = int(state.timestamp.batch.value) + 1
        loss = _loss_scalar(state.loss)
        elapsed = time.perf_counter() - self._micro_t0
        loss_s = f"{loss:.4f}" if loss is not None else "n/a"
        logits = ""
        outputs = state.outputs
        if isinstance(outputs, dict) and isinstance(outputs.get("logits"), torch.Tensor):
            logits = f" logits={tuple(outputs['logits'].shape)}"
        if batch_no == 1 and self._micro_idx == 0 and not self._compile_done:
            self._compile_done = True
            step_log(
                "model",
                f"torch.compile done | first forward finished in {elapsed:.1f}s",
                always=True,
            )
        self._emit(
            f"batch {batch_no} | micro {self._micro_idx + 1}/{self._n_micros} | "
            f"forward loss={loss_s}{logits} | {elapsed:.1f}s"
        )

    def after_backward(self, state: State, logger_cb: Logger) -> None:
        del logger_cb
        if not self._should_log_micro(self._micro_idx):
            self._micro_idx += 1
            return
        batch_no = int(state.timestamp.batch.value) + 1
        elapsed = time.perf_counter() - self._micro_t0
        self._emit(
            f"batch {batch_no} | micro {self._micro_idx + 1}/{self._n_micros} | "
            f"backward done | {elapsed:.1f}s total"
        )
        self._micro_idx += 1

    def batch_end(self, state: State, logger_cb: Logger) -> None:
        del logger_cb
        batch_no = int(state.timestamp.batch.value)
        elapsed = time.perf_counter() - self._batch_t0
        total_loss = None
        if state.total_loss_dict:
            total_loss = state.total_loss_dict.get("loss/train/total")
        loss_s = f"{total_loss:.4f}" if total_loss is not None else "n/a"
        self._emit(
            f"batch {batch_no} | optimizer.step() | mean_loss={loss_s} | "
            f"batch_time={elapsed:.1f}s | elapsed={time.perf_counter() - self._fit_t0:.0f}s"
        )

    def eval_start(self, state: State, logger_cb: Logger) -> None:
        del logger_cb
        if not self.log_eval_batches:
            return
        self._eval_batch_idx = 0
        self._emit(f"eval start | after train batch {state.timestamp.batch.value}")

    def eval_batch_start(self, state: State, logger_cb: Logger) -> None:
        del logger_cb, state
        if not self.log_eval_batches:
            return
        self._micro_t0 = time.perf_counter()
        self._eval_batch_idx += 1
        step_log("data", f"eval batch {self._eval_batch_idx} | forward (tokenize+mlm already in collator)...")

    def eval_after_forward(self, state: State, logger_cb: Logger) -> None:
        del logger_cb
        if not self.log_eval_batches:
            return
        loss = _loss_scalar(state.loss)
        loss_s = f"{loss:.4f}" if loss is not None else "n/a"
        elapsed = time.perf_counter() - self._micro_t0
        shapes = _tensor_shapes(state.batch) if isinstance(state.batch, dict) else ""
        self._emit(
            f"eval batch {self._eval_batch_idx} | loss={loss_s} | {elapsed:.1f}s | {shapes}"
        )

    def eval_end(self, state: State, logger_cb: Logger) -> None:
        del logger_cb
        if not self.log_eval_batches:
            return
        metrics = state.eval_metric_values or {}
        parts = [f"{k}={v}" for k, v in sorted(metrics.items()) if isinstance(v, (int, float))]
        summary = ", ".join(parts[:6]) or "metrics pending"
        self._emit(f"eval end | {summary}")
