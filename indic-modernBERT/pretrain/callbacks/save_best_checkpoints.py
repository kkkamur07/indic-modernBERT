"""Keep the best-N Composer checkpoints by eval MLM loss."""

from __future__ import annotations

import json
from pathlib import Path

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils.checkpoint import save_checkpoint
from loguru import logger

__all__ = ["SaveBestCheckpoints"]

def _metric_value(metric) -> float:
    if hasattr(metric, "compute_final"):
        result = metric.compute_final()
        if isinstance(result, dict):
            return float(next(iter(result.values())))
        return float(result)
    return float(metric.compute())


def _read_eval_metrics(state: State, eval_label: str = "eval") -> tuple[float, float]:
    metrics = state.eval_metrics.get(eval_label, {})
    loss: float | None = None
    accuracy: float | None = None

    for name, metric in metrics.items():
        value = _metric_value(metric)

        if "CrossEntropy" in name or name.endswith("Loss"):
            loss = value

        if "MaskedAccuracy" in name or name.endswith("Accuracy"):
            accuracy = value

    if loss is None:
        raise RuntimeError(f"No eval loss metric found under eval_metrics[{eval_label!r}]: {list(metrics)}")

    if accuracy is None:
        accuracy = float("nan")

    return loss, accuracy


class SaveBestCheckpoints(Callback):
    """After each eval, log MLM loss/accuracy and keep the best N checkpoints on disk."""

    def __init__(
        self,
        save_folder: str | Path,
        num_checkpoints: int = 3,
        eval_label: str = "eval",
    ) -> None:
    
        self.save_folder = Path(save_folder) / "best"
        self.num_checkpoints = num_checkpoints
        self.eval_label = eval_label
        self._ranked: list[tuple[float, float, int, Path]] = []

    def _qualifies(self, loss: float) -> bool:
        if self.num_checkpoints <= 0:
            return False
        if len(self._ranked) < self.num_checkpoints:
            return True
        return loss < max(row[0] for row in self._ranked)

    def _write_manifest(self) -> None:
        manifest = [
            {
                "path": p.name,
                "batch": batch,
                "eval_loss": loss_val,
                "masked_accuracy": acc_val,
            }
            for loss_val, acc_val, batch, p in sorted(self._ranked, key=lambda row: row[0])
        ]
        self.save_folder.mkdir(parents=True, exist_ok=True)
        (self.save_folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def eval_after_all(self, state: State, logger_cb: Logger) -> None:
        loss, accuracy = _read_eval_metrics(state, self.eval_label)
        logger_cb.log_metrics(
            {
                "eval/loss": loss,
                "eval/masked_accuracy": accuracy,
            }
        )

        batch = state.timestamp.batch.value
        if not self._qualifies(loss):
            logger.debug(
                "eval ba={} loss={:.4f} acc={:.4f} — not in top {}",
                batch,
                loss,
                accuracy,
                self.num_checkpoints,
            )
            return

        saved_path = save_checkpoint(
            state,
            filename=f"best-ba{batch}-rank{{rank}}",
            weights_only=False,
        )
        if saved_path is None:
            return

        path = Path(saved_path)
        dest = self.save_folder / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != dest.resolve():
            path.replace(dest)
        path = dest

        self._ranked.append((loss, accuracy, batch, path))
        self._ranked.sort(key=lambda row: row[0])
        while len(self._ranked) > self.num_checkpoints:
            _, _, _, worst = self._ranked.pop()
            if worst.is_file():
                worst.unlink()

        self._write_manifest()
        logger.info(
            "saved best checkpoint ba={} loss={:.4f} acc={:.4f} → {} (keeping {})",
            batch,
            loss,
            accuracy,
            path.name,
            len(self._ranked),
        )
