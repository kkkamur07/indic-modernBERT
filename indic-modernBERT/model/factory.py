"""Build ModernBERT Composer models."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import transformers
from omegaconf import DictConfig, OmegaConf

from config import ModernBertArchConfig, load_modernbert_arch_config
from model.modernbert.configuration import FlexBertConfig
from model.modernbert.model import FlexBertForMaskedLM
from utils.paths import resolve_hf_tokenizer_dir

try:
    from composer.devices import DeviceCPU
    from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
    from composer.models.huggingface import HuggingFaceModel
except ImportError:  # pragma: no cover - optional until pretrain extras installed
    DeviceCPU = None  # type: ignore[misc, assignment]
    LanguageCrossEntropy = None  # type: ignore[misc, assignment]
    MaskedAccuracy = None  # type: ignore[misc, assignment]
    HuggingFaceModel = None  # type: ignore[misc, assignment]

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = None

try:
    from torchmetrics import Metric as TorchMetric
except ImportError:  # pragma: no cover
    TorchMetric = object  # type: ignore[misc, assignment]


def _require_pretrain_deps() -> None:
    if HuggingFaceModel is None or MaskedAccuracy is None:
        raise ImportError(
            "Composer pretrain dependencies are missing. Install with: uv sync --extra pretrain"
        )


def _coerce_arch_config(
    model_config: ModernBertArchConfig | dict | DictConfig | Path | str | None,
) -> ModernBertArchConfig:
    if model_config is None:
        return load_modernbert_arch_config(Path("configs/model/modernbert_base.yaml"))
    if isinstance(model_config, ModernBertArchConfig):
        return model_config
    if isinstance(model_config, (Path, str)) and Path(model_config).suffix in {".yaml", ".yml"}:
        return load_modernbert_arch_config(Path(model_config))
    if isinstance(model_config, DictConfig):
        model_config = OmegaConf.to_container(model_config, resolve=True)
    return ModernBertArchConfig.model_validate(model_config)


def build_modernbert_config(
    *,
    pretrained_model_name: str = "bert-base-uncased",
    model_config: ModernBertArchConfig | dict | DictConfig | Path | str | None = None,
) -> FlexBertConfig:
    """Instantiate FlexBertConfig without Composer."""
    arch = _coerce_arch_config(model_config)
    config = FlexBertConfig.from_pretrained(pretrained_model_name, **arch.to_hf_kwargs())

    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    return config


if TorchMetric is not object and HuggingFaceModel is not None:

    class EfficientCrossEntropy(TorchMetric):
        """Reads precomputed loss from FlexBERT masked_prediction outputs."""

        full_state_update = False

        def __init__(self, dist_sync_on_step: bool = False):
            super().__init__(dist_sync_on_step=dist_sync_on_step)
            self.add_state("sum_loss", default=__import__("torch").tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total_items", default=__import__("torch").tensor(0), dist_reduce_fx="sum")

        def update(self, loss):
            if isinstance(loss, torch.Tensor):
                loss = loss.detach().to(self.sum_loss.device)
            self.sum_loss += loss
            self.total_items += 1

        def compute(self):
            return self.sum_loss / self.total_items

    class EfficientHuggingFaceModel(HuggingFaceModel):
        def eval_forward(self, batch, outputs: Any | None = None):
            outputs = self.forward(batch) if outputs is None else outputs
            self.labels = batch.pop("labels")
            return outputs

        def update_metric(self, batch: Any, outputs: Any, metric: TorchMetric) -> dict:
            _require_pretrain_deps()
            if metric.device.type == "cpu":
                self.labels = DeviceCPU().batch_to_device(self.labels)

            if getattr(metric, "needs_batch", False):
                raise ValueError(f"Unsupported metric {metric=}")

            if isinstance(metric, EfficientCrossEntropy):
                metric_result = metric.update(outputs["loss"])
            else:
                metric_result = metric.update(outputs["logits"], outputs.get("labels", self.labels))

            if metric_result is not None:
                metric_result["metric_name"] = [metric.__class__.__name__ for _ in range(batch["input_ids"].shape[0])]
                return metric_result
            return {}

else:  # pragma: no cover

    class EfficientCrossEntropy:  # type: ignore[no-redef]
        pass

    class EfficientHuggingFaceModel:  # type: ignore[no-redef]
        pass


def create_modernbert_mlm(
    *,
    pretrained_model_name: str = "bert-base-uncased",
    model_config: ModernBertArchConfig | dict | DictConfig | Path | str | None = None,
    tokenizer_name: str | None = None,
    tokenizer_path: str | None = None,
    gradient_checkpointing: bool = False,
    pretrained_checkpoint: str | None = None,
    recompute_metric_loss: bool = False,
    disable_train_metrics: bool = False,
):
    """Wrap FlexBertForMaskedLM for Composer MLM pretraining."""
    _require_pretrain_deps()
    arch = _coerce_arch_config(model_config)
    config = build_modernbert_config(
        pretrained_model_name=pretrained_model_name,
        model_config=arch,
    )

    if pretrained_checkpoint is not None:
        model = FlexBertForMaskedLM.from_composer(
            pretrained_checkpoint=pretrained_checkpoint,
            config=config,
        )
    else:
        model = FlexBertForMaskedLM(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore[operator]

    if tokenizer_path:
        tokenizer_dir = resolve_hf_tokenizer_dir(tokenizer_path)
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    elif tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name)

    metrics: list[Any] = [MaskedAccuracy(ignore_index=-100)]
    loss_fn = arch.loss_function
    if recompute_metric_loss or loss_fn not in {"fa_cross_entropy", "cross_entropy"}:
        if CrossEntropyLoss is not None:
            from composer.metrics.nlp import LanguageCrossEntropy as _LCE

            metrics = [_LCE(ignore_index=-100)] + metrics
        else:
            metrics = [LanguageCrossEntropy(ignore_index=-100)] + metrics
    else:
        metrics = [EfficientCrossEntropy()] + metrics

    eval_metrics = copy.deepcopy(metrics)
    if disable_train_metrics:
        metrics = None

    hf_model = EfficientHuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        use_logits=True,
        metrics=metrics,
        eval_metrics=eval_metrics,
        allow_embedding_resizing=config.allow_embedding_resizing,
    )

    hf_model.model.resize_token_embeddings(config.vocab_size)
    return hf_model
