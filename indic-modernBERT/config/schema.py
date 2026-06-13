"""Pydantic schemas for Hydra YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from constants import validate_vocab_size
from utils.paths import resolve_from_cwd, resolve_vocab_output_dir

PaddingMode = Literal["unpadded", "padded"]
LossFunction = Literal["cross_entropy", "fa_cross_entropy", "binary_cross_entropy", "mean_squared_error"]


class PretokenizationConfig(BaseModel):
    use_script_norm: bool = True
    use_nfkc: bool = True
    stage: Literal["subword", "superword"] = "subword"


class BpeTrainingRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vocab_size: int
    output_dir: Path


class BpeTrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_root: Path
    output_dir: Path
    text_column: str
    vocab_sizes: list[int]
    min_frequency: int = Field(ge=1)

    @field_validator("data_root", "output_dir", mode="before")
    @classmethod
    def resolve_paths(cls, value: Path | str) -> Path:
        return resolve_from_cwd(value)

    @field_validator("vocab_sizes")
    @classmethod
    def validate_vocab_sizes(cls, vocab_sizes: list[int]) -> list[int]:
        for vocab_size in vocab_sizes:
            validate_vocab_size(vocab_size)
        return vocab_sizes

    def iter_runs(self) -> list[BpeTrainingRun]:
        return [
            BpeTrainingRun(
                vocab_size=vocab_size,
                output_dir=resolve_vocab_output_dir(self.output_dir, self.vocab_sizes, vocab_size),
            )
            for vocab_size in self.vocab_sizes
        ]


class SuperBpeTrainingRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vocab_size: int
    transition_vocab_size: int | None
    output_dir: Path


class SuperBpeTrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_root: Path
    output_dir: Path
    text_column: str
    vocab_sizes: list[int]
    min_frequency: int = Field(ge=1)
    transition_fraction: float = Field(default=0.9, gt=0.0, lt=1.0)
    transition_vocab_sizes: list[int] | None = None

    @field_validator("data_root", "output_dir", mode="before")
    @classmethod
    def resolve_paths(cls, value: Path | str) -> Path:
        return resolve_from_cwd(value)

    @field_validator("vocab_sizes", "transition_vocab_sizes")
    @classmethod
    def validate_aligned_vocab_sizes(cls, vocab_sizes: list[int] | None) -> list[int] | None:
        if vocab_sizes is None:
            return None

        for vocab_size in vocab_sizes:
            validate_vocab_size(vocab_size)

        return vocab_sizes

    @model_validator(mode="after")
    def validate_transition_vocab_sizes(self) -> SuperBpeTrainerConfig:
        if self.transition_vocab_sizes is None:
            return self

        if len(self.transition_vocab_sizes) != len(self.vocab_sizes):
            raise ValueError(
                "transition_vocab_sizes must have the same length as vocab_sizes "
                f"({len(self.transition_vocab_sizes)} != {len(self.vocab_sizes)})."
            )

        for transition_vocab_size, vocab_size in zip(
            self.transition_vocab_sizes,
            self.vocab_sizes,
            strict=True,
        ):
            if transition_vocab_size >= vocab_size:
                raise ValueError(
                    f"transition_vocab_size={transition_vocab_size} must be < vocab_size={vocab_size}."
                )

        return self

    def iter_runs(self) -> list[SuperBpeTrainingRun]:
        return [
            SuperBpeTrainingRun(
                vocab_size=vocab_size,
                transition_vocab_size=(
                    self.transition_vocab_sizes[index]
                    if self.transition_vocab_sizes is not None
                    else None
                ),
                output_dir=resolve_vocab_output_dir(self.output_dir, self.vocab_sizes, vocab_size),
            )
            for index, vocab_size in enumerate(self.vocab_sizes)
        ]


class TrainerSectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bpe: BpeTrainerConfig | None = None
    superbpe: SuperBpeTrainerConfig


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pretokenization: PretokenizationConfig
    trainer: TrainerSectionConfig


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokenizer_path: Path
    data_root: Path
    text_column: str
    baseline_tokenizer_names: list[str] | None = None
    baseline_tokenizer_name: str | None = None
    reference_tokenizer_name: str | None = None
    renyi_alpha: float = Field(default=2.5, gt=0.0)
    parallel_data_path: Path | None = None
    parallel_hindi_column: str = "text_hi"
    parallel_reference_column: str = "text_eng"

    @field_validator("tokenizer_path", "data_root", "parallel_data_path", mode="before")
    @classmethod
    def resolve_paths(cls, value: Path | str | None) -> Path | None:
        if value is None:
            return None
        return resolve_from_cwd(value)

    @model_validator(mode="after")
    def validate_baselines(self) -> EvalConfig:
        if not self.baseline_tokenizer_names and not self.baseline_tokenizer_name:
            raise ValueError(
                "Expected one of `baseline_tokenizer_names` or `baseline_tokenizer_name`."
            )
        return self

    @property
    def baseline_names(self) -> list[str]:
        if self.baseline_tokenizer_names is not None:
            return self.baseline_tokenizer_names
        assert self.baseline_tokenizer_name is not None
        return [self.baseline_tokenizer_name]


EvalSection = Literal["intrinsic", "parity"]


def load_tokenizer_config(cfg: DictConfig) -> TokenizerConfig:
    tokenizer_cfg = OmegaConf.to_container(cfg.tokenizer, resolve=True)

    training_payload = {
        "pretokenization": tokenizer_cfg["pretokenization"],
        "trainer": tokenizer_cfg["trainer"],
    }
    return TokenizerConfig.model_validate(training_payload)


def load_eval_config(cfg: DictConfig, section: EvalSection) -> EvalConfig:
    tokenizer_cfg = OmegaConf.to_container(cfg.tokenizer, resolve=True)
    eval_cfg = EvalConfig.model_validate(tokenizer_cfg["evals"][section])

    if section == "parity":
        if eval_cfg.parallel_data_path is None:
            raise ValueError("`parallel_data_path` is required for parity evaluation.")
        if eval_cfg.reference_tokenizer_name is None:
            raise ValueError("`reference_tokenizer_name` is required for parity evaluation.")

    return eval_cfg


class NormKwargsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eps: float = 1e-5
    bias: bool = False

    @field_validator("eps", mode="before")
    @classmethod
    def coerce_eps(cls, value: float | str) -> float:
        return float(value)


class LossKwargsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    reduction: str = "mean"
    return_z_loss: bool = False
    lse_square_scale: float = 0.0
    inplace_backward: bool = False


class ModernBertArchConfig(BaseModel):
    """Validated ModernBERT architecture knobs from configs/model/*.yaml."""

    model_config = ConfigDict(extra="forbid")

    vocab_size: int = Field(ge=1)
    num_hidden_layers: int = Field(ge=1)
    hidden_size: int = Field(ge=1)
    intermediate_size: int = Field(ge=1)
    num_attention_heads: int = Field(ge=1)

    attention_layer: str = "rope"
    attention_probs_dropout_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    attn_out_bias: bool = False
    attn_out_dropout_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    attn_qkv_bias: bool = False

    bert_layer: str = "prenorm"
    embed_dropout_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    embed_norm: bool = True
    final_norm: bool = True
    skip_first_prenorm: bool = False
    embedding_layer: str = "sans_pos"

    loss_function: LossFunction = "cross_entropy"
    loss_kwargs: LossKwargsConfig = Field(default_factory=LossKwargsConfig)

    mlp_dropout_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    mlp_in_bias: bool = False
    mlp_layer: str = "glu"
    mlp_out_bias: bool = False

    normalization: str = "layernorm"
    norm_kwargs: NormKwargsConfig = Field(default_factory=NormKwargsConfig)

    hidden_act: str = "gelu"
    head_pred_act: str = "gelu"
    activation_function: str = "gelu"

    padding: PaddingMode = "unpadded"
    rotary_emb_dim: int | None = None
    rotary_emb_base: float = 10000.0
    rotary_emb_interleaved: bool = False

    init_method: str = "default"
    allow_embedding_resizing: bool = True
    use_fa2: bool = True

    sliding_window: int = -1
    global_attn_every_n_layers: int = -1
    local_attn_rotary_emb_base: float = -1
    local_attn_rotary_emb_dim: int | None = None

    unpad_embeddings: bool = False
    pad_logits: bool = False
    compile_model: bool = False
    masked_prediction: bool = True

    @field_validator("vocab_size")
    @classmethod
    def validate_model_vocab_size(cls, vocab_size: int) -> int:
        validate_vocab_size(vocab_size)
        return vocab_size

    @model_validator(mode="after")
    def validate_architecture(self) -> ModernBertArchConfig:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by "
                f"num_attention_heads={self.num_attention_heads}"
            )

        if "prenorm" in self.bert_layer:
            if not self.final_norm:
                raise ValueError("final_norm must be true when bert_layer uses prenorm")
        elif "postnorm" in self.bert_layer:
            if self.final_norm:
                raise ValueError("final_norm must be false when bert_layer uses postnorm")
        else:
            raise ValueError("bert_layer must contain either 'prenorm' or 'postnorm'")

        if self.loss_kwargs.return_z_loss:
            if self.loss_function != "fa_cross_entropy":
                raise ValueError("loss_function must be 'fa_cross_entropy' when return_z_loss is true")
            if self.loss_kwargs.lse_square_scale <= 0:
                raise ValueError("loss_kwargs.lse_square_scale must be > 0 when return_z_loss is true")

        if self.global_attn_every_n_layers > 0:
            if (self.num_hidden_layers - 1) % self.global_attn_every_n_layers != 0:
                raise ValueError(
                    f"global_attn_every_n_layers={self.global_attn_every_n_layers} must divide "
                    f"num_hidden_layers - 1 ({self.num_hidden_layers - 1})"
                )

        if self.sliding_window != -1:
            if not self.use_fa2:
                raise ValueError("sliding_window requires use_fa2=true (FlashAttention2)")
            if self.sliding_window % 2 != 0 or self.sliding_window % 64 != 0:
                raise ValueError(
                    f"sliding_window={self.sliding_window} must be even and divisible by 64"
                )
        else:
            if self.global_attn_every_n_layers != -1:
                raise ValueError("global_attn_every_n_layers must be -1 when sliding_window is disabled")
            if self.local_attn_rotary_emb_base != -1:
                raise ValueError("local_attn_rotary_emb_base must be -1 when sliding_window is disabled")
            if self.local_attn_rotary_emb_dim is not None:
                raise ValueError("local_attn_rotary_emb_dim must be null when sliding_window is disabled")

        if self.pad_logits and not self.unpad_embeddings:
            raise ValueError("pad_logits=true requires unpad_embeddings=true")

        if self.unpad_embeddings and self.embedding_layer == "absolute_pos":
            raise ValueError("unpad_embeddings=true is incompatible with embedding_layer='absolute_pos'")

        if self.unpad_embeddings and self.padding != "unpadded":
            raise ValueError("unpad_embeddings=true requires padding='unpadded'")

        return self

    def to_hf_kwargs(self) -> dict[str, Any]:
        payload = self.model_dump(mode="python")
        payload["loss_kwargs"] = self.loss_kwargs.model_dump(mode="python")
        payload["norm_kwargs"] = self.norm_kwargs.model_dump(mode="python")
        return payload


def load_modernbert_arch_config(path: Path | str) -> ModernBertArchConfig:
    resolved = resolve_from_cwd(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Model config not found: {resolved}")

    raw = yaml.safe_load(resolved.read_text())
    if not isinstance(raw, dict) or "model_config" not in raw:
        raise ValueError(f"{resolved} must contain a top-level 'model_config' mapping")

    return ModernBertArchConfig.model_validate(raw["model_config"])


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pretrained_model_name: str = "bert-base-uncased"
    arch_config_path: Path = Path("configs/model/modernbert_base.yaml")
    gradient_checkpointing: bool = False
    disable_train_metrics: bool = True

    @field_validator("arch_config_path", mode="before")
    @classmethod
    def resolve_paths(cls, value: Path | str) -> Path:
        return resolve_from_cwd(value)

    def load_arch(self) -> ModernBertArchConfig:
        return load_modernbert_arch_config(self.arch_config_path)


class PretrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_root: Path
    eval_data_root: Path | None = None
    tokenizer_path: Path
    output_dir: Path = Path("artifacts/model/modernbert")
    max_seq_len: int = Field(default=1024, ge=1)
    mlm_probability: float = Field(default=0.3, gt=0.0, lt=1.0)
    text_column: str = "text"
    pretrained_model_name: str = "bert-base-uncased"
    arch_config_path: Path = Path("configs/model/modernbert_base.yaml")
    gradient_checkpointing: bool = False
    disable_train_metrics: bool = True
    global_train_batch_size: int = Field(default=8, ge=1)
    device_train_microbatch_size: int = Field(default=2, ge=1)
    max_duration: str = "100ba"
    save_folder: Path = Path("artifacts/model/modernbert/checkpoints")

    @field_validator(
        "data_root",
        "eval_data_root",
        "tokenizer_path",
        "output_dir",
        "arch_config_path",
        "save_folder",
        mode="before",
    )
    @classmethod
    def resolve_paths(cls, value: Path | str | None) -> Path | None:
        if value is None:
            return None
        return resolve_from_cwd(value)

    def load_arch(self) -> ModernBertArchConfig:
        return load_modernbert_arch_config(self.arch_config_path)


def load_pretrain_config(cfg: DictConfig) -> PretrainConfig:
    pretrain_cfg = OmegaConf.to_container(cfg.pretrain, resolve=True)
    return PretrainConfig.model_validate(pretrain_cfg)
