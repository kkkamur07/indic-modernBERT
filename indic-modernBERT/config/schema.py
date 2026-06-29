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


class TrainerSectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bpe: BpeTrainerConfig


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pretokenization: PretokenizationConfig
    trainer: TrainerSectionConfig


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokenizer_path: Path | None = None
    data_root: Path
    text_column: str
    baseline_tokenizer_names: list[str] | None = None
    baseline_tokenizer_name: str | None = None
    reference_tokenizer_name: str | None = None
    renyi_alpha: float = Field(default=2.5, gt=0.0)
    max_shards: int | None = Field(
        default=None,
        ge=1,
        description="Cap parquet shards under data_root (sorted); None uses all.",
    )

    @field_validator("tokenizer_path", "data_root", mode="before")
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


EvalSection = Literal["intrinsic"]


def load_tokenizer_config(cfg: DictConfig) -> TokenizerConfig:
    tokenizer_cfg = OmegaConf.to_container(cfg.tokenizer, resolve=True)

    training_payload = {
        "pretokenization": tokenizer_cfg["pretokenization"],
        "trainer": tokenizer_cfg["trainer"],
    }
    return TokenizerConfig.model_validate(training_payload)


def load_eval_config(cfg: DictConfig, section: EvalSection) -> EvalConfig:
    tokenizer_cfg = OmegaConf.to_container(cfg.tokenizer, resolve=True)
    return EvalConfig.model_validate(tokenizer_cfg["evals"][section])


class NormKwargsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eps: float = 1e-5
    bias: bool = False

    @field_validator("eps", mode="before")
    @classmethod
    def coerce_eps(cls, value: float | str) -> float:
        return float(value)


class HardwareAlignmentConfig(BaseModel):
    """Optional tile/wave checks for LM-head GEMM ablations (see scripts/ablate_hardware_alignment.py)."""

    model_config = ConfigDict(extra="forbid")

    enforce: bool = Field(
        default=False,
        description="When true, fail validation if tile/wave alignment does not pass.",
    )
    tile_m: int = Field(default=128, ge=1, description="Vocab (N) tile for LM-head GEMM.")
    tile_n: int = Field(default=256, ge=1, description="Hidden (K) tile for LM-head GEMM.")
    tensor_align: int = Field(default=64, ge=1, description="Tensor-core alignment (divisibility).")
    sm_count: int | None = Field(
        default=None,
        ge=1,
        description="Streaming-multiprocessor count for wave check; None skips wave.",
    )


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
    hardware_alignment: HardwareAlignmentConfig = Field(default_factory=HardwareAlignmentConfig)

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

        if self.hardware_alignment.enforce:
            from config.hardware_alignment import validate_hardware_alignment

            validate_hardware_alignment(self)

        return self

    def to_hf_kwargs(self) -> dict[str, Any]:
        payload = self.model_dump(mode="python", exclude={"hardware_alignment"})
        payload["loss_kwargs"] = self.loss_kwargs.model_dump(mode="python")
        payload["norm_kwargs"] = self.norm_kwargs.model_dump(mode="python")
        return payload


def _deep_merge_mapping(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_mapping(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_modernbert_arch_config(path: Path | str) -> ModernBertArchConfig:
    resolved = resolve_from_cwd(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Model config not found: {resolved}")

    raw = yaml.safe_load(resolved.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{resolved} must be a YAML mapping")

    if "extends" in raw:
        parent_path = raw["extends"]
        if not isinstance(parent_path, str):
            raise ValueError(f"{resolved}: 'extends' must be a path string")
        parent = load_modernbert_arch_config(
            parent_path if Path(parent_path).is_absolute() else resolved.parent / parent_path
        )
        overrides = raw.get("model_config", {})
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, dict):
            raise ValueError(f"{resolved}: 'model_config' overrides must be a mapping")
        merged = _deep_merge_mapping(parent.model_dump(mode="python"), overrides)
        return ModernBertArchConfig.model_validate(merged)

    if "model_config" not in raw:
        raise ValueError(f"{resolved} must contain 'model_config' or 'extends'")

    model_config = raw["model_config"]
    if not isinstance(model_config, dict):
        raise ValueError(f"{resolved}: 'model_config' must be a mapping")
    return ModernBertArchConfig.model_validate(model_config)


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


class OptimizerConfig(BaseModel):
    """Matches _support_repo/ModernBERT/yamls/modernbert/modernbert-base-pretrain.yaml."""

    model_config = ConfigDict(extra="forbid")

    name: str = "decoupled_stableadamw"
    lr: float = Field(default=8e-4, gt=0.0)
    beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    beta2: float = Field(default=0.98, gt=0.0, lt=1.0)
    eps: float = Field(default=1e-6, gt=0.0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    filter_bias_norm_wd: bool = True
    log_grad_norm: bool = True


class SchedulerConfig(BaseModel):
    """Matches upstream scheduler section in modernbert-base-pretrain.yaml."""

    model_config = ConfigDict(extra="forbid")

    name: str = "warmup_stable_decay"
    t_warmup: str = "100ba"
    alpha_f: float = 0.0
    t_decay: str = "0tok"
    t_cooldown: str = "0.1dur"
    t_cosine: str = "0.25dur"
    t_max: str = "1dur"
    alpha_s: float = 0.0
    warmup_schedule: str = "linear"
    cooldown_schedule: str = "linear"


class PretrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_root: Path
    eval_data_root: Path | None = None
    tokenizer_path: Path
    output_dir: Path = Path("artifacts/model/modernbert/hi")
    max_seq_len: int = Field(default=1024, ge=1)
    mlm_probability: float = Field(default=0.3, gt=0.0, lt=1.0)
    eval_mlm_probability: float = Field(
        default=0.15,
        gt=0.0,
        lt=1.0,
        description="Eval masking rate; upstream uses 0.15 vs 0.3 train.",
    )
    text_column: str = "text"
    pretrained_model_name: str = "bert-base-uncased"
    arch_config_path: Path = Path("configs/model/modernbert_base.yaml")
    gradient_checkpointing: bool = False
    disable_train_metrics: bool = True
    global_train_batch_size: int = Field(default=8, ge=1)
    device_train_microbatch_size: int = Field(default=2, ge=1)
    max_duration: str = "100ba"
    run_name: str | None = Field(
        default=None,
        description="Composer run name; required by Composer when autoresume=True.",
    )
    save_folder: Path = Path("artifacts/model/modernbert/hi/checkpoints")
    save_interval: str = Field(
        default="1000ba",
        description="Composer checkpoint save interval (e.g. 4000ba).",
    )
    save_num_checkpoints_to_keep: int = Field(
        default=-1,
        description="Checkpoints to retain on disk; -1 keeps all.",
    )
    save_overwrite: bool = Field(
        default=False,
        description="Overwrite conflicting checkpoints in save_folder (smoke reruns).",
    )
    autoresume: bool | None = Field(
        default=None,
        description="Resume from latest checkpoint in save_folder on restart.",
    )
    progress_bar: bool = Field(
        default=False,
        description="Show Composer tqdm-style progress bar.",
    )
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    seed: int = 17
    precision: str = "amp_bf16"
    log_to_console: bool = True
    console_log_interval: str = "100ba"
    num_workers: int = Field(default=6, ge=0, description="Train DataLoader workers.")
    dataloader_prefetch_factor: int = Field(default=4, ge=1, description="Train prefetch_factor.")
    eval_num_workers: int = Field(
        default=3,
        ge=0,
        description="Eval DataLoader workers (upstream eval_loader uses 3).",
    )
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = Field(
        default=True,
        description=(
            "Keep DataLoader workers alive across epochs. Disable for in-process "
            "multi-run (Optuna sweep) so workers are reaped between trials and RAM "
            "does not accumulate to OOM."
        ),
    )
    max_train_shards: int | None = Field(
        default=None,
        ge=1,
        description="Cap parquet shards for training; None streams all shards.",
    )
    callbacks: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {
            "speed_monitor": {"window_size": 100},
            "lr_monitor": {},
            "log_grad_norm": {"batch_log_interval": 10},
            "packing_efficiency": {"log_interval": 10},
        }
    )
    loggers: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Composer loggers (e.g. tensorboard with log_dir).",
    )
    sequence_packing: bool = Field(
        default=False,
        description="GreedyBestFitSequencePacker train path (upstream modernbert-base-pretrain.yaml).",
    )
    eval_sequence_packing: bool = Field(
        default=False,
        description="Upstream eval_loader uses sequence_packing: false (padded MLM @ 0.15).",
    )
    packing_buffer_size: int | None = Field(
        default=None,
        ge=1,
        description="Packer buffer; default 5 * global_train_batch_size.",
    )
    packing_prefetch_factor: int = Field(default=5, ge=1)
    batch_size_warmup_min_size: int | None = Field(default=None, ge=1)
    batch_size_warmup_tokens: str | int | None = None
    load_path: Path | None = Field(
        default=None,
        description="Composer checkpoint to resume (e.g. phase-1 latest-rank0.pt for context extension).",
    )
    reset_time: bool = Field(
        default=False,
        description="Restart Composer timestamp / schedulers from zero on resume (upstream context extension).",
    )
    restart_override: bool = Field(
        default=False,
        description="After load_path, apply LR/WD/microbatch from this yaml instead of the checkpoint.",
    )
    eval_interval: str | None = Field(
        default=None,
        description="Composer eval interval (e.g. 4000ba); eval runs only when set with eval_data_root.",
    )
    global_eval_batch_size: int | None = Field(default=None, ge=1)
    device_eval_microbatch_size: int | None = Field(default=None, ge=1)
    eval_subset_num_batches: int = Field(
        default=-1,
        description="Cap eval batches when eval dataloader has no length (-1 = full pass).",
    )
    shuffle_seed: int = 42
    drop_last: bool = False
    count_padding_tokens: bool = Field(
        default=False,
        description="If false, token schedulers count only non-pad tokens (upstream default).",
    )

    @field_validator(
        "data_root",
        "eval_data_root",
        "tokenizer_path",
        "output_dir",
        "arch_config_path",
        "save_folder",
        "load_path",
        mode="before",
    )
    @classmethod
    def resolve_paths(cls, value: Path | str | None) -> Path | None:
        if value is None:
            return None
        return resolve_from_cwd(value)

    @field_validator("tokenizer_path", mode="after")
    @classmethod
    def normalize_tokenizer_dir(cls, value: Path) -> Path:
        from utils.paths import resolve_hf_tokenizer_dir

        return resolve_hf_tokenizer_dir(value)

    @model_validator(mode="after")
    def validate_training_setup(self) -> PretrainConfig:
        if self.device_train_microbatch_size > self.global_train_batch_size:
            raise ValueError(
                "device_train_microbatch_size cannot exceed global_train_batch_size "
                f"({self.device_train_microbatch_size} > {self.global_train_batch_size})"
            )

        if self.global_train_batch_size % self.device_train_microbatch_size != 0:
            raise ValueError(
                "global_train_batch_size must be divisible by device_train_microbatch_size "
                f"({self.global_train_batch_size} % {self.device_train_microbatch_size} != 0)"
            )

        arch = self.load_arch()
        tokenizer_json = self.tokenizer_path / "tokenizer.json"
        if tokenizer_json.is_file():
            tokenizer_vocab_size = _read_tokenizer_vocab_size(self.tokenizer_path)
            if arch.vocab_size != tokenizer_vocab_size:
                raise ValueError(
                    f"arch vocab_size={arch.vocab_size} does not match tokenizer "
                    f"vocab_size={tokenizer_vocab_size} at {self.tokenizer_path}"
                )

        return self

    @property
    def grad_accum_steps(self) -> int:
        """Optimizer steps per global batch (Composer ``device_train_microbatch_size``)."""
        return self.global_train_batch_size // self.device_train_microbatch_size

    def load_arch(self) -> ModernBertArchConfig:
        return load_modernbert_arch_config(self.arch_config_path)


class PretrainJobConfig(BaseModel):
    """Top-level Hydra job config: validates the `pretrain:` YAML section via Pydantic."""

    model_config = ConfigDict(extra="ignore")

    pretrain: PretrainConfig


def _read_tokenizer_vocab_size(tokenizer_dir: Path) -> int:
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"tokenizer.json not found under {tokenizer_dir}")

    raw = yaml.safe_load(tokenizer_json.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid tokenizer.json at {tokenizer_json}")

    model = raw.get("model")
    if isinstance(model, dict):
        vocab = model.get("vocab")
        if isinstance(vocab, dict):
            return len(vocab)

    raise ValueError(f"Could not read vocab size from {tokenizer_json}")


def load_pretrain_config(cfg: DictConfig) -> PretrainConfig:
    if "pretrain" not in cfg:
        raise ValueError(
            "Hydra config must contain a top-level 'pretrain' key "
            "(e.g. configs/hi/pretrain/hindi_mlm.yaml or configs/hi/sweep/hindi_mlm_lr_sweep.yaml)."
        )

    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise ValueError("Resolved Hydra config must be a mapping")

    return PretrainJobConfig.model_validate(container).pretrain
