"""Pydantic schemas for Hydra YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from constants import validate_vocab_size
from utils.paths import resolve_from_cwd, resolve_vocab_output_dir


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

    bpe: BpeTrainerConfig
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
