"""Pydantic schemas for Hydra-driven evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from utils.paths import resolve_from_cwd

ContextMode = Literal["common_128", "model_max"]


class ModelEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name_or_path: str
    tokenizer_name_or_path: str | None = None
    trust_remote_code: bool = False
    max_sequence_length: int = Field(default=128, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    context_mode: ContextMode = "common_128"

    @property
    def tokenizer_source(self) -> str:
        return self.tokenizer_name_or_path or self.model_name_or_path


class SupervisedDefaultsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    do_train: bool = True
    do_eval: bool = True
    max_train_samples: int | None = Field(default=None, ge=1)
    max_eval_samples: int | None = Field(default=None, ge=1)
    num_train_epochs: float = Field(default=3.0, gt=0.0)
    learning_rate: float = Field(default=3e-5, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    per_device_train_batch_size: int = Field(default=16, ge=1)
    per_device_eval_batch_size: int = Field(default=32, ge=1)
    max_seq_length: int = Field(default=128, ge=1)
    fp16: bool = False
    bf16: bool = True
    save_total_limit: int = Field(default=1, ge=1)
    report_to: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_mixed_precision(self) -> SupervisedDefaultsConfig:
        if self.fp16 and self.bf16:
            raise ValueError("fp16 and bf16 cannot both be enabled for supervised evaluation")
        return self


class TaskOverrideConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    do_train: bool | None = None
    do_eval: bool | None = None
    max_train_samples: int | None = Field(default=None, ge=1)
    max_eval_samples: int | None = Field(default=None, ge=1)
    num_train_epochs: float | None = Field(default=None, gt=0.0)
    learning_rate: float | None = Field(default=None, gt=0.0)
    weight_decay: float | None = Field(default=None, ge=0.0)
    warmup_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    per_device_train_batch_size: int | None = Field(default=None, ge=1)
    per_device_eval_batch_size: int | None = Field(default=None, ge=1)
    max_seq_length: int | None = Field(default=None, ge=1)
    fp16: bool | None = None
    bf16: bool | None = None

    @model_validator(mode="after")
    def validate_mixed_precision(self) -> TaskOverrideConfig:
        if self.fp16 is True and self.bf16 is True:
            raise ValueError("fp16 and bf16 cannot both be enabled in a task override")
        return self

    def apply_to(self, defaults: SupervisedDefaultsConfig) -> SupervisedDefaultsConfig:
        payload = defaults.model_dump()
        for key, value in self.model_dump(exclude_none=True).items():
            payload[key] = value
        return SupervisedDefaultsConfig.model_validate(payload)


class MlmEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    data_root: Path = Path("data/eval/hi")
    text_column: str = "text"
    max_seq_length: int | None = Field(default=None, ge=1)
    mlm_probability: float = Field(default=0.15, gt=0.0, lt=1.0)
    batch_size: int = Field(default=8, ge=1)
    max_shards: int | None = Field(default=None, ge=1)
    max_samples: int | None = Field(default=1024, ge=0)
    max_batches: int | None = Field(default=None, ge=1)
    num_workers: int = Field(default=0, ge=0)
    seed: int = 17

    @field_validator("data_root", mode="before")
    @classmethod
    def resolve_path(cls, value: Path | str) -> Path:
        return resolve_from_cwd(value)


class EfficiencyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    sequence_lengths: list[Annotated[int, Field(ge=1)]] | None = None
    batch_size: int = Field(default=8, ge=1)
    warmup_steps: int = Field(default=2, ge=0)
    measured_steps: int = Field(default=5, ge=1)
    use_mlm_head: bool = False
    use_bf16_autocast: bool = True
    measure_power: bool = False
    sample_texts: list[str] = Field(
        default_factory=lambda: [
            "भारत में हिंदी भाषा अनेक रूपों में बोली और लिखी जाती है।",
            "यह मूल्यांकन छोटे बैचों पर अनुमान गति और स्मृति उपयोग मापता है।",
        ]
    )

    @field_validator("sequence_lengths")
    @classmethod
    def validate_lengths(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and not value:
            raise ValueError("sequence_lengths must contain at least one length")
        return value


class RetrievalDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["hf_beir", "mldr"]
    dataset_name: str
    enabled: bool = True
    corpus_config: str | None = None
    corpus_split: str = "corpus"
    queries_config: str | None = None
    queries_split: str = "queries"
    qrels_config: str | None = None
    qrels_split: str = "test"
    language: str | None = None
    max_corpus_docs: int | None = Field(default=None, ge=1)
    max_queries: int | None = Field(default=None, ge=1)
    trust_remote_code: bool = False


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # We are not exposing some of the configs here. 
    enabled: bool = False
    datasets: list[RetrievalDatasetConfig] = Field(default_factory=list)
    max_seq_length: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=32, ge=1)
    corpus_chunk_size: int = Field(default=50_000, ge=1)
    top_k: int = Field(default=10, ge=1)
    trust_remote_code: bool | None = None

    @model_validator(mode="after")
    def validate_datasets(self) -> RetrievalConfig:
        if self.enabled and not self.datasets:
            raise ValueError("eval.retrieval.datasets must contain at least one dataset when retrieval is enabled")
        return self


class ReportingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    write_json: bool = True
    write_csv: bool = True
    write_markdown: bool = True


class EvalSuiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelEvalConfig
    models: list[ModelEvalConfig] = Field(default_factory=list)
    context_modes: list[ContextMode] | None = None
    output_dir: Path = Path("artifacts/evals/hi")
    seed: int = 17
    device: Literal["auto", "cpu", "cuda"] = "auto"
    tasks: list[str] = Field(default_factory=lambda: ["sentiment", "ner", "qa", "copa"])
    supervised: SupervisedDefaultsConfig = Field(default_factory=SupervisedDefaultsConfig)
    task_overrides: dict[str, TaskOverrideConfig] = Field(default_factory=dict)
    mlm: MlmEvalConfig = Field(default_factory=MlmEvalConfig)
    efficiency: EfficiencyConfig = Field(default_factory=EfficiencyConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @field_validator("output_dir", mode="before")
    @classmethod
    def resolve_path(cls, value: Path | str) -> Path:
        return resolve_from_cwd(value)

    def task_config(self, name: str) -> SupervisedDefaultsConfig:
        override = self.task_overrides.get(name)
        if override is None:
            return self.supervised
        return override.apply_to(self.supervised)

    @model_validator(mode="after")
    def populate_models(self) -> EvalSuiteConfig:
        base_models = self.models or [self.model]

        if self.context_modes is None:
            self.models = base_models
        else:
            self.models = [
                model.model_copy(update={"context_mode": mode})
                for model in base_models
                for mode in self.context_modes
                if mode != "model_max" or model.max_sequence_length != 128
            ]

        if not self.models:
            raise ValueError("eval.models must contain at least one model after context expansion")
        
        self.model = self.models[0]
        return self

    def for_model(self, model: ModelEvalConfig) -> EvalSuiteConfig:
        return self.model_copy(update={"model": model, "models": [model]}, deep=True)


class EvalJobConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    eval: EvalSuiteConfig


def load_eval_suite_config(cfg: DictConfig) -> EvalSuiteConfig:
    if "eval" not in cfg:
        raise ValueError("Hydra config must contain a top-level 'eval' key.")
    eval_container = OmegaConf.to_container(cfg.eval, resolve=True)
    if not isinstance(eval_container, dict):
        raise ValueError("Resolved eval config must be a mapping")
        
    if "model" not in eval_container and "models" in eval_container:
        models = eval_container["models"]

        if not isinstance(models, list) or not models:
            raise ValueError("eval.models must contain at least one model when eval.model is omitted")
        eval_container["model"] = models[0]

    elif "model" in eval_container and "models" not in eval_container:
        eval_container["models"] = [eval_container["model"]]

    return EvalSuiteConfig.model_validate(eval_container)


def config_to_jsonable(cfg: DictConfig) -> dict[str, Any]:
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("Resolved Hydra config must be a mapping")
    return payload
