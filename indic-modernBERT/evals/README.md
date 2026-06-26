# Hindi Evaluation Suite

This package contains the first Hindi-only evaluation suite for phase-1 Indic ModernBERT checkpoints. It is meant to answer a practical checkpoint question: is the model learning useful Hindi representations yet?

The suite is Hydra-driven through `configs/evals/hindi_phase1.yaml`. Change `eval.model.model_name_or_path` to point at any Hugging Face Hub model ID or local HF export directory, then run the same pipeline against that model.

## What It Runs

The runner combines three evaluation layers:

- **MLM holdout**: evaluates masked-language-model loss and masked accuracy on `data/eval/hi`.
- **Supervised Hindi gate**: fine-tunes and evaluates one representative task per major downstream task type.
- **Efficiency smoke**: measures a small single-GPU/CPU inference path over Hindi inputs. Broaden it manually for dedicated profiling runs.
- **Retrieval benchmark**: optional, separate `nDCG@10` evaluation for retrieval-finetuned checkpoints.

The supervised gate starts with:

- **IndicSentiment** for sentence classification.
- **Naamapadam NER** for token classification / named entity recognition.
- **IndicQA** for question answering.
- **IndicCOPA** for multiple-choice commonsense reasoning.

Retrieval is intentionally separate from the phase-1/phase-2 checkpoint gates. During MLM-only phases, retrieval scores mostly reflect an untuned pooling/indexing choice rather than the mature retrieval strength ModernBERT is known for. For fair benchmarking, first fine-tune each backbone with the same retrieval recipe, select the best checkpoint/hyperparameters, and then run `configs/evals/hindi_retrieval.yaml`.

## Why This Shape

The suite is intentionally small and repeatable. A phase-1 checkpoint may change often, so the first gate should be cheap enough to run repeatedly but broad enough to catch different failure modes:

- MLM holdout tracks whether pretraining itself is improving.
- Sentiment checks sentence-level semantics.
- NER checks token-level Hindi representations and subword boundaries.
- QA checks whether the model can connect a question to evidence in context.
- COPA checks sentence-pair reasoning and commonsense plausibility.
- Efficiency sweep checks whether the architecture is preserving ModernBERT's practical speed and memory advantages as sequence length grows.

Everything is config-first because model comparison should not require code edits. The intended workflow is to swap `eval.model.model_name_or_path`, choose task subsets with Hydra overrides, and compare outputs under `artifacts/evals/`.

## Inspirations

This package borrows from two local references:

- `_support_repo/IndicBERT`: task selection, Hindi/Indic benchmark framing, dataset choices, and metrics. Its `fine-tuning/` scripts and `eval.sh` show how IndicBERT evaluates IndicSentiment, Naamapadam, IndicQA, IndicCOPA, XNLI, paraphrase, and retrieval.
- `_support_repo/ModernBERT`: evaluation orchestration and efficiency measurement style. `RunEvals.md` motivates config-driven fine-tuning runs across tasks/seeds, while `benchmark.py` motivates reporting latency, tokens/sec, tokens/sec per million parameters, memory, and optional GPU power.

The result is not a direct copy of either repo. It keeps the IndicBERT task families, but wraps them in this repo's Hydra workflow and adds ModernBERT-style efficiency reporting.

## How To Run

Full configured suite:

```bash
uv sync --extra evals
make run-evals ARGS="eval.model.model_name_or_path=<hf-id-or-local-hf-export>"
```

Tiny smoke path:

```bash
make run-evals-smoke ARGS="eval.model.model_name_or_path=hf-internal-testing/tiny-random-bert"
```

Useful overrides:

```bash
make run-evals ARGS="eval.model.model_name_or_path=<model> eval.tasks='[sentiment,ner]'"
make run-evals ARGS="eval.model.model_name_or_path=<model> eval.efficiency.sequence_lengths='[128,1024]'"
make run-evals ARGS="eval.model.model_name_or_path=<model> eval.efficiency.measure_power=true"
make run-evals ARGS="eval.model.model_name_or_path=<model> eval.model.context_mode=model_max eval.model.max_sequence_length=1024"
make run-evals-retrieval ARGS="eval.models.0.model_name_or_path=<retrieval-finetuned-checkpoint>"
```

Outputs are written under `artifacts/evals/<model-slug>__<context>/`, for example
`xlm-roberta-base__common_128` followed by `xlm-roberta-base__model_max_512`:

- `suite_summary.json`
- `suite_metrics.csv`
- `suite_report.md`
- per-task supervised metrics
- `efficiency_metrics.json`
- `retrieval_metrics.json` for retrieval runs

## Benchmark Policy

Use HF-native ModernBERT checkpoints for benchmark runs. With `transformers>=4.57.1`, Auto classes can load checkpoints whose `config.json` uses `model_type: "modernbert"`, so the preferred path is `trust_remote_code: false` with no custom `auto_map` fallback.

Local Composer checkpoints should be exported before evaluation. The current export entry point is:

```bash
python scripts/export_hf.py <composer-checkpoint.pt> <hf-export-dir> \
  --config configs/model/modernbert_base.yaml \
  --tokenizer artifacts/tokenizer/bpe_vs50368
```

The exporter writes HF-native ModernBERT artifacts and verifies that `<hf-export-dir>/config.json` has `"model_type": "modernbert"` before returning.

`tokenizer_name_or_path: null` intentionally falls back to `model_name_or_path`, so model and tokenizer stay coupled unless a run explicitly overrides the tokenizer.

The supervised tasks support two sequence-length modes:

- `common_128`: caps all supervised tasks at 128 tokens for fair comparison with older IndicBERT-style baselines.
- `model_max`: caps each task by `eval.model.max_sequence_length`, while still respecting the task's own configured upper bound.

For multi-model suites, list each checkpoint once under `eval.models` and set `eval.context_modes`
to the ordered passes to run, for example `[common_128, model_max]`. The loader expands this into
one run per checkpoint/context pair and skips `model_max` when a checkpoint already has
`max_sequence_length: 128`.

When `eval.efficiency.sequence_lengths: null`, the efficiency sweep uses the active context length for that pass:
128 for `common_128`, and `eval.model.max_sequence_length` for `model_max`.

The current phase-1 suite keeps efficiency as smoke coverage, not a final benchmark. TODOs that can be done sometime reporting: add multi-seed averaging and optional hyperparameter optimization.

Retrieval follows the upstream ModernBERT evaluation policy:

- Train/fine-tune every backbone with the same retrieval recipe before comparing scores.
- Report `nDCG@10` as the headline metric.
- Use `AIhnIndicRag/mmarco_hindi` for Hindi MS-MARCO-style retrieval quality.
- Use `Shitao/MLDR` with `language=hi` for long-document retrieval capacity at 8192 context.


### Learnings and checks : 

The notes below are historical scratch notes from reviewing the suite. The settled benchmark policy above is the source of truth for current behavior.

Query : `_support_repo/IndicBERT` has this sub-project commit and it has a commit 4890441925986d4523aa1c8fe01c6dfa06abe7cd which is the hash of head of IndicBERT repository and we can call it using `git submodule update --init` and we can remove it using `git rm --cached _support_repo/IndicBERT` which I think we should do it. 

Hydra : 

It has the log directory as  `dir: logs/${now:%Y-%m-%d}_${now:%H-%M-%S}_evals` which means that it will store the logs as something like logs/2026-04-20_time which as I realize typing this that is all good because all the evals will be container there but one major concern will be that there is a chance that in `supervised` we don't have the tuned evals but due to time constraints have to keep that as it is ( maybe latter we do hyper-parameter optimization )

Also huggingface would probably put you in error when you have `trust_remote_code:false` and the code should have a fallback when `tokenizer_name_or_path:null` should resolve to taking the tokenizer of that particular model, it can be resolved with `autoTokenizer.from_model()` don't really know the exact method but this should be good. 

Ideally a good experimentation should be that we randomize the seeds and average the performance over various seeds but due to time constraints we won't be able to do that. 

For all the models we should be careful with `max_seq_length` because they don't support more than `128` and I don't think 

```yaml
sample_texts:
      - भारत में हिंदी भाषा अनेक रूपों में बोली और लिखी जाती है।
      - यह मूल्यांकन छोटे बैचों पर अनुमान गति और स्मृति उपयोग मापता है।
```

Will be enough to measure the efficiency of the models. 

For config management, one good rule is that which one should follow is to use `Pydantic` because of its validation features like `weight_decay: float = Field(default=0.01, ge=0.0)` here **ge** suggests that this value should be greater or equal to something right, but 

```python
fp16: bool = False
bf16: bool = False
```
values should be computed automatically because it depends on the device if `cuda` is available then we can use `bf16` right and these should be exposed to via hydra which are not but by default values should be `torch.is_cuda_availabble()` then set bf16 to true. 

I don't understand why in efficiency config we need warmup steps right ? `warmup_steps: int = Field(default=2, ge=0)` and are these `measured_steps: int = Field(default=5, ge=1)` going to be enough ? don't think so and also this `use_bf16_autocast: bool = True` should be dependent on whether cuda is available or not right ? and again the same concern that whether these samples_texts would be enough to measure the efficiency. I think we should check with the reference i.e. implementation of `modernBERT` on how they are going it and apply our fixes. 

```yaml
 @field_validator("sequence_lengths")
    @classmethod
    def validate_lengths(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("sequence_lengths must contain at least one length")
        if any(length < 1 for length in value):
            raise ValueError("sequence_lengths must be positive")
        return value
```

Can't we replace this with `ge>0` or something using `Field()`. 

We need to remove `device: Literal["auto", "cpu", "cuda", "mps"] = "auto"` the MPS support and make this call like torch.cuda_is_available() to do this. 

```python
@field_validator("output_dir", mode="before")
    @classmethod
    def resolve_path(cls, value: Path | str) -> Path:
        return resolve_from_cwd(value)
```

Don't you think if the path doesn't exist we should do something like `mkdir` I believe there is a python command which allows you to do this. 

```python
@field_validator("output_dir", mode="before")
    @classmethod
    def resolve_path(cls, value: Path | str) -> Path:
        return resolve_from_cwd(value)
why do we need to convert it to json ? we can just use the OmegaConf right ? 
```

Moving on to efficiency.py, we have : 

```python
model_cls = AutoModelForMaskedLM if eff_cfg.use_mlm_head else AutoModel
    device = choose_device(cfg.device)
    model = model_cls.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
    ).to(device)
```
But why would be ever run efficiency on `AutoModelForMaskedLM`

```python
for seq_len in eff_cfg.sequence_lengths:
        batch = _build_batch(tokenizer, eff_cfg.sample_texts, eff_cfg.batch_size, seq_len) # is this using unpadding ? 
        batch = {key: value.to(device) for key, value in batch.items()}
```

at all the places we should just have torch.is_cuda_available() instead of this `if device.type == "cuda":`. 

I think there is an simplication opportunity here `_gpu_power_watts(eff_cfg.gpu_index)` we are only going to use one GPU for our testing. 

```python
tokens = eff_cfg.batch_size * seq_len
```

Just ensure that we are building good batch sizes of the exact sequence lengths. 

```python
rows.append(
            {
                "sequence_length": seq_len,
                "batch_size": eff_cfg.batch_size,
                "num_parameters": num_parameters,
                "num_parameters_m": num_parameters_m,
                "latency_mean_s": mean_latency,
                "latency_std_s": std_latency,
                "examples_per_second": examples / mean_latency if mean_latency else float("inf"),
                "tokens_per_second": tokens_per_second,
                "tokens_per_second_per_million_params": tokens_per_second / num_parameters_m if num_parameters_m else None,
                "peak_cuda_memory_allocated_mb": _peak_cuda_memory_allocated_mb(device),
                "peak_cuda_memory_reserved_mb": _peak_cuda_memory_reserved_mb(device),
                "avg_power_watts": statistics.fmean(power_readings) if power_readings else None,
                "max_power_watts": max(power_readings) if power_readings else None,
                "device": str(device),
            }
        )

    result = {
        "name": "efficiency_sweep",
        "type": "efficiency",
        "status": "completed",
        "metrics": {"lengths": rows},
        "config": {
            "sequence_lengths": eff_cfg.sequence_lengths,
            "batch_size": eff_cfg.batch_size,
            "warmup_steps": eff_cfg.warmup_steps,
            "measured_steps": eff_cfg.measured_steps,
            "use_mlm_head": eff_cfg.use_mlm_head,
            "use_bf16_autocast": eff_cfg.use_bf16_autocast,
            "measure_power": eff_cfg.measure_power,
            "gpu_index": eff_cfg.gpu_index,
        },
    }
    (output_dir / "efficiency_metrics.json").write_text(_json_dumps(result), encoding="utf-8")
    return result

def _json_dumps(payload: object) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
```

I think creating a function resuable one and keep it seperate in like helpers.py in utils should work well because the loggers are also there and this shouldn't be there in the core measurement logic. 

```python
@torch.no_grad()
def _forward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    use_bf16_autocast: bool,
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if device.type == "cuda" and use_bf16_autocast:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(**batch)
    else:
        model(**batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def _peak_cuda_memory_allocated_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))


def _peak_cuda_memory_reserved_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_reserved(device) / (1024 * 1024))
```

I think there is simplication opportunities available here to automatically detect and add the support towards one gpu, because really we won't need more than that. Use `torch_is_cuda_available()` for throw value erros which can be defined explicity in errors.py file or something, so that we have custom errors. 


TaskRegistry : The area where all tasks get defined. Here we are using `@dataclass = frozen` which means that after the object is created we cannot change it which is required with tasks configs. 

Why do we have `remove_columns` because when we run `.map(tokenize_)` the dataset goes from raw columns to model-ready columns. Cool right ? Add stuff like okay it is taking labels from hugging face datasets, so that we are sure of things happening. 


```python
"qa": TaskSpec(
        name="qa",
        display_name="IndicQA Hindi",
        task_type="question_answering",
        dataset_name="ai4bharat/IndicQA",
        dataset_config="indicqa.hi",
        metric_names=("exact_match", "f1"),
        label_column="answers",
        text_columns=("question", "context"),
        max_seq_length=384,
        extra={"doc_stride": 128, "n_best": 20, "max_answer_length": 30},
    ),
    "copa": TaskSpec(
        name="copa",
        display_name="IndicCOPA Hindi",
        task_type="multiple_choice",
        dataset_name="ai4bharat/IndicCOPA",
        dataset_config="translation-hi",
        metric_names=("accuracy",),
        label_column="label",
        text_columns=("premise", "question", "choice1", "choice2"),
        max_seq_length=512,
    ),
```
Do you think other models having sequence length of around 128, will be able to handle `max_seq_len=512`? 

In runtime.py would prefer if : 

```python
if torch.backends.mps.is_available():
            return torch.device("mps")
``` 
Because never going to use MPS to do my testings. Yup loved that `run_evals.py` is in scripts because it is the main entry point for this. 

Why in the naming we have : 

```python
from evals.tasks.multiple_choice import run_multiple_choice
from evals.tasks.question_answering import run_question_answering
from evals.tasks.sequence_classification import run_sequence_classification
from evals.tasks.token_classification import run_token_classification

__all__ = [
    "run_multiple_choice",
    "run_question_answering",
    "run_sequence_classification",
    "run_token_classification",
]
```

We can just do something like multiple_choice or question_answering. There is no need to anything else right ? Again posing the same question because in `common.py` we have warmup_ratio, this is not an extended training run so I don't think warmup ratio is required, we can always just set it to 0.0. 

Just asking, there are libraries, which calculate 

```python
def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for label in sorted(set(labels.tolist()) | set(preds.tolist())):
        tp = int(((preds == label) & (labels == label)).sum())
        fp = int(((preds == label) & (labels != label)).sum())
        fn = int(((preds != label) & (labels == label)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else 0.0
``` 
and other metrics pretty well, so why are we doing it ourselves ? 

Coming to the tasks now : 

For a very good comparision, we can have models using there maximum context length because having a short context length is also the fault of the models right and our models having a longer context length changes the game right ? so we can compare at two context length. 

```python
inputs = tokenizer(
            questions,
            examples["context"],
            max_length=task_cfg.max_seq_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
```





