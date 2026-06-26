"""Retrieval fine-tuning pipeline mirroring upstream ModernBERT's DPR recipe.

Upstream trains every backbone with the same recipe (MS-MARCO, contrastive loss,
select by nDCG@10 on a BEIR subset), then reports full BEIR scores. This module
implements that same pipeline for Hindi checkpoints, but trains on the
Hindi-translated MS-MARCO triples (unicamp-dl/mmarco, config "hindi") so the
contrastive signal and in-training dev evaluator are Hindi rather than English.

Reference: AnswerDotAI/ModernBERT/examples/train_st.py (Apache-2.0)
Paper: Appendix E.2 — select by avg nDCG@10 on {NFCorpus, SciFact, TREC-Covid,
FiQA}. LR exploration is handled by Hydra Optuna configs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class RetrievalFinetuneConfig:
    """Parsed from the Hydra YAML retrieval_ft block."""

    backbone: str = "answerdotai/ModernBERT-base"
    trust_remote_code: bool = False
    max_seq_length: int = 8192

    train_dataset: str = "unicamp-dl/mmarco"
    train_dataset_config: str = "hindi"
    train_dataset_split: str = "train"
    # mMARCO ships as a loading script, so it needs trust_remote_code at load time.
    train_trust_remote_code: bool = True
    max_train_samples: int = 1_250_000

    loss: str = "CachedMultipleNegativesRankingLoss"
    mini_batch_size: int = 16

    num_train_epochs: int = 1
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 8
    learning_rate: float = 8e-5
    warmup_ratio: float = 0.05
    bf16: bool = True
    fp16: bool = False
    batch_sampler: str = "NO_DUPLICATES"

    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 2
    logging_steps: int = 100

    eval_strategy: str = "steps"
    eval_steps: int = 500
    eval_split_size: int = 1000
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_train-dev_cosine_accuracy"
    greater_is_better: bool = True
    early_stopping_patience: int | None = 2
    early_stopping_threshold: float = 0.0

    # --- Model selection (Hindi) ---
    # LR selection ranks a carved subset of a Hindi retrieval benchmark by
    # nDCG@10. Defaults to a subset of the same mmarco_hindi used at eval time,
    # so selection and reporting share the same language and benchmark family.
    selection_dataset_name: str = "AIhnIndicRag/mmarco_hindi"
    selection_kind: str = "hf_beir"
    selection_language: str | None = None
    selection_trust_remote_code: bool = False
    selection_corpus_config: str | None = "corpus"
    selection_corpus_split: str = "corpus"
    selection_queries_config: str | None = "queries"
    selection_queries_split: str = "queries"
    selection_qrels_config: str | None = "default"
    selection_qrels_split: str = "test"
    selection_max_corpus_docs: int | None = 100_000
    selection_max_queries: int | None = 1_000
    selection_top_k: int = 10
    selection_batch_size: int = 32
    selection_corpus_chunk_size: int = 50_000

    output_dir: str = "artifacts/retrieval_finetune"
    seed: int = 17
    eval_config: str = "hindi_retrieval"


def run_retrieval_finetune(cfg: RetrievalFinetuneConfig) -> dict[str, Any]:
    """Fine-tune a backbone for DPR retrieval and return metrics."""
    from datasets import load_dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.evaluation import TripletEvaluator
    from sentence_transformers.training_args import BatchSamplers

    backbone_name = Path(cfg.backbone).name or cfg.backbone
    run_name = f"{backbone_name}-DPR-{cfg.learning_rate}"
    run_output = Path(cfg.output_dir) / backbone_name / run_name

    logger.info(
        "Retrieval fine-tune | backbone={} | lr={} | output={}",
        cfg.backbone, cfg.learning_rate, run_output,
    )

    model = SentenceTransformer(
        cfg.backbone,
        trust_remote_code=cfg.trust_remote_code,
    )
    model.max_seq_length = cfg.max_seq_length

    dataset = load_dataset(
        cfg.train_dataset,
        cfg.train_dataset_config,
        split=cfg.train_dataset_split,
        trust_remote_code=cfg.train_trust_remote_code,
    )
    dataset_dict = dataset.train_test_split(
        test_size=cfg.eval_split_size, seed=cfg.seed
    )
    train_dataset = dataset_dict["train"]

    if cfg.max_train_samples and len(train_dataset) > cfg.max_train_samples:
        train_dataset = train_dataset.select(range(cfg.max_train_samples))
        
    eval_dataset = dataset_dict["test"]

    logger.info(
        "Train samples: {} | Eval samples: {}",
        len(train_dataset), len(eval_dataset),
    )

    loss = _build_loss(cfg, model)

    sampler = getattr(BatchSamplers, cfg.batch_sampler, BatchSamplers.NO_DUPLICATES)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(run_output),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        batch_sampler=sampler,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        run_name=run_name,
        seed=cfg.seed,
    )

    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="train-dev",
    )

    logger.info("Baseline eval (before fine-tuning):")
    dev_evaluator(model)

    callbacks = _build_callbacks(cfg)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
        callbacks=callbacks,
    )
    trainer.train()

    logger.info("Post-training eval:")
    dev_evaluator(model)

    final_dir = run_output / "final"
    model.save_pretrained(str(final_dir))
    logger.info("Saved fine-tuned model to {}", final_dir)

    selection_score = _run_selection_eval(cfg, model)

    result = {
        "backbone": cfg.backbone,
        "learning_rate": cfg.learning_rate,
        "run_name": run_name,
        "final_model_path": str(final_dir),
        "selection_score": selection_score,
    }
    (run_output / "finetune_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _build_loss(cfg: RetrievalFinetuneConfig, model: Any) -> Any:
    from sentence_transformers.losses import (
        CachedMultipleNegativesRankingLoss,
        MultipleNegativesRankingLoss,
    )

    if cfg.loss == "CachedMultipleNegativesRankingLoss":
        return CachedMultipleNegativesRankingLoss(
            model, mini_batch_size=cfg.mini_batch_size
        )
    if cfg.loss == "MultipleNegativesRankingLoss":
        return MultipleNegativesRankingLoss(model)
    raise ValueError(f"Unsupported loss: {cfg.loss}")


def _build_callbacks(cfg: RetrievalFinetuneConfig) -> list[Any]:
    callbacks: list[Any] = []
    if cfg.early_stopping_patience is not None and cfg.early_stopping_patience > 0:
        from transformers import EarlyStoppingCallback

        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience,
                early_stopping_threshold=cfg.early_stopping_threshold,
            )
        )
    return callbacks


def _run_selection_eval(
    cfg: RetrievalFinetuneConfig, model: Any
) -> dict[str, float]:
    """Rank a carved Hindi subset by nDCG@10 for LR/model selection.

    Selection uses the same InformationRetrievalEvaluator machinery as the eval
    suite, on a capped subset of a Hindi benchmark (default: mmarco_hindi). This
    keeps selection in Hindi and consistent with what we report. Returns the
    per-dataset nDCG@k plus an ``avg_ndcg@10`` alias used as the Optuna objective.
    """
    from evals.config import RetrievalDatasetConfig
    from evals.retrieval import evaluate_retrieval_dataset
    from evals.retrieval_datasets import load_retrieval_dataset

    dataset_cfg = RetrievalDatasetConfig(
        name="selection",
        kind=cfg.selection_kind,
        dataset_name=cfg.selection_dataset_name,
        language=cfg.selection_language,
        trust_remote_code=cfg.selection_trust_remote_code,
        corpus_config=cfg.selection_corpus_config,
        corpus_split=cfg.selection_corpus_split,
        queries_config=cfg.selection_queries_config,
        queries_split=cfg.selection_queries_split,
        qrels_config=cfg.selection_qrels_config,
        qrels_split=cfg.selection_qrels_split,
        max_corpus_docs=cfg.selection_max_corpus_docs,
        max_queries=cfg.selection_max_queries,
    )

    logger.info(
        "Selection eval | dataset={} | corpus_cap={} | query_cap={}",
        cfg.selection_dataset_name,
        cfg.selection_max_corpus_docs,
        cfg.selection_max_queries,
    )
    dataset = load_retrieval_dataset(dataset_cfg)
    metrics = evaluate_retrieval_dataset(
        model,
        dataset,
        top_k=cfg.selection_top_k,
        batch_size=cfg.selection_batch_size,
        corpus_chunk_size=cfg.selection_corpus_chunk_size,
    )

    k = cfg.selection_top_k
    ndcg = float(metrics.get(f"ndcg@{k}", 0.0))
    scores: dict[str, float] = {
        f"{cfg.selection_dataset_name}.ndcg@{k}": ndcg,
        # Alias kept for the Hydra/Optuna objective and downstream readers.
        "avg_ndcg@10": ndcg,
    }
    logger.info("Selection eval | nDCG@{}: {:.4f}", k, ndcg)
    return scores


