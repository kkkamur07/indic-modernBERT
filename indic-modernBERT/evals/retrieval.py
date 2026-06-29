"""Dense retrieval evaluation for Hindi BEIR-style and MLDR datasets.

Uses sentence-transformers' InformationRetrievalEvaluator for encoding, chunked
cosine scoring, and metric computation (nDCG, recall, MRR, MAP, precision).
This matches upstream ModernBERT's use of the sentence-transformers ecosystem
for dense retrieval (see _support_repo/ModernBERT/examples/evaluate_st.py and
train_st.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from evals.config import EvalSuiteConfig
from evals.retrieval_datasets import RetrievalDataset, load_retrieval_dataset
from evals.runtime import active_context_length, choose_device, set_eval_seed


def run_retrieval_eval(cfg: EvalSuiteConfig, output_dir: Path) -> dict[str, Any]:
    """Run dense retrieval ranking via InformationRetrievalEvaluator.

    Expects a retrieval-capable (SentenceTransformers-compatible) checkpoint.
    Raw MLM checkpoints can be loaded, but their scores mostly measure pooling
    quality -- not a fair ModernBERT-style retrieval recipe.
    """
    from sentence_transformers import SentenceTransformer

    retrieval_cfg = cfg.retrieval
    set_eval_seed(cfg.seed)
    device = choose_device(cfg.device)

    trust_remote_code = (
        cfg.model.trust_remote_code
        if retrieval_cfg.trust_remote_code is None
        else retrieval_cfg.trust_remote_code
    )
    model = SentenceTransformer(
        cfg.model.model_name_or_path,
        trust_remote_code=trust_remote_code,
        device=str(device),
    )
    model.max_seq_length = _resolve_max_seq_length(cfg)

    dataset_results = []
    flat_metrics: dict[str, float | int] = {}
    for dataset_cfg in retrieval_cfg.datasets:
        if not dataset_cfg.enabled:
            continue
        dataset = load_retrieval_dataset(dataset_cfg)
        metrics = _evaluate_dataset(cfg, model, dataset)
        dataset_results.append(
            {
                "name": dataset.name,
                "metrics": metrics,
                "metadata": dataset.metadata,
            }
        )
        for metric, value in metrics.items():
            flat_metrics[f"{dataset.name}.{metric}"] = value

    result = {
        "name": "retrieval",
        "type": "retrieval",
        "status": "completed",
        "metrics": flat_metrics,
        "datasets": dataset_results,
        "config": {
            "max_seq_length": model.max_seq_length,
            "configured_max_seq_length": retrieval_cfg.max_seq_length,
            "batch_size": retrieval_cfg.batch_size,
            "corpus_chunk_size": retrieval_cfg.corpus_chunk_size,
            "top_k": retrieval_cfg.top_k,
        },
    }
    (output_dir / "retrieval_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _resolve_max_seq_length(cfg: EvalSuiteConfig) -> int:
    model_limit = active_context_length(cfg.model)

    if cfg.retrieval.max_seq_length is None:
        return model_limit

    return min(cfg.retrieval.max_seq_length, model_limit)


def _evaluate_dataset(
    cfg: EvalSuiteConfig, model: Any, dataset: RetrievalDataset
) -> dict[str, float | int]:
    retrieval_cfg = cfg.retrieval
    return evaluate_retrieval_dataset(
        model,
        dataset,
        top_k=retrieval_cfg.top_k,
        batch_size=retrieval_cfg.batch_size,
        corpus_chunk_size=retrieval_cfg.corpus_chunk_size,
    )


def evaluate_retrieval_dataset(
    model: Any,
    dataset: RetrievalDataset,
    *,
    top_k: int,
    batch_size: int,
    corpus_chunk_size: int,
) -> dict[str, float | int]:
    """Rank a single retrieval dataset and return @top_k metrics.

    Shared by the eval suite and the fine-tune selection step so both compute
    nDCG@k identically with InformationRetrievalEvaluator.
    """
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    k = top_k

    query_ids_with_qrels = {qid for qid in dataset.queries if qid in dataset.qrels}

    if not query_ids_with_qrels or not dataset.corpus:
        raise ValueError(f"Dataset {dataset.name} has no query/doc pairs to evaluate")

    evaluator = InformationRetrievalEvaluator(
        queries=dataset.queries,
        corpus=dataset.corpus,
        relevant_docs=dataset.relevant_docs,
        corpus_chunk_size=corpus_chunk_size,
        ndcg_at_k=[k],
        mrr_at_k=[k],
        accuracy_at_k=[k],
        precision_recall_at_k=[k],
        map_at_k=[k],
        batch_size=batch_size,
        show_progress_bar=True,
        name=dataset.name,
        write_csv=False,
    )

    # Call the evaluator's __call__ (not compute_metrices) so it lazily wires up
    # the model's score function. It returns a flat dict keyed like
    # "{name}_{score_fn}_{metric}@{k}" (e.g. "mldr_hi_cosine_ndcg@10").
    all_scores = evaluator(model)

    def _metric(metric: str) -> float:
        return _select_metric(all_scores, dataset.name, metric, k)

    logger.info(
        "Retrieval {} | nDCG@{}: {:.4f} | recall@{}: {:.4f} | MRR@{}: {:.4f}",
        dataset.name,
        k,
        _metric("ndcg"),
        k,
        _metric("recall"),
        k,
        _metric("mrr"),
    )

    metrics: dict[str, float | int] = {
        f"ndcg@{k}": _metric("ndcg"),
        f"recall@{k}": _metric("recall"),
        f"mrr@{k}": _metric("mrr"),
        f"map@{k}": _metric("map"),
        f"precision@{k}": _metric("precision"),
        f"accuracy@{k}": _metric("accuracy"),
        "queries_evaluated": len(query_ids_with_qrels),
        "corpus_docs": len(dataset.corpus),
    }
    return metrics


def _select_metric(
    scores: dict[str, float], dataset_name: str, metric: str, k: int
) -> float:
    """Pull one metric out of InformationRetrievalEvaluator's flat output.

    Keys look like "{dataset_name}_{score_fn}_{metric}@{k}". We prefer the
    cosine score function (SentenceTransformer's default), falling back to
    whatever score function is present.
    """
    suffix = f"{metric}@{k}"
    candidates = {key: value for key, value in scores.items() if key.endswith(suffix)}
    if not candidates:
        return 0.0
    for key, value in candidates.items():
        if "cosine" in key:
            return float(value)
    return float(next(iter(candidates.values())))
