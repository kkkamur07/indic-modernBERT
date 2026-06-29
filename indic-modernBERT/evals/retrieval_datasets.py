"""Dataset loaders for Hindi retrieval evaluation (BEIR-style and MLDR)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evals.config import RetrievalDatasetConfig


@dataclass(frozen=True)
class RetrievalDataset:
    name: str
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, float]]
    metadata: dict[str, Any]

    @property
    def relevant_docs(self) -> dict[str, set[str]]:
        """Binary relevance view for InformationRetrievalEvaluator."""
        return {
            qid: {docid for docid, score in docs.items() if score > 0}
            for qid, docs in self.qrels.items()
        }


def load_retrieval_dataset(cfg: RetrievalDatasetConfig) -> RetrievalDataset:
    if cfg.kind == "hf_beir":
        return _load_hf_beir_dataset(cfg)
    if cfg.kind == "mldr":
        return _load_mldr_dataset(cfg)
    raise ValueError(f"Unsupported retrieval dataset kind: {cfg.kind}")


def _load_hf_beir_dataset(cfg: RetrievalDatasetConfig) -> RetrievalDataset:
    from datasets import load_dataset

    corpus_config = cfg.corpus_config or "corpus"
    queries_config = cfg.queries_config or "queries"
    qrels_config = cfg.qrels_config or "default"

    corpus_rows = load_dataset(cfg.dataset_name, corpus_config, split=cfg.corpus_split, trust_remote_code=cfg.trust_remote_code)
    query_rows = load_dataset(cfg.dataset_name, queries_config, split=cfg.queries_split, trust_remote_code=cfg.trust_remote_code)
    qrel_rows = load_dataset(cfg.dataset_name, qrels_config, split=cfg.qrels_split, trust_remote_code=cfg.trust_remote_code)

    qrels = _qrels_from_beir_rows(qrel_rows)
    queries = _queries_from_rows(query_rows, cfg.max_queries)

    if cfg.max_queries is not None:
        qrels = {qid: docs for qid, docs in qrels.items() if qid in queries}

    required_doc_ids = {docid for docs in qrels.values() for docid in docs}
    corpus = _corpus_from_rows(corpus_rows, cfg.max_corpus_docs, required_doc_ids)
    queries, qrels = _filter_to_available_pairs(queries, qrels, corpus)

    return RetrievalDataset(
        name=cfg.name,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        metadata={
            "kind": cfg.kind,
            "dataset_name": cfg.dataset_name,
            "corpus_docs": len(corpus),
            "queries": len(queries),
            "qrel_queries": len(qrels),
            "max_corpus_docs": cfg.max_corpus_docs,
            "max_queries": cfg.max_queries,
        },
    )


def _load_mldr_dataset(cfg: RetrievalDatasetConfig) -> RetrievalDataset:
    from datasets import load_dataset

    language = cfg.language or "hi"
    query_rows = load_dataset(cfg.dataset_name, language, split=cfg.qrels_split, trust_remote_code=cfg.trust_remote_code)
    corpus_rows = load_dataset(cfg.dataset_name, f"corpus-{language}", split=cfg.corpus_split, trust_remote_code=cfg.trust_remote_code)

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, float]] = {}

    for row in query_rows:
        if cfg.max_queries is not None and len(queries) >= cfg.max_queries:
            break

        qid = str(row["query_id"])
        queries[qid] = str(row["query"])
        qrels[qid] = {str(passage["docid"]): 1.0 for passage in row["positive_passages"]}

    required_doc_ids = {docid for docs in qrels.values() for docid in docs}
    corpus = _corpus_from_mldr_rows(corpus_rows, cfg.max_corpus_docs, required_doc_ids)
    queries, qrels = _filter_to_available_pairs(queries, qrels, corpus)

    return RetrievalDataset(
        name=cfg.name,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        metadata={
            "kind": cfg.kind,
            "dataset_name": cfg.dataset_name,
            "language": language,
            "split": cfg.qrels_split,
            "corpus_docs": len(corpus),
            "queries": len(queries),
            "qrel_queries": len(qrels),
            "max_corpus_docs": cfg.max_corpus_docs,
            "max_queries": cfg.max_queries,
        },
    )


def _qrels_from_beir_rows(rows: Any) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    
    for row in rows:
        qid = str(row.get("query-id", row.get("query_id")))
        docid = str(row.get("corpus-id", row.get("corpus_id")))
        score = float(row.get("score", 1.0))
        qrels.setdefault(qid, {})[docid] = score
        
    return qrels


def _queries_from_rows(rows: Any, limit: int | None) -> dict[str, str]:
    queries: dict[str, str] = {}
    for row in rows:
        if limit is not None and len(queries) >= limit:
            break
        queries[str(row["_id"])] = str(row["text"])
    return queries


def _filter_to_available_pairs(
    queries: dict[str, str],
    qrels: dict[str, dict[str, float]],
    corpus: dict[str, str],
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    filtered_qrels: dict[str, dict[str, float]] = {}
    for qid, docs in qrels.items():
        available_docs = {docid: score for docid, score in docs.items() if docid in corpus}
        if available_docs:
            filtered_qrels[qid] = available_docs

    filtered_queries = {qid: query for qid, query in queries.items() if qid in filtered_qrels}
    return filtered_queries, filtered_qrels


def _corpus_from_rows(
    rows: Any, limit: int | None, required_ids: set[str] | None = None
) -> dict[str, str]:
    corpus: dict[str, str] = {}
    required_ids = required_ids or set()
    sampled_docs = 0
    for row in rows:
        docid = str(row["_id"])
        include_sample = limit is None or sampled_docs < limit
        if not include_sample and docid not in required_ids:
            continue
        text = str(row["text"])
        title = str(row.get("title") or "")
        corpus[docid] = f"{title}\n{text}" if title else text
        if include_sample:
            sampled_docs += 1
    return corpus


def _corpus_from_mldr_rows(
    rows: Any, limit: int | None, required_ids: set[str] | None = None
) -> dict[str, str]:
    corpus: dict[str, str] = {}
    required_ids = required_ids or set()
    sampled_docs = 0
    for row in rows:
        docid = str(row["docid"])
        include_sample = limit is None or sampled_docs < limit
        if not include_sample and docid not in required_ids:
            continue
        corpus[docid] = str(row["text"])
        if include_sample:
            sampled_docs += 1
    return corpus
