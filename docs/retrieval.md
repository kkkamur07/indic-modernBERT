# Retrieval Measurement Notes

This repo measures information retrieval (IR) with dense embeddings from a
`SentenceTransformer` checkpoint. The evaluator encodes every query and every
document, ranks documents by cosine similarity to each query, and compares the
top ranked documents against qrels (query relevance labels).

## Evaluation Flow

1. Load a BEIR-style dataset as:
   - `corpus`: document id to document text.
   - `queries`: query id to query text.
   - `qrels`: query id to relevant document ids and relevance scores.
2. Encode queries and corpus documents with the same retrieval model.
3. Score query-document pairs by cosine similarity.
4. Sort each query's corpus ranking by descending score.
5. Compute metrics at `k`, currently `k = eval.retrieval.top_k` and defaults to
   `10`.

The implementation is in `indic-modernBERT/evals/retrieval.py` and uses
`sentence_transformers.evaluation.InformationRetrievalEvaluator`.

## Metrics

- `nDCG@10`: Normalized Discounted Cumulative Gain at 10. This rewards relevant
  documents near the top of the ranking and discounts relevant documents that
  appear lower. This is the main selection/reporting metric used by the
  ModernBERT retrieval recipe.
- `recall@10`: Fraction of all relevant documents for a query that appear in
  the top 10. This answers: "Did we retrieve the relevant documents at all?"
- `MRR@10`: Mean Reciprocal Rank at 10. For each query, it is `1 / rank` of the
  first relevant document in the top 10, or `0` if none appears.
- `MAP@10`: Mean Average Precision at 10. This averages precision at each rank
  where a relevant document appears, then averages over queries.
- `precision@10`: Fraction of the top 10 documents that are relevant.
- `accuracy@10`: Whether at least one relevant document appears in the top 10.

## Training Selection

Retrieval fine-tuning follows the upstream ModernBERT DPR-style recipe, but
localized to Hindi:

1. Train with Hindi-translated MS-MARCO triples (`unicamp-dl/mmarco`, config
   `hindi`) using `CachedMultipleNegativesRankingLoss`. Training and the
   in-training dev evaluator are therefore Hindi, not English. This matches the
   upstream ModernBERT retrieval setup in sequence-length character: MS-MARCO
   passage triples are short-passage DPR data, not 8192-token long documents.
2. Explore learning rates with Hydra Optuna over `1e-6..1e-2` on a log scale.
3. Rank every trial by `nDCG@10` on a carved subset of a Hindi benchmark
   (default: `mmarco_hindi` capped at 50k corpus docs and 500 queries).
   Selection is therefore Hindi and consistent with what we report, not an
   English BEIR proxy. LR sweeps run on a subset by design; raise
   `selection_max_corpus_docs`/`selection_max_queries` for a stronger signal.
4. Use the trial with the highest `nDCG@10` as the selected retriever.

The 8192-token retrieval question is measured in the final eval suite via
`mldr_hi`, whose Hindi documents average thousands of words. That long-context
ability should come from the model's pretraining/context extension; the
retrieval fine-tune intentionally stays aligned with upstream's short-passage
DPR recipe.

Run the Optuna study with:

```bash
make retrieval-optuna ARGS="retrieval_ft.backbone=<hf-export-path>"
```

Prepare the fixed full-budget local training split separately:

```bash
make retrieval-prepare-full-subset-nohup ARGS="--overwrite"
```

This writes `artifacts/retrieval_finetune/subsets/mmarco_hindi_train1250k_eval1k_seed17.jsonl`
with 1.25M train rows plus the 1k held-out triplet dev split used during
training. The downloader streams mMARCO TSV bytes and decodes UTF-8 explicitly
to avoid mojibake in Hindi text.

Then run the Hindi retrieval evaluation with:

```bash
make run-evals-retrieval ARGS="eval.models.0.model_name_or_path=<best-trial-final_model_path>"
```

## Interpreting Results

Use `nDCG@10` as the primary quality number because it captures ranking quality,
not just whether a positive document exists somewhere in the top 10. Use
`recall@10` to diagnose missing relevant documents, and `MRR@10` to check how
often the first relevant result appears very high in the ranking.

Raw MLM checkpoints can be evaluated with this machinery, but those scores are
not directly comparable to ModernBERT retrieval numbers unless the checkpoint
has also gone through the same retrieval fine-tuning and LR selection recipe.
