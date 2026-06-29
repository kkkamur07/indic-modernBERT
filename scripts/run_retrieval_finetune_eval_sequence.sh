#!/usr/bin/env bash
set -euo pipefail

LR="${RETRIEVAL_BEST_LR:-0.00010972521281842244}"
SUBSET="${RETRIEVAL_FULL_SUBSET:-artifacts/retrieval_finetune/hi/subsets/mmarco_hindi_train1250k_eval1k_seed17.jsonl}"
OUTPUT_DIR="${RETRIEVAL_LOCAL_RUN_OUTPUT:-artifacts/retrieval_finetune/hi/full_local_jsonl_train_eval_runs}"
LOG_DIR="${RETRIEVAL_LOG_DIR:-logs/hi/retrieval/finetune}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

if [[ ! -s "${SUBSET}" ]]; then
  echo "Missing local retrieval subset: ${SUBSET}" >&2
  exit 1
fi

COMMON_ARGS="retrieval_ft.train_local_path=${SUBSET} retrieval_ft.max_train_samples=1250000 retrieval_ft.eval_split_size=1000 retrieval_ft.learning_rate=${LR} retrieval_ft.output_dir=${OUTPUT_DIR}"

run_finetune() {
  local slug="$1"
  local backbone="$2"
  local max_seq_length="$3"
  local log_path="${LOG_DIR}/local_train_eval_${slug}_finetune_lr${LR}.log"

  echo "[$(date -Is)] Fine-tune ${slug}"
  PYTHONUNBUFFERED=1 make retrieval-finetune \
    ARGS="retrieval_ft.backbone=${backbone} retrieval_ft.max_seq_length=${max_seq_length} ${COMMON_ARGS}" \
    > "${log_path}" 2>&1
}

run_eval() {
  local slug="$1"
  local model_path="$2"
  local max_seq_length="$3"
  local log_path="${LOG_DIR}/local_train_eval_${slug}_eval_lr${LR}.log"
  local model_override="'eval.models=[{model_name_or_path:${model_path},tokenizer_name_or_path:null,trust_remote_code:false,max_sequence_length:${max_seq_length},context_mode:model_max}]'"

  echo "[$(date -Is)] Full retrieval eval ${slug}"
  PYTHONUNBUFFERED=1 make run-evals-retrieval ARGS="${model_override}" > "${log_path}" 2>&1
}

run_finetune "indicbertv2" "ai4bharat/IndicBERTv2-MLM-only" "512"
run_eval "indicbertv2" "${OUTPUT_DIR}/IndicBERTv2-MLM-only/IndicBERTv2-MLM-only-DPR-${LR}/final" "512"

run_finetune "phase2_latest_ba1157" "artifacts/model/modernbert/hi/hf_export/phase2_latest_ba1157" "8192"
run_eval "phase2_latest_ba1157" "${OUTPUT_DIR}/phase2_latest_ba1157/phase2_latest_ba1157-DPR-${LR}/final" "8192"

run_finetune "mmbert_small" "jhu-clsp/mmBERT-small" "8192"
run_eval "mmbert_small" "${OUTPUT_DIR}/mmBERT-small/mmBERT-small-DPR-${LR}/final" "8192"

echo "[$(date -Is)] Retrieval fine-tune/eval sequence complete"
