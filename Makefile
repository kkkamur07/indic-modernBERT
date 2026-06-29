# Run from repo root. Source: indic-modernBERT/ (flat imports, package = false).
SHELL := /bin/bash
export PYTHONPATH := indic-modernBERT
TMPDIR_ENV := TMPDIR=$(PWD)/.tmp TORCHINDUCTOR_CACHE_DIR=$(PWD)/.tmp/torchinductor

.PHONY: train-bpe train-bpe-nohup eval-bpe eval-bpe-nohup pretokenization \
        convert-indiccorp \
        train-pretrain train-phase1 train-phase1-nohup \
        train-phase2 train-phase2-nohup train-smoke-phase2 \
        train-smoke-50ba train-smoke-50ba-nohup lr-sweep lr-sweep-nohup \
        run-evals run-evals-transfer run-evals-phase2 run-evals-phase2-nohup \
        run-evals-retrieval run-evals-smoke eval-comparison-reports export-hf upload-hf-mlm upload-hf-retriever pipeline-trace \
        retrieval-finetune retrieval-finetune-nohup \
        retrieval-finetune-eval-local-all retrieval-finetune-eval-local-all-nohup \
        retrieval-optuna retrieval-optuna-nohup \
        retrieval-prepare-optuna-subset \
        retrieval-prepare-full-subset retrieval-prepare-full-subset-nohup \
        retrieval-optuna-all retrieval-optuna-all-nohup

# --- Tokenizer ---

train-bpe:
	uv run python -m tokenizer.trainer.bpe_trainer

train-bpe-nohup:
	mkdir -p logs/hi/tokenizer
	PYTHONUNBUFFERED=1 nohup $(MAKE) train-bpe > logs/hi/tokenizer/train_bpe.log 2>&1 &

eval-bpe:
	uv run python scripts/compare_bpe_vocabs.py

eval-bpe-nohup:
	mkdir -p logs/hi/tokenizer
	PYTHONUNBUFFERED=1 nohup $(MAKE) eval-bpe > logs/hi/tokenizer/eval_bpe.log 2>&1 &

pretokenization:
	uv run python -m tokenizer.pretokenization

# --- Data ---
# Convert from txt to parquet. 
convert-indiccorp:
	uv run python -m dataset.indiccorp_dataset

# --- Pretrain ---

train-pretrain:
	PYTHONPATH=indic-modernBERT uv run --extra pretrain python scripts/run_pretrain.py

# Phase-1 production pretrain (~23.6B tok, configs/hi/pretrain/hindi_mlm_phase1.yaml).
train-phase1:
	mkdir -p .tmp logs/hi/pretrain/phase1
	$(TMPDIR_ENV) TRAIN_STEP_LOG=0 PYTHONPATH=indic-modernBERT PYTHONUNBUFFERED=1 \
	  uv run --extra pretrain python scripts/run_pretrain.py --config-name hindi_mlm_phase1

train-phase1-nohup:
	mkdir -p logs/hi/pretrain/phase1 .tmp
	PYTHONUNBUFFERED=1 nohup $(MAKE) train-phase1 > logs/hi/pretrain/phase1/nohup.log 2>&1 &

# Phase-2 production context extension (~4.85B tok @ 8192, configs/hi/pretrain/hindi_mlm_context_extension.yaml).
train-phase2:
	mkdir -p .tmp logs/hi/pretrain/phase2
	$(TMPDIR_ENV) TRAIN_STEP_LOG=0 PYTHONPATH=indic-modernBERT PYTHONUNBUFFERED=1 \
	  uv run --extra pretrain python scripts/run_pretrain.py --config-name hindi_mlm_context_extension $(ARGS)

train-phase2-nohup:
	mkdir -p logs/hi/pretrain/phase2 .tmp
	PYTHONUNBUFFERED=1 nohup $(MAKE) train-phase2 > logs/hi/pretrain/phase2/nohup.log 2>&1 &

# Phase-2 (context extension @ 8192) VRAM smoke: short run to measure GPU memory.
# Override the microbatch to probe VRAM, e.g.:
#   make train-smoke-phase2 ARGS="pretrain.device_train_microbatch_size=4"
# Watch in another shell: watch -n1 nvidia-smi
train-smoke-phase2:
	mkdir -p .tmp logs/hi/pretrain/phase2_smoke
	$(TMPDIR_ENV) TRAIN_STEP_LOG=0 PYTHONPATH=indic-modernBERT PYTHONUNBUFFERED=1 \
	  uv run --extra pretrain python scripts/run_pretrain.py --config-name hindi_mlm_context_extension_smoke $(ARGS)

tensorboard-phase1:
	tensorboard --logdir artifacts/model/modernbert/hi/tensorboard/phase1

# Optuna LR sweep — same stack as hindi_mlm_phase1 (modernbert_base, micro=8, 500M warmup).
lr-sweep:
	mkdir -p .tmp logs/hi/pretrain/lr_sweep
	$(TMPDIR_ENV) TRAIN_STEP_LOG=0 PYTHONPATH=indic-modernBERT PYTHONUNBUFFERED=1 \
	  uv run --extra pretrain --extra sweep python scripts/run_pretrain.py \
	  --config-path ../configs/hi/sweep --config-name hindi_mlm_lr_sweep -m

lr-sweep-nohup:
	mkdir -p logs/hi/pretrain/lr_sweep .tmp
	PYTHONUNBUFFERED=1 nohup $(MAKE) lr-sweep > logs/hi/pretrain/lr_sweep/nohup.log 2>&1 &

# --- Evaluation ---

run-evals:
	PYTHONPATH=indic-modernBERT uv run --extra evals python scripts/run_evals.py $(ARGS)

run-evals-transfer:
	PYTHONPATH=indic-modernBERT uv run --extra evals python scripts/run_evals.py --config-name hindi_transfer $(ARGS)

# Phase-2 selected checkpoint (full-corpus ba1157), downstream-only
# (NER + MASSIVE intent). See configs/hi/evals/hindi_phase2.yaml.
run-evals-phase2:
	PYTHONPATH=indic-modernBERT uv run --extra evals python scripts/run_evals.py --config-name hindi_phase2 $(ARGS)

run-evals-phase2-nohup:
	mkdir -p logs/hi/evals/phase2
	PYTHONUNBUFFERED=1 nohup $(MAKE) run-evals-phase2 > logs/hi/evals/phase2/nohup.log 2>&1 &

run-evals-retrieval:
	PYTHONPATH=indic-modernBERT uv run --extra evals python scripts/run_evals.py --config-name hindi_retrieval $(ARGS)

eval-comparison-reports:
	python3 scripts/generate_eval_comparison_reports.py $(if $(LANG),--lang $(LANG),)

# --- Retrieval Fine-tuning (upstream DPR recipe) ---

# Single-LR retrieval fine-tune. Override LR or backbone:
#   make retrieval-finetune ARGS="retrieval_ft.learning_rate=8e-5"
#   make retrieval-finetune ARGS="retrieval_ft.backbone=artifacts/model/modernbert/hi/hf_export/phase2_latest_ba1157"
retrieval-finetune:
	PYTHONPATH=indic-modernBERT uv run --extra evals python scripts/run_retrieval_finetune.py $(ARGS)

retrieval-finetune-nohup:
	mkdir -p logs/hi/retrieval/finetune
	PYTHONUNBUFFERED=1 nohup $(MAKE) retrieval-finetune ARGS="$(ARGS)" > logs/hi/retrieval/finetune/nohup.log 2>&1 &

# For optuna to not mix different backbones. Each entry is backbone=max_seq_length.
RETRIEVAL_SWEEP_BACKBONES ?= \
	artifacts/model/modernbert/hi/hf_export/phase2_latest_ba1157=8192 \
	ai4bharat/IndicBERTv2-MLM-only=512 \
	jhu-clsp/mmBERT-small=8192
RETRIEVAL_OPTUNA_SUBSET ?= artifacts/retrieval_finetune/hi/subsets/mmarco_hindi_train100k_eval1k_seed17.jsonl
RETRIEVAL_FULL_SUBSET ?= artifacts/retrieval_finetune/hi/subsets/mmarco_hindi_train1250k_eval1k_seed17.jsonl
RETRIEVAL_FULL_CANDIDATE_TRIPLES ?= 40000000
RETRIEVAL_MMARCO_RAW_DIR ?= artifacts/retrieval_finetune/hi/raw/unicamp-dl_mmarco
RETRIEVAL_BEST_LR ?= 0.00010972521281842244
RETRIEVAL_LOCAL_RUN_OUTPUT ?= artifacts/retrieval_finetune/hi/full_local_jsonl_train_eval_runs

# Optuna LR exploration: 10 log-scale trials over 1e-6..1e-2 with trainer early
# stopping inside each trial, maximizing Hindi mmarco_hindi selection nDCG@10.
retrieval-optuna:
	PYTHONPATH=indic-modernBERT uv run --extra evals --extra sweep python scripts/run_retrieval_finetune.py \
	  --config-name hindi_dpr_optuna -m $(ARGS)

retrieval-optuna-nohup:
	mkdir -p logs/hi/retrieval/optuna
	PYTHONUNBUFFERED=1 nohup $(MAKE) retrieval-optuna ARGS="$(ARGS)" > logs/hi/retrieval/optuna/nohup.log 2>&1 &

retrieval-prepare-optuna-subset:
	PYTHONPATH=indic-modernBERT uv run --extra evals python scripts/prepare_retrieval_subset.py \
	  --train-samples 100000 --eval-samples 1000 --candidate-triples 1000000 \
	  --download-dir $(RETRIEVAL_MMARCO_RAW_DIR) \
	  --output $(RETRIEVAL_OPTUNA_SUBSET) $(ARGS)

retrieval-prepare-full-subset:
	PYTHONPATH=indic-modernBERT uv run --extra evals python scripts/prepare_retrieval_subset.py \
	  --train-samples 1250000 --eval-samples 1000 \
	  --candidate-triples $(RETRIEVAL_FULL_CANDIDATE_TRIPLES) \
	  --download-dir $(RETRIEVAL_MMARCO_RAW_DIR) \
	  --output $(RETRIEVAL_FULL_SUBSET) $(ARGS)

retrieval-prepare-full-subset-nohup:
	mkdir -p logs/hi/retrieval/prepare_full
	PYTHONUNBUFFERED=1 nohup make retrieval-prepare-full-subset ARGS="$(ARGS)" > logs/hi/retrieval/prepare_full/nohup.log 2>&1 &

retrieval-finetune-eval-local-all:
	PYTHONUNBUFFERED=1 RETRIEVAL_BEST_LR=$(RETRIEVAL_BEST_LR) \
	  RETRIEVAL_FULL_SUBSET=$(RETRIEVAL_FULL_SUBSET) \
	  RETRIEVAL_LOCAL_RUN_OUTPUT=$(RETRIEVAL_LOCAL_RUN_OUTPUT) \
	  bash scripts/run_retrieval_finetune_eval_sequence.sh

retrieval-finetune-eval-local-all-nohup:
	mkdir -p logs/hi/retrieval/finetune; \
	PYTHONUNBUFFERED=1 nohup $(MAKE) retrieval-finetune-eval-local-all \
	  > logs/hi/retrieval/finetune/local_train_eval_all_models.nohup.log 2>&1 & \
	  echo $$! > logs/hi/retrieval/finetune/local_train_eval_all_models.pid

retrieval-optuna-all: retrieval-prepare-optuna-subset
	for spec in $(RETRIEVAL_SWEEP_BACKBONES); do \
	  backbone=$${spec%=*}; \
	  max_seq_length=$${spec#*=}; \
	  echo "=== Retrieval Optuna sweep: $$backbone ==="; \
	  $(MAKE) retrieval-optuna ARGS="retrieval_ft.backbone=$$backbone retrieval_ft.max_seq_length=$$max_seq_length $(ARGS)"; \
	done

retrieval-optuna-all-nohup:
	mkdir -p logs/hi/retrieval/optuna_all
	PYTHONUNBUFFERED=1 nohup $(MAKE) retrieval-optuna-all ARGS="$(ARGS)" > logs/hi/retrieval/optuna_all/nohup.log 2>&1 &

# --- Utilities ---

export-hf:
	uv run python scripts/export_hf.py $(ARGS)

HF_REPO_ID_MLM ?= YOUR_ORG/hindi-modernbert
HF_REPO_ID_RETRIEVER ?= YOUR_ORG/hindi-modernbert-retriever

upload-hf-mlm:
	uv run python scripts/upload_hf_model.py $(HF_REPO_ID_MLM) --variant mlm $(ARGS)

upload-hf-retriever:
	uv run python scripts/upload_hf_model.py $(HF_REPO_ID_RETRIEVER) --variant retriever $(ARGS)

pipeline-trace:
	PYTHONPATH=indic-modernBERT uv run python scripts/pipeline_trace.py $(ARGS)
