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
        run-evals-retrieval run-evals-smoke export-hf pipeline-trace \
        retrieval-finetune retrieval-finetune-nohup \
        retrieval-optuna retrieval-optuna-nohup \
        retrieval-prepare-optuna-subset \
        retrieval-optuna-all retrieval-optuna-all-nohup

# --- Tokenizer ---

train-bpe:
	uv run python -m tokenizer.trainer.bpe_trainer

train-bpe-nohup:
	mkdir -p logs
	PYTHONUNBUFFERED=1 nohup $(MAKE) train-bpe > logs/train_bpe.log 2>&1 &

eval-bpe:
	uv run python scripts/compare_bpe_vocabs.py

eval-bpe-nohup:
	mkdir -p logs
	PYTHONUNBUFFERED=1 nohup $(MAKE) eval-bpe > logs/eval_bpe.log 2>&1 &

pretokenization:
	uv run python -m tokenizer.pretokenization

# --- Data ---
# Convert from txt to parquet. 
convert-indiccorp:
	uv run python -m dataset.indiccorp_dataset

# --- Pretrain ---

train-pretrain:
	PYTHONPATH=indic-modernBERT uv run --extra pretrain python scripts/run_pretrain.py

# Phase-1 production pretrain (~23.6B tok, configs/pretrain/hindi_mlm_phase1.yaml).
train-phase1:
	mkdir -p .tmp logs/phase1
	$(TMPDIR_ENV) TRAIN_STEP_LOG=0 PYTHONPATH=indic-modernBERT PYTHONUNBUFFERED=1 \
	  uv run --extra pretrain python scripts/run_pretrain.py --config-name hindi_mlm_phase1

train-phase1-nohup:
	mkdir -p logs/phase1 .tmp
	PYTHONUNBUFFERED=1 nohup $(MAKE) train-phase1 > logs/phase1/nohup.log 2>&1 &

# Phase-2 production context extension (~4.85B tok @ 8192, configs/pretrain/hindi_mlm_context_extension.yaml).
train-phase2:
	mkdir -p .tmp logs/phase2
	$(TMPDIR_ENV) TRAIN_STEP_LOG=0 PYTHONPATH=indic-modernBERT PYTHONUNBUFFERED=1 \
	  uv run --extra pretrain python scripts/run_pretrain.py --config-name hindi_mlm_context_extension $(ARGS)

train-phase2-nohup:
	mkdir -p logs/phase2 .tmp
	PYTHONUNBUFFERED=1 nohup $(MAKE) train-phase2 > logs/phase2/nohup.log 2>&1 &

# Phase-2 (context extension @ 8192) VRAM smoke: short run to measure GPU memory.
# Override the microbatch to probe VRAM, e.g.:
#   make train-smoke-phase2 ARGS="pretrain.device_train_microbatch_size=4"
# Watch in another shell: watch -n1 nvidia-smi
train-smoke-phase2:
	mkdir -p .tmp logs/phase2_smoke
	$(TMPDIR_ENV) TRAIN_STEP_LOG=0 PYTHONPATH=indic-modernBERT PYTHONUNBUFFERED=1 \
	  uv run --extra pretrain python scripts/run_pretrain.py --config-name hindi_mlm_context_extension_smoke $(ARGS)

tensorboard-phase1:
	tensorboard --logdir artifacts/model/modernbert/tensorboard/phase1

# Optuna LR sweep — same stack as hindi_mlm_phase1 (modernbert_base, micro=8, 500M warmup).
lr-sweep:
	mkdir -p .tmp logs/lr_sweep
	$(TMPDIR_ENV) TRAIN_STEP_LOG=0 PYTHONPATH=indic-modernBERT PYTHONUNBUFFERED=1 \
	  uv run --extra pretrain --extra sweep python scripts/run_pretrain.py \
	  --config-path ../configs/sweep --config-name hindi_mlm_lr_sweep -m

lr-sweep-nohup:
	mkdir -p logs/lr_sweep .tmp
	PYTHONUNBUFFERED=1 nohup $(MAKE) lr-sweep > logs/lr_sweep/nohup.log 2>&1 &

# --- Evaluation ---

run-evals:
	PYTHONPATH=indic-modernBERT uv run --extra evals python scripts/run_evals.py $(ARGS)

run-evals-transfer:
	PYTHONPATH=indic-modernBERT uv run --extra evals python scripts/run_evals.py --config-name hindi_transfer $(ARGS)

# Phase-2 selected checkpoint (full-corpus ba1157), downstream-only
# (NER + MASSIVE intent). See configs/evals/hindi_phase2.yaml.
run-evals-phase2:
	PYTHONPATH=indic-modernBERT uv run --extra evals python scripts/run_evals.py --config-name hindi_phase2 $(ARGS)

run-evals-phase2-nohup:
	mkdir -p logs/evals_phase2
	PYTHONUNBUFFERED=1 nohup $(MAKE) run-evals-phase2 > logs/evals_phase2/nohup.log 2>&1 &

run-evals-retrieval:
	PYTHONPATH=indic-modernBERT uv run --extra evals python scripts/run_evals.py --config-name hindi_retrieval $(ARGS)

# --- Retrieval Fine-tuning (upstream DPR recipe) ---

# Single-LR retrieval fine-tune. Override LR or backbone:
#   make retrieval-finetune ARGS="retrieval_ft.learning_rate=8e-5"
#   make retrieval-finetune ARGS="retrieval_ft.backbone=artifacts/model/modernbert/hf_export/phase2_latest_ba1157"
retrieval-finetune:
	PYTHONPATH=indic-modernBERT uv run --extra evals python scripts/run_retrieval_finetune.py $(ARGS)

retrieval-finetune-nohup:
	mkdir -p logs/retrieval_finetune
	PYTHONUNBUFFERED=1 nohup $(MAKE) retrieval-finetune ARGS="$(ARGS)" > logs/retrieval_finetune/nohup.log 2>&1 &

# For optuna to not mix the different backbones. 
RETRIEVAL_SWEEP_BACKBONES ?= \
	artifacts/model/modernbert/hf_export/phase2_latest_ba1157 \
	ai4bharat/IndicBERTv2-MLM-only \
	jhu-clsp/mmBERT-small
RETRIEVAL_OPTUNA_SUBSET ?= artifacts/retrieval_finetune/subsets/mmarco_hindi_train100k_eval1k_seed17.jsonl

# Optuna LR exploration: 10 log-scale trials over 1e-6..1e-2 with trainer early
# stopping inside each trial, maximizing Hindi mmarco_hindi selection nDCG@10.
retrieval-optuna:
	PYTHONPATH=indic-modernBERT uv run --extra evals --extra sweep python scripts/run_retrieval_finetune.py \
	  --config-name hindi_dpr_optuna -m $(ARGS)

retrieval-optuna-nohup:
	mkdir -p logs/retrieval_optuna
	PYTHONUNBUFFERED=1 nohup $(MAKE) retrieval-optuna ARGS="$(ARGS)" > logs/retrieval_optuna/nohup.log 2>&1 &

retrieval-prepare-optuna-subset:
	PYTHONPATH=indic-modernBERT uv run --extra evals python scripts/prepare_retrieval_subset.py \
	  --train-samples 100000 --eval-samples 1000 --candidate-triples 1000000 \
	  --output $(RETRIEVAL_OPTUNA_SUBSET)

retrieval-optuna-all: retrieval-prepare-optuna-subset
	for backbone in $(RETRIEVAL_SWEEP_BACKBONES); do \
	  echo "=== Retrieval Optuna sweep: $$backbone ==="; \
	  $(MAKE) retrieval-optuna ARGS="retrieval_ft.backbone=$$backbone $(ARGS)"; \
	done

retrieval-optuna-all-nohup:
	mkdir -p logs/retrieval_optuna_all
	PYTHONUNBUFFERED=1 nohup $(MAKE) retrieval-optuna-all ARGS="$(ARGS)" > logs/retrieval_optuna_all/nohup.log 2>&1 &

# --- Utilities ---

export-hf:
	uv run python scripts/export_hf.py $(ARGS)

pipeline-trace:
	PYTHONPATH=indic-modernBERT uv run python scripts/pipeline_trace.py $(ARGS)
