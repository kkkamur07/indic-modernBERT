# Run from repo root. Source: indic-modernBERT/ (flat imports, package = false).
SHELL := /bin/bash
export PYTHONPATH := indic-modernBERT
TMPDIR_ENV := TMPDIR=$(PWD)/.tmp TORCHINDUCTOR_CACHE_DIR=$(PWD)/.tmp/torchinductor

.PHONY: train-bpe train-bpe-nohup eval-bpe eval-bpe-nohup pretokenization \
        train-pretrain train-phase1 train-phase1-nohup \
        train-smoke-50ba train-smoke-50ba-nohup lr-sweep lr-sweep-nohup \
        run-evals run-evals-smoke export-hf pipeline-trace

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

tensorboard-phase1:
	tensorboard --logdir artifacts/model/modernbert/tensorboard/phase1

# Test Pretrain runs
train-smoke-50ba:
	mkdir -p .tmp logs/smoke_50ba
	rm -rf artifacts/model/modernbert/checkpoints/smoke_50ba artifacts/model/modernbert/tensorboard/smoke_50ba
	script -q -e -f logs/smoke_50ba/train.log -c "$(TMPDIR_ENV) TRAIN_PROGRESS_BAR=1 TRAIN_STEP_LOG=0 PYTHONPATH=indic-modernBERT PYTHONUNBUFFERED=1 uv run --extra pretrain python scripts/run_pretrain.py --config-name hindi_mlm_smoke_50ba"

train-smoke-50ba-nohup:
	mkdir -p logs/smoke_50ba .tmp
	PYTHONUNBUFFERED=1 nohup $(MAKE) train-smoke-50ba > logs/smoke_50ba/nohup.log 2>&1 &

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

run-evals-smoke:
	PYTHONPATH=indic-modernBERT uv run --extra evals python scripts/run_evals.py \
	  eval.supervised.max_train_samples=8 eval.supervised.max_eval_samples=8 \
	  eval.supervised.num_train_epochs=0.01 eval.mlm.max_samples=8 eval.mlm.max_batches=1 \
	  eval.efficiency.sequence_lengths='[128]' eval.efficiency.warmup_steps=0 \
	  eval.efficiency.measured_steps=1 $(ARGS)

# --- Utilities ---

export-hf:
	uv run python scripts/export_hf.py $(ARGS)

pipeline-trace:
	PYTHONPATH=indic-modernBERT uv run python scripts/pipeline_trace.py $(ARGS)
