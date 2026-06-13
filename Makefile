# Run from repo root. Source: indic-modernBERT/ (flat imports, package = false).
export PYTHONPATH := indic-modernBERT

.PHONY: train-bpe train-bpe-nohup eval-bpe eval-bpe-nohup train-pretrain validate-modernbert export-hf pretokenization

train-bpe:
	uv run python -m tokenizer.trainer.bpe_trainer

train-bpe-nohup:
	mkdir -p logs
	PYTHONUNBUFFERED=1 nohup $(MAKE) train-bpe > logs/train_bpe.log 2>&1 &

# Intrinsic metrics on eval holdout for every trained BPE vocab + baselines.
eval-bpe:
	uv run python scripts/compare_bpe_vocabs.py

eval-bpe-nohup:
	mkdir -p logs
	PYTHONUNBUFFERED=1 nohup $(MAKE) eval-bpe > logs/eval_bpe.log 2>&1 &

train-pretrain:
	uv run python -m pretrain.trainer

validate-modernbert:
	uv run python scripts/validate_modernbert.py

export-hf:
	uv run python scripts/export_hf.py $(ARGS)

pretokenization:
	uv run python -m tokenizer.pretokenization
