# Run from repo root. Source: indic-modernBERT/ (flat imports, package = false).
export PYTHONPATH := indic-modernBERT

.PHONY: train-bpe train-superbpe train-superbpe-nohup train-pretrain \
	validate-superbpe validate-modernbert export-hf \
	eval-intrinsic eval-intrinsic-smoke eval-parity pretokenization

train-bpe:
	uv run python -m tokenizer.trainer.bpe_trainer

train-superbpe:
	uv run python -m tokenizer.trainer.superbpe_trainer

train-superbpe-nohup:
	mkdir -p logs
	PYTHONUNBUFFERED=1 nohup $(MAKE) train-superbpe > logs/train_superbpe_ablation.log 2>&1 &

train-pretrain:
	uv run python -m pretrain.trainer

validate-superbpe:
	uv run python scripts/validate_superbpe.py

validate-modernbert:
	uv run python scripts/validate_modernbert.py

export-hf:
	uv run python scripts/export_hf.py $(ARGS)

eval-intrinsic:
	uv run python -m tokenizer.evals.intrinsic

eval-intrinsic-smoke:
	uv run python -m tokenizer.evals.intrinsic \
		tokenizer.evals.intrinsic.tokenizer_path=artifacts/tokenizer/smoke/superbpe/tokenizer.json \
		tokenizer.evals.intrinsic.data_root=data/smoke/eval/hi

eval-parity:
	uv run python -m tokenizer.evals.parity

pretokenization:
	uv run python -m tokenizer.pretokenization
