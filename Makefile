# Run from repo root. Source: indic-modernBERT/ (flat imports, package = false).
export PYTHONPATH := indic-modernBERT

.PHONY: train-bpe train-superbpe validate-superbpe eval-intrinsic eval-intrinsic-smoke eval-parity pretokenization

train-bpe:
	uv run python -m tokenizer.trainer.bpe_trainer

train-superbpe:
	uv run python -m tokenizer.trainer.superbpe_trainer

validate-superbpe:
	cd indic-modernBERT && uv run python scripts/validate_superbpe.py

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
