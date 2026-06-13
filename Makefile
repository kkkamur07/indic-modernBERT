# Run from repo root. Source: indic-modernBert/ (flat imports, package = false).
export PYTHONPATH := indic-modernBert

.PHONY: train-bpe train-superbpe validate-superbpe eval-intrinsic eval-parity pretokenization

train-bpe:
	uv run python -m tokenizer.trainer.bpe_trainer

train-superbpe:
	uv run python -m tokenizer.trainer.superbpe_trainer

validate-superbpe:
	cd indic-modernBert && uv run python scripts/validate_superbpe.py

eval-intrinsic:
	uv run python -m tokenizer.evals.intrinsic

eval-parity:
	uv run python -m tokenizer.evals.parity

pretokenization:
	uv run python -m tokenizer.pretokenization
