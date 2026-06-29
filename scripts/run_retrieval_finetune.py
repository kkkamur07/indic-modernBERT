"""Hydra entrypoint for retrieval fine-tuning.

Mirrors upstream ModernBERT's DPR recipe, localized to Hindi:
  1. Fine-tune each backbone with CachedMultipleNegativesRankingLoss on
     Hindi-translated MS-MARCO triples (unicamp-dl/mmarco, "hindi")
  2. Select/evaluate by nDCG@10 on a carved Hindi subset (default mmarco_hindi)
  4. Optionally run full Hindi retrieval eval on the winner

Single run:
  python scripts/run_retrieval_finetune.py retrieval_ft.learning_rate=8e-5

Optuna LR exploration:
  make retrieval-optuna

After tuning, evaluate the selected trial's `final_model_path` from
`finetune_result.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "indic-modernBERT"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from evals.retrieval_finetune import RetrievalFinetuneConfig, run_retrieval_finetune
from utils.log_helpers import setup_run_log


def _parse_config(cfg: DictConfig) -> RetrievalFinetuneConfig:
    raw = OmegaConf.to_container(cfg.retrieval_ft, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("retrieval_ft must be a mapping")
    return RetrievalFinetuneConfig(**raw)


@hydra.main(
    version_base=None,
    config_path="../configs/hi/retrieval_finetune",
    config_name="hindi_dpr",
)
def main(cfg: DictConfig) -> float | None:
    ft_cfg = _parse_config(cfg)
    setup_run_log(
        f"retrieval_ft__{Path(ft_cfg.backbone).name or ft_cfg.backbone}__lr{ft_cfg.learning_rate}.log"
    )
    logger.info("Retrieval fine-tune config:\n{}", OmegaConf.to_yaml(cfg.retrieval_ft))

    result = run_retrieval_finetune(ft_cfg)

    logger.info("Fine-tune complete. Result:\n{}", json.dumps(result, indent=2))

    avg_score = result.get("selection_score", {}).get("avg_ndcg@10", 0.0)
    logger.info("Selection score (avg nDCG@10): {:.4f}", avg_score)

    return avg_score


if __name__ == "__main__":
    main()
