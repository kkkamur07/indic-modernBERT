"""ModernBERT encoder (ported from _support_repo/ModernBERT)."""

from model.factory import build_modernbert_config, create_modernbert_mlm
from model.modernbert.configuration import FlexBertConfig as ModernBertConfig
from model.modernbert.model import FlexBertForMaskedLM as ModernBertForMaskedLM
from model.modernbert.model import FlexBertModel as ModernBertModel

__all__ = [
    "ModernBertConfig",
    "ModernBertForMaskedLM",
    "ModernBertModel",
    "build_modernbert_config",
    "create_modernbert_mlm",
]
