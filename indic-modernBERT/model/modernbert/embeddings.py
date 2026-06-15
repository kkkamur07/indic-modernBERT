# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, Tri Dao.


import torch
import torch.nn as nn
from typing import Optional

from .configuration import FlexBertConfig
from .normalization import get_norm_layer
from .initialization import ModuleType, init_weights


class FlexBertEmbeddingsBase(nn.Module):
    """A FlexBERT embeddings base class for type hints."""

    def __init__(self, config: FlexBertConfig):
        super().__init__()
        self.config = config

    def _init_weights(self, reset_params: bool = False):
        raise NotImplementedError("This is a base class and should not be used directly.")

    def reset_parameters(self):
        self._init_weights(reset_params=True)

    def forward(self, input_ids: torch.LongTensor, position_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        raise NotImplementedError("This is a base class and should not be used directly.")


class FlexBertAbsoluteEmbeddings(FlexBertEmbeddingsBase):
    """Construct the embeddings with absolute positional embeddings."""

    def __init__(self, config: FlexBertConfig):
        super().__init__(config)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.norm = get_norm_layer(config) if config.embed_norm else nn.Identity()
        self.drop = nn.Dropout(config.embed_dropout_prob) if config.embed_dropout_prob > 0.0 else nn.Identity()

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def _init_weights(self, reset_params: bool = False):
        init_weights(self.config, self.tok_embeddings, type_of_module=ModuleType.emb)
        init_weights(self.config, self.position_embeddings, type_of_module=ModuleType.emb)

        if reset_params:
            if self.config.embed_norm:
                self.norm.reset_parameters()  # type: ignore

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = self.position_ids[:, 0 : input_ids.shape[1]]

        embeddings = self.tok_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = self.norm(embeddings + position_embeddings)
        return self.drop(embeddings)


class FlexBertCompiledSansPositionEmbeddings(FlexBertEmbeddingsBase):
    """Construct the embeddings from token embeddings without any positional embeddings."""

    def __init__(self, config: FlexBertConfig):
        super().__init__(config)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.norm = get_norm_layer(config, compiled_norm=config.compile_model) if config.embed_norm else nn.Identity()
        self.drop = nn.Dropout(config.embed_dropout_prob) if config.embed_dropout_prob > 0.0 else nn.Identity()

    def _init_weights(self, reset_params: bool = False):
        init_weights(self.config, self.tok_embeddings, type_of_module=ModuleType.emb)

        if reset_params:
            if self.config.embed_norm:
                self.norm.reset_parameters()  # type: ignore

    @torch.compile(dynamic=True)
    def forward(self, input_ids: torch.LongTensor, position_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        return self.drop(self.norm(self.tok_embeddings(input_ids)))


class FlexBertSansPositionEmbeddings(FlexBertEmbeddingsBase):
    """Construct the embeddings from token embeddings without any positional embeddings."""

    def __init__(self, config: FlexBertConfig):
        super().__init__(config)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.norm = get_norm_layer(config) if config.embed_norm else nn.Identity()
        self.drop = nn.Dropout(config.embed_dropout_prob) if config.embed_dropout_prob > 0.0 else nn.Identity()

    def _init_weights(self, reset_params: bool = False):
        init_weights(self.config, self.tok_embeddings, type_of_module=ModuleType.emb)

        if reset_params:
            if self.config.embed_norm:
                self.norm.reset_parameters()  # type: ignore

    def forward(self, input_ids: torch.LongTensor, position_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        return self.drop(self.norm(self.tok_embeddings(input_ids)))


EBB2CLS = {
    "absolute_pos": FlexBertAbsoluteEmbeddings,
    "sans_pos": FlexBertSansPositionEmbeddings,
}


def get_embedding_layer(config: FlexBertConfig) -> FlexBertEmbeddingsBase:
    try:
        if config.compile_model and config.embedding_layer == "sans_pos":
            return FlexBertCompiledSansPositionEmbeddings(config)
        elif config.compile_model:
            raise ValueError(f"{config.compile_model=} only supports sans_pos embeddings.")
        return EBB2CLS[config.embedding_layer](config)
    except KeyError:
        raise ValueError(f"Invalid embeddings layer type: {config.embedding_layer=}, must be one of {EBB2CLS.keys()}.")
