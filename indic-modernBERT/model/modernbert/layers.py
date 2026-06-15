# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, Tri Dao.


from typing import Optional

import torch
import torch.nn as nn

from model import bert_padding

from .attention import FlexBertAttentionBase, get_attention_layer
from .mlp import FlexBertMLPBase, get_mlp_layer
from .configuration import FlexBertConfig, maybe_add_padding
from .normalization import get_norm_layer
from .initialization import ModuleType, init_weights


class FlexBertLayerBase(nn.Module):
    """A FlexBERT Layer base class for type hints."""

    attn: FlexBertAttentionBase
    mlp: FlexBertMLPBase

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

    def _init_weights(self, reset_params: bool = False):
        if hasattr(self, "attn"):
            self.attn._init_weights(reset_params)
        if hasattr(self, "mlp"):
            self.mlp._init_weights(reset_params)

    def reset_parameters(self):
        self._init_weights(reset_params=True)

    def forward(self, hidden_states: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError("This is a base class and should not be used directly.")


class FlexBertCompileUnpadPreNormLayer(FlexBertLayerBase):
    """Composes the FlexBERT attention and MLP blocks into a single layer using pre-normalization."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        if config.skip_first_prenorm and config.embed_norm and layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = get_norm_layer(config)
        self.attn = get_attention_layer(config, layer_id=layer_id)
        self.mlp_norm = get_norm_layer(config, compiled_norm=config.compile_model)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)
        self.compile_model = config.compile_model

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params:
            self.attn_norm.reset_parameters()
            self.mlp_norm.reset_parameters()

    @torch.compile(dynamic=True)
    def compiled_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.mlp_norm(hidden_states))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen: int
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen)
        """
        attn_out = hidden_states + self.attn(self.attn_norm(hidden_states), cu_seqlens, max_seqlen, indices, attn_mask)
        return attn_out + self.compiled_mlp(attn_out)


class FlexBertUnpadPreNormLayer(FlexBertLayerBase):
    """Composes the FlexBERT attention and MLP blocks into a single layer using pre-normalization."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        if config.skip_first_prenorm and config.embed_norm and layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = get_norm_layer(config)
        self.attn = get_attention_layer(config, layer_id=layer_id)
        self.mlp_norm = get_norm_layer(config)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params:
            self.attn_norm.reset_parameters()
            self.mlp_norm.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen: int
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen)
        """
        attn_out = hidden_states + self.attn(self.attn_norm(hidden_states), cu_seqlens, max_seqlen, indices, attn_mask)
        return attn_out + self.mlp(self.mlp_norm(attn_out))


class FlexBertUnpadParallelPreNormLayer(FlexBertLayerBase):
    """Composes the FlexBERT parallel attention and MLP blocks into a single layer using pre-normalization."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.attn_size = config.hidden_size * 3
        self.mlp_size = config.intermediate_size * 2
        # Compute QKV and FF outputs at once
        self.Wqkvff = nn.Linear(config.hidden_size, self.attn_size + self.mlp_size, bias=config.attn_qkv_bias)
        if config.skip_first_prenorm and config.embed_norm and layer_id == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_norm_layer(config)
        self.attn = get_attention_layer(config, layer_id=layer_id)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params and hasattr(self.norm, "reset_parameters"):
            self.norm.reset_parameters()

        init_weights(
            self.config,
            self.Wqkvff,
            layer_dim=self.config.hidden_size,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (total_nnz, dim)
            attn_mask: None or (batch, max_seqlen)
        """
        # Compute QKV and FF outputs at once and split them
        qkv, intermediate_ff = self.Wqkvff(self.norm(hidden_states)).split([self.attn_size, self.mlp_size], dim=1)
        return hidden_states + self.attn(qkv, cu_seqlens, max_seqlen, indices, attn_mask) + self.mlp(intermediate_ff)


class FlexBertPaddedPreNormLayer(FlexBertLayerBase):
    """Composes the FlexBERT attention and MLP blocks into a single layer using pre-normalization."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        if config.skip_first_prenorm and config.embed_norm and layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = get_norm_layer(config)
        self.attn = get_attention_layer(config, layer_id=layer_id)
        self.mlp_norm = get_norm_layer(config)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params:
            self.attn_norm.reset_parameters()
            self.mlp_norm.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (batch, max_seqlen, dim)
            attn_mask: None or (batch, max_seqlen)
        """
        attn_out = hidden_states + self.attn(self.attn_norm(hidden_states), attn_mask)
        return attn_out + self.mlp(self.mlp_norm(attn_out))


class FlexBertPaddedParallelPreNormLayer(FlexBertLayerBase):
    """Composes the FlexBERT attention and MLP blocks into a single layer using pre-normalization."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.attn_size = config.hidden_size * 3
        self.mlp_size = config.intermediate_size * 2
        # Compute QKV and FF outputs at once
        self.Wqkvff = nn.Linear(config.hidden_size, self.attn_size + self.mlp_size, bias=config.attn_qkv_bias)
        if config.skip_first_prenorm and config.embed_norm and layer_id == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_norm_layer(config)
        self.attn = get_attention_layer(config, layer_id=layer_id)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params:
            self.norm.reset_parameters()

        init_weights(
            self.config,
            self.Wqkvff,
            layer_dim=self.config.hidden_size,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (batch, max_seqlen, dim)
            attn_mask: None or (batch, max_seqlen)
        """
        # Compute QKV and FF outputs at once and split them
        qkv, intermediate_ff = self.Wqkvff(self.norm(hidden_states)).split([self.attn_size, self.mlp_size], dim=2)
        return hidden_states + self.attn(qkv, attn_mask) + self.mlp(intermediate_ff)


class FlexBertUnpadPostNormLayer(FlexBertLayerBase):
    """Composes the FlexBERT attention and MLP blocks into a single layer using post-normalization."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.attn = get_attention_layer(config, layer_id=layer_id)
        self.attn_norm = get_norm_layer(config)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)
        self.mlp_norm = get_norm_layer(config)

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params:
            self.attn_norm.reset_parameters()
            self.mlp_norm.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen: int
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen)
        """
        attn_out = self.attn_norm(hidden_states + self.attn(hidden_states, cu_seqlens, max_seqlen, indices, attn_mask))
        return self.mlp_norm(attn_out + self.mlp(attn_out))


class FlexBertPaddedPostNormLayer(FlexBertLayerBase):
    """Composes the FlexBERT attention and MLP blocks into a single layer using post-normalization."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.attn = get_attention_layer(config, layer_id=layer_id)
        self.attn_norm = get_norm_layer(config)
        self.mlp = get_mlp_layer(config, layer_id=layer_id)
        self.mlp_norm = get_norm_layer(config)

    def _init_weights(self, reset_params: bool = False):
        super()._init_weights(reset_params)
        if reset_params:
            self.mlp_norm.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (batch, max_seqlen, dim)
            attn_mask: None or (batch, max_seqlen)
        """
        attn_out = self.attn_norm(hidden_states + self.attn(hidden_states, attn_mask))
        return self.mlp_norm(attn_out + self.mlp(attn_out))


LAYER2CLS = {
    "unpadded_prenorm": FlexBertUnpadPreNormLayer,
    "unpadded_compile_prenorm": FlexBertCompileUnpadPreNormLayer,
    "unpadded_parallel_prenorm": FlexBertUnpadParallelPreNormLayer,
    "unpadded_postnorm": FlexBertUnpadPostNormLayer,
    "padded_prenorm": FlexBertPaddedPreNormLayer,
    "padded_parallel_prenorm": FlexBertPaddedParallelPreNormLayer,
    "padded_postnorm": FlexBertPaddedPostNormLayer,
}


def get_bert_layer(config: FlexBertConfig, layer_id: Optional[int] = None) -> FlexBertLayerBase:
    try:
        bert_layer = (
            config.initial_bert_layer
            if layer_id < config.num_initial_layers and getattr(config, "initial_bert_layer", None) is not None
            else config.bert_layer
        )
        bert_layer = maybe_add_padding(config, bert_layer)
        if config.compile_model and bert_layer == "unpadded_prenorm":
            bert_layer = "unpadded_compile_prenorm"
        return LAYER2CLS[bert_layer](config, layer_id=layer_id)
    except KeyError:
        if layer_id < config.num_initial_layers and getattr(config, "initial_bert_layer", None) is not None:
            raise ValueError(
                f"Invalid BERT layer type: {config.initial_bert_layer=}, must be one of {LAYER2CLS.keys()}."
                f"{config.padding=} will be automatically prepended to `config.bert_layer` if unspecified."
            )
        else:
            raise ValueError(
                f"Invalid BERT layer type: {config.bert_layer=}, must be one of {LAYER2CLS.keys()}. "
                f"{config.padding=} will be automatically prepended to `config.bert_layer` if unspecified."
            )


class FlexBertEncoderBase(nn.Module):
    """A FlexBERT base class for type hints."""

    layers: nn.ModuleList

    def _init_weights(self, reset_params: bool = False):
        if hasattr(self, "layers"):
            for layer in self.layers:
                layer._init_weights(reset_params=reset_params)

    def reset_parameters(self):
        self._init_weights(reset_params=True)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is a base class and should not be used directly.")


class FlexBertUnpadEncoder(FlexBertEncoderBase):
    """A stack of BERT layers providing the backbone of FlexBERT.

    This module is modeled after the Hugging Face BERT's :class:`~transformers.model.bert.modeling_bert.BertAlibiEncoder`,
    but with substantial modifications to implement unpadding and ALiBi.

    Compared to the analogous Hugging Face BERT module, this module handles unpadding to reduce unnecessary computation
    at padded tokens, and pre-computes attention biases to implement ALiBi.
    """

    def __init__(self, config: FlexBertConfig):
        super().__init__()
        self.layers = nn.ModuleList([get_bert_layer(config, layer_id=i) for i in range(config.num_hidden_layers)])
        self.num_attention_heads = config.num_attention_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        if indices is None and cu_seqlens is None and max_seqlen is None:
            attention_mask_bool = attention_mask.bool()
            batch, seqlen = hidden_states.shape[:2]
            hidden_states, indices, cu_seqlens, max_seqlen = bert_padding.unpad_input(
                hidden_states, attention_mask_bool
            )

            for layer_module in self.layers:
                hidden_states = layer_module(
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    indices,
                    attn_mask=attention_mask,
                )

            return bert_padding.pad_input(hidden_states, indices, batch, seqlen)
        else:
            for layer_module in self.layers:
                hidden_states = layer_module(
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    indices,
                    attn_mask=attention_mask,
                )
            return hidden_states


class FlexBertPaddedEncoder(FlexBertEncoderBase):
    """A stack of BERT layers providing the backbone of FlexBERT.

    This module is modeled after the Hugging Face BERT's :class:`~transformers.model.bert.modeling_bert.BertAlibiEncoder`,
    but with substantial modifications to implement unpadding and ALiBi.

    Compared to the analogous Hugging Face BERT module, this module handles unpadding to reduce unnecessary computation
    at padded tokens, and pre-computes attention biases to implement ALiBi.
    """

    def __init__(self, config: FlexBertConfig):
        super().__init__()
        self.layers = nn.ModuleList([get_bert_layer(config, layer_id=i) for i in range(config.num_hidden_layers)])
        self.num_attention_heads = config.num_attention_heads

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, attn_mask=attention_mask)

        return hidden_states


ENC2CLS = {
    "unpadded_base": FlexBertUnpadEncoder,
    "padded_base": FlexBertPaddedEncoder,
}


def get_encoder_layer(config: FlexBertConfig) -> FlexBertEncoderBase:
    try:
        return ENC2CLS[maybe_add_padding(config, config.encoder_layer)](config)
    except KeyError:
        raise ValueError(
            f"Invalid encoder layer type: {config.encoder_layer=}, must be one of {ENC2CLS.keys()}. "
            f"{config.padding=} will be automatically prepended to `config.encoder_layer` if unspecified."
        )
