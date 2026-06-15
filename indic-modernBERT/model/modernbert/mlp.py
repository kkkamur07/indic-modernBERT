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

from .configuration import FlexBertConfig
from .activation import get_act_fn
from .initialization import ModuleType, init_weights


class FlexBertMLPBase(nn.Module):
    """A FlexBERT MLP base class for type hints."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

    def _init_weights(self, reset_params: bool = False):
        raise NotImplementedError("This is a base class and should not be used directly.")

    def reset_parameters(self):
        self._init_weights(reset_params=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is a base class and should not be used directly.")


class FlexBertMLP(FlexBertMLPBase):
    """Applies the MLP at the end of each FlexBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_in_bias)
        self.act = get_act_fn(config.hidden_act)
        self.drop = nn.Dropout(config.mlp_dropout_prob) if config.mlp_dropout_prob > 0.0 else nn.Identity()
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias)

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.Wi,
            layer_dim=self.config.hidden_size,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )
        init_weights(
            self.config,
            self.Wo,
            layer_dim=self.config.intermediate_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute new hidden states from current hidden states.

        Args:
            hidden_states (torch.Tensor): The (unpadded) hidden states from
                the attention layer [nnz, dim].
        """
        return self.Wo(self.drop(self.act(self.Wi(hidden_states))))


class FlexBertGLU(FlexBertMLPBase):
    """Applies the GLU at the end of each FlexBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_in_bias)
        self.act = get_act_fn(config.hidden_act)
        self.drop = nn.Dropout(config.mlp_dropout_prob) if config.mlp_dropout_prob > 0.0 else nn.Identity()
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias)

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.Wi,
            layer_dim=self.config.hidden_size,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )
        init_weights(
            self.config,
            self.Wo,
            layer_dim=self.config.intermediate_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


class FlexBertParallelGLU(FlexBertMLPBase):
    """Applies the GLU at the end of each FlexBERT layer using intermediate_ff computed in parallel of the attention.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.act = get_act_fn(config.hidden_act)
        self.drop = nn.Dropout(config.mlp_dropout_prob) if config.mlp_dropout_prob > 0.0 else nn.Identity()
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias)

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.Wo,
            layer_dim=self.config.intermediate_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(self, intermediate_ff: torch.Tensor) -> torch.Tensor:
        input, gate = intermediate_ff.chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


MLP2CLS = {
    "mlp": FlexBertMLP,
    "glu": FlexBertGLU,
    "parallel_glu": FlexBertParallelGLU,
}


def get_mlp_layer(config: FlexBertConfig, layer_id: Optional[int] = None) -> FlexBertMLPBase:
    try:
        mlp_layer = (
            config.initial_mlp_layer
            if layer_id < config.num_initial_layers and getattr(config, "initial_mlp_layer", None) is not None
            else config.mlp_layer
        )
        return MLP2CLS[mlp_layer](config, layer_id=layer_id)
    except KeyError as e:
        if layer_id < config.num_initial_layers and getattr(config, "initial_mlp_layer", None) is not None:
            raise ValueError(
                f"Invalid MLP layer type: {config.initial_mlp_layer=}, must be one of {MLP2CLS.keys()}. {e}"
            )
        else:
            raise ValueError(f"Invalid MLP layer type: {config.mlp_layer=}, must be one of {MLP2CLS.keys()}. {e}")
