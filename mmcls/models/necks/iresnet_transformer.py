# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import NECKS

from typing import Optional, Any, Union, Callable
from torch import Tensor
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from mmcv.runner import BaseModule


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class MyTransformerEncoderLayer(nn.Module):
   
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MyTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)

        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)



        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)


        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)


        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(MyTransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
       

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

@NECKS.register_module()
class IResNet_Transformer_Neck(nn.Module):

    def __init__(self, in_channel, num_classes, embed_dim=256, num_layers=6):
        super(IResNet_Transformer_Neck, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)
        

        self.fcs = nn.Sequential()
        for i in range(num_classes):
            self.fcs.add_module('neck_fc{}'.format(i), nn.Linear(in_channel, embed_dim))
        
        self.encoder_layers = nn.Sequential()
        for i in range(num_layers):
            self.encoder_layers.add_module('neck_transformer_layer{}'.format(i), MyTransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True))


    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[-1]

        fc_outs = []
        for i in range(self.num_classes):
            fc_outs.append(self.fcs[i](inputs))
        
        transformer_inputs = torch.stack(fc_outs, dim=1)

        outs = self.encoder_layers(transformer_inputs)

        # tmp = self.encoder_layer(transformer_inputs)
        # outs = self.transformer_encoder(transformer_inputs)
        # outs = tmp

        return outs
