# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import NECKS

from typing import Optional, Any, Union, Callable
from torch import Tensor

# from mmcv.cnn.bricks.transformer import FFN
# from mmcv.runner.base_module import BaseModule
# from ..utils import MultiheadAttention
# from mmcv.cnn import build_norm_layer

from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from .gap import GlobalAveragePooling

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class MAD(nn.Module):
    def __init__(self, local_attention_num=12, p=0.2):
        super(MAD, self).__init__()
        self.local_attention_num = local_attention_num
        self.p = p

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[-1]
        bs = inputs.shape[0]
        index = torch.randint(0, self.local_attention_num, (bs,))
        
        for i in range(bs):
            if torch.rand(1,) > (1.0 - self.p):
                # inputs[0, index[0], ...] = 0
                inputs[i, index[i], ...] = 0 
             
        # print(inputs[0, index[0], ...])
        return inputs


class LocalCNN(nn.Module):
    def __init__(self, in_channel, local_attention_num=12, channel_reduction_rate=2, 
                 p1=0.2, use_mad=False):
        super(LocalCNN, self).__init__()
        self.in_channel = in_channel
        self.channel_reduction_rate = channel_reduction_rate
        self.local_attention_num = local_attention_num
        self.use_mad = use_mad
        
        self.lanets = nn.Sequential()
        for i in range(local_attention_num):
            self.lanets.add_module('neck_lanets{}'.format(i), 
                                   nn.Sequential(nn.Conv2d(in_channel, in_channel//channel_reduction_rate, 1),
                                                 nn.Conv2d(in_channel//channel_reduction_rate, 1, 1)))
        if self.use_mad:
            self.mad = MAD(local_attention_num=local_attention_num, p=p1)

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[-1]

        lanet_outs = []
        for i in range(self.local_attention_num):
            lanet_outs.append(self.lanets[i](inputs))
        lanet_outs = tuple(lanet_outs)
        
        outs = torch.cat(lanet_outs, dim=1) # (bs, atten_map_num, h, w)
        outs = torch.sigmoid(outs)
        if self.use_mad:
            outs = self.mad(outs)
        outs = torch.max(outs, 1, keepdim=True)[0]
        return outs


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
class TransFER_Neck(nn.Module):
    '''
    input x (bs, c, h, w)
    '''
    def __init__(self, in_channel, num_classes, local_attention_num=12, lanet_channel_reduction_rate=2,
                 p1=0.2, vit_embed_dim=768, msa_heads_num=8, num_layers=6, use_mad=False):
        super(TransFER_Neck, self).__init__()
        
        seq_num = 7 * 7

        self.in_channel = in_channel
        self.num_classes = num_classes
        self.local_attention_num = local_attention_num
        self.msa_heads_num = msa_heads_num
        self.lanet_channel_reduction_rate = lanet_channel_reduction_rate
        self.p1 = p1
        self.use_mad = use_mad
        self.vit_embed_dim = vit_embed_dim

        self.conv = nn.Conv2d(in_channel, vit_embed_dim, 1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_num + 1, vit_embed_dim))

        self.local_cnn = LocalCNN(in_channel=in_channel, local_attention_num=local_attention_num,
                               channel_reduction_rate=lanet_channel_reduction_rate,
                               p1=p1, use_mad=use_mad)
        
        self.encoder_layers = nn.Sequential()
        for i in range(num_layers):
            self.encoder_layers.add_module('neck_transformer_layer{}'.format(i), MyTransformerEncoderLayer(d_model=vit_embed_dim, nhead=msa_heads_num, batch_first=True))

        # self.gab = GlobalAveragePooling()

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[-1]

        bs = inputs.shape[0]
        local_atten_map = self.local_cnn(inputs) # (b, 1, h, w)
        inputs = torch.mul(local_atten_map, inputs) #(b, 512, h, w)

        inputs = self.conv(inputs) # (b, vit_embed_dim, h, w)
        inputs = inputs.contiguous().view(bs, self.vit_embed_dim, -1).permute(0, 2, 1) #(b, h*w, vit_embed_dim)

        cls_token = self.cls_token.expand(bs, -1, -1)
        inputs = torch.cat((cls_token, inputs), dim=1)  # [B, h*w+1, vit_embed_dim]

        inputs = inputs + self.pos_embed

        outs = self.encoder_layers(inputs)[:, 0]
        # outs = self.gab(inputs)

        return outs