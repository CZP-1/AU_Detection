# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import NECKS

from typing import Optional, Any, Union, Callable
from torch import Tensor

from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.cnn.bricks.transformer import FFN
from mmcv.runner.base_module import BaseModule
from ..utils import MultiheadAttention
from mmcv.cnn import build_norm_layer
from .gap import GlobalAveragePooling

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


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super(TransformerEncoderLayer, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = self.ffn(self.norm2(x), identity=x)
        return x

@NECKS.register_module()
class TransFER_MMvit_Neck(BaseModule):
    '''
    input x (bs, c, h, w)
    '''
    def __init__(self, in_channel, num_classes, feedforward_channels, local_attention_num=12, lanet_channel_reduction_rate=2,
                 p1=0.2, vit_embed_dim=768, msa_heads_num=8, num_layers=6, use_mad=False, init_cfg=None):
        super(TransFER_MMvit_Neck, self).__init__(init_cfg)
        
        seq_num = 7 * 7

        self.in_channel = in_channel
        self.num_classes = num_classes
        self.local_attention_num = local_attention_num
        self.msa_heads_num = msa_heads_num
        self.lanet_channel_reduction_rate = lanet_channel_reduction_rate
        self.p1 = p1
        self.use_mad = use_mad
        self.vit_embed_dim = vit_embed_dim
        self.feedforward_channels = feedforward_channels

        self.conv = nn.Conv2d(in_channel, vit_embed_dim, 1)

        self.cls_tokens = nn.Parameter(torch.zeros(1, 1, vit_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_num + 1, vit_embed_dim))

        self.local_cnn = LocalCNN(in_channel=in_channel, local_attention_num=local_attention_num,
                               channel_reduction_rate=lanet_channel_reduction_rate,
                               p1=p1, use_mad=use_mad)
        
        # self.gab = GlobalAveragePooling()

        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module('layers{}'.format(i), TransformerEncoderLayer(embed_dims=vit_embed_dim, num_heads=msa_heads_num,
            feedforward_channels=feedforward_channels, init_cfg=None))

    def init_weights(self):
        # pass
        super(TransFER_MMvit_Neck, self).init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[-1]

        bs = inputs.shape[0]
        local_atten_map = self.local_cnn(inputs) # (b, 1, h, w)
        inputs = torch.mul(local_atten_map, inputs) #(b, 512, h, w)
        inputs = self.conv(inputs) # (b, vit_embed_dim, h, w)
        inputs = inputs.contiguous().view(bs, self.vit_embed_dim, -1).permute(0, 2, 1) #(b, h*w, vit_embed_dim)

        cls_tokens = self.cls_tokens.expand(bs, -1, -1)
        inputs = torch.cat((cls_tokens, inputs), dim=1)  # [B, h*w+1, vit_embed_dim]

        inputs = inputs + self.pos_embed

        outs = self.layers(inputs)[:, 0]
        # outs = self.gab(inputs)


        return outs