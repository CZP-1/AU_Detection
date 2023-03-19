# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.necks import GlobalAveragePooling
from ..builder import NECKS
from mmcv.runner import BaseModule, ModuleList



@NECKS.register_module()
class Residual_Attention_Neck(nn.Module):
    """Residual_Attention_Neck

    Args:
    in_channel: the channel of backbone's output[-1]
    lamda: the weight of class-specific residual attention
    """

    def __init__(self, in_channel, num_classes, lamda):
        super(Residual_Attention_Neck, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.lamda = lamda
        self.fc=nn.Conv2d(in_channels=self.in_channel,out_channels=self.num_classes, kernel_size=1, stride=1, bias=False)
        self.gap = GlobalAveragePooling()

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[-1]
        bs = inputs.shape[0]
        gap_outputs = self.gap(inputs).unsqueeze(2).repeat(1, 1, 12)

        assert isinstance(inputs, torch.Tensor)
        outs = self.fc(inputs)
        outs = outs.view(bs, self.num_classes, -1)
        attention_score = F.softmax(outs, dim=-1).permute(0, 2, 1).contiguous()

        inputs = inputs.view(bs, self.in_channel, -1)
        atten_outputs = torch.matmul(inputs, attention_score)
        
        outs = (gap_outputs + self.lamda * atten_outputs).permute(0, 2, 1).contiguous()


        return outs
    

@NECKS.register_module()
class Residual_Attention_Neck_Official(nn.Module):
    """Class-specific residual attention classifier head.

    Residual Attention: A Simple but Effective Method for Multi-Label
                        Recognition (ICCV 2021)
    Please refer to the `paper <https://arxiv.org/abs/2108.02456>`__ for
    details.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        num_heads (int): Number of residual at tensor heads.
        loss (dict): Config of classification loss.
        lam (float): Lambda that combines global average and max pooling
            scores.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """
    temperature_settings = {  # softmax temperature settings
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        6: [1, 2, 3, 4, 5, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99]
    }

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_heads,
                 lam,
                 use_gap=True):
        assert num_heads in self.temperature_settings.keys(
        ), 'The num of heads is not in temperature setting.'
        assert lam > 0, 'Lambda should be between 0 and 1.'
        super(Residual_Attention_Neck_Official, self).__init__()
        self.temp_list = self.temperature_settings[num_heads]
        self.csra_heads = ModuleList([
            CSRAModule(num_classes, in_channels, self.temp_list[i], lam, use_gap)
            for i in range(num_heads)
        ])


    def forward(self, x):
        logit = 0.
        for head in self.csra_heads:
            logit += head(x)
        return logit

class CSRAModule(BaseModule):
    """Basic module of CSRA with different temperature.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        T (int): Temperature setting.
        lam (float): Lambda that combines global average and max pooling
            scores.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self, num_classes, in_channels, T, lam, use_gap=True, init_cfg=None):

        super(CSRAModule, self).__init__(init_cfg=init_cfg)
        self.T = T  # temperature
        self.lam = lam  # Lambda
        self.head = nn.Conv2d(in_channels, num_classes, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.use_gap = use_gap

    def forward(self, x):
        score = self.head(x) / torch.norm(
            self.head.weight, dim=1, keepdim=True).transpose(0, 1)
        score = score.flatten(2) # (bs, numcls, h*w)

        if self.use_gap:
            base_logit = torch.mean(score, dim=2)
        else:
            base_logit = 0.0

        if self.T == 99:  # max-pooling
            att_logit = torch.max(score, dim=2)[0]
        else:
            score_soft = self.softmax(score * self.T) # (bs, numcls, h*w)
            att_logit = torch.sum(score * score_soft, dim=2)

        return base_logit + self.lam * att_logit