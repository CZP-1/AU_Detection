import torch
import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class Identity_Neck(nn.Module):

    def __init__(self):
        super(Identity_Neck, self).__init__()


    def forward(self, inputs):
        return inputs