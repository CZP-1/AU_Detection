from mmcls.models import ResNet
from mmcls.models.necks import GlobalAveragePooling
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual_Attention_Neck(nn.Module):
    """Residual_Attention_Neck

    Args:
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
        attention_score = F.softmax(outs, dim=-1).permute(0, 2, 1)

        inputs = inputs.view(bs, self.in_channel, -1)
        atten_outputs = torch.matmul(inputs, attention_score)
        
        outs = (gap_outputs + self.lamda * atten_outputs).permute(0, 2, 1)

        return outs

model = ResNet(
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',)

neck = GlobalAveragePooling()
residual_attention_neck = Residual_Attention_Neck(2048, 12, 0.02)
linear = nn.Linear(2048, 1)

inputs = torch.rand(32, 3, 224, 224)
backbone_outputs = model(inputs)
neck_outputs = residual_attention_neck(backbone_outputs)
linear_outputs = linear(neck_outputs)
linear_outputs = linear(neck_outputs)

print('debug')
# outputs = neck(level_outputs)

# for backbone_output in backbone_outputs:
#     print(tuple(level_out.shape))