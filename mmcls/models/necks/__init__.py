# Copyright (c) OpenMMLab. All rights reserved.
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .hr_fuse import HRFuseScales
from .multi_label_csra import Residual_Attention_Neck, Residual_Attention_Neck_Official
from .iresnet_transformer import  IResNet_Transformer_Neck
from .featuremap_attention_neck import FeatureMap_Attention_Neck
from .trans_fer_neck import TransFER_Neck
from .trans_fer_mmclsvit_neck import TransFER_MMvit_Neck
from .anfl import Anfl_Neck
from .mefl import Mefl_Neck
from .identity import Identity_Neck


__all__ = ['GlobalAveragePooling', 'GeneralizedMeanPooling', 'HRFuseScales', 
           'Residual_Attention_Neck', 'IResNet_Transformer_Neck', 'FeatureMap_Attention_Neck',
           'Residual_Attention_Neck_Official', 'TransFER_Neck', 'TransFER_MMvit_Neck',
           'Anfl_Neck', 'Mefl_Neck', 'Identity_Neck']
