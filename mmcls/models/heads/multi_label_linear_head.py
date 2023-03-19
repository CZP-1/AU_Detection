# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .multi_label_head import MultiLabelClsHead

class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, eps=1e-8, disable_torch_grad=True, weight=None):
        super(WeightedAsymmetricLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps
        self.weight = weight

    def forward(self, x, y):

        bs = x.shape[0]        
        x = torch.sigmoid(x)

        losses = {}
        xs_pos = x
        xs_neg = 1 - x

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # # Asymmetric Focusing
        # if self.disable_torch_grad:
        #     torch.set_grad_enabled(False)
        # neg_weight = 1 - xs_neg
        # if self.disable_torch_grad:
        #     torch.set_grad_enabled(True)
        # loss = los_pos + neg_weight * los_neg
        loss = -(los_pos + los_neg).sum() / bs

        # loss = -(y * torch.log(x) + (1 - y) * torch.log(1 - x)).mean()

        if self.weight is not None:
            loss = loss * self.weight.view(1,-1)

        # loss = loss.mean(dim=-1)
        losses['loss'] = loss
        return losses

@HEADS.register_module()
class MultiLabelLinearClsHead(MultiLabelClsHead):
    """Linear classification head for multilabel task.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 use_centerloss=False,
                 use_circleloss=True,
                 use_mixup=False,
                 circleloss_thr_pos=0.9,
                 circleloss_thr_neg=0.1,
                 use_bceloss=True,
                 use_csra=False,

                 use_transformer_neck=False,
                 transformer_embed_dim=None,

                 use_featuremap_transformer=False,

                 use_two_backbone=False,
                 use_anti_person_loss=False,

                 use_anfl=False,
                 lam = 0.7,

                use_anfl_with_csra=False,
                use_anfl_with_backbone=False,
                use_fc_fusion_weight=False,
                use_attn_fusion_weight=False,
                 
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01)):
        super(MultiLabelLinearClsHead, self).__init__(
            loss=loss, init_cfg=init_cfg)

        if num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')


        self.in_channels = in_channels
        self.num_classes = num_classes

        self.use_centerloss = use_centerloss
        self.use_circleloss = use_circleloss
        self.use_mixup = use_mixup
        self.use_bceloss = use_bceloss
        self.use_csra = use_csra

        self.use_transformer_neck = use_transformer_neck
        self.transformer_embed_dim = transformer_embed_dim

        self.use_featuremap_transformer = use_featuremap_transformer

        self.use_two_backbone = use_two_backbone
        self.use_anti_person_loss = use_anti_person_loss
        
        self.use_anfl = use_anfl
        self.lam = lam

        self.use_anfl_with_backbone = use_anfl_with_backbone
        self.use_fc_fusion_weight = use_fc_fusion_weight 
        self.use_attn_fusion_weight = use_attn_fusion_weight 

        self.use_anfl_with_csra = use_anfl_with_csra 

        self.circleloss_thr_pos = circleloss_thr_pos
        self.circleloss_thr_neg = circleloss_thr_neg

        # when use iresnet backbone with transformer neck
        if self.use_transformer_neck:
            assert self.transformer_embed_dim != None

        if self.use_anti_person_loss:
            assert self.use_two_backbone
        
        # when use iresnet backbone with transformer neck
        if self.use_transformer_neck:
            self.fcs = nn.Sequential()
            for i in range(num_classes):
                # self.fcs.add_module('head_fc{}'.format(i), nn.Linear(self.transformer_embed_dim, 1))
                self.fcs.add_module('head_fc{}'.format(i), nn.Linear(self.in_channels, 1))

        elif self.use_csra:
            pass
        
        # elif self.use_anfl:
        #     if self.use_anfl_with_backbone and not self.use_fc_fusion_weight:
        #         # self.fcs = nn.Sequential()
        #         # for i in range(num_classes):
        #         #     self.fcs.add_module('head_fc{}'.format(i), nn.Linear(self.in_channels, 1))
                
        #         self.fc = nn.Linear(self.in_channels, 1) 
        #         self.fc_backbone_out = nn.Linear(self.in_channels, self.num_classes)
        #     elif self.use_anfl_with_backbone and self.use_fc_fusion_weight:
        #         self.fc = nn.Linear(self.in_channels, 1) 
        #         # self.fcs = nn.Sequential()
        #         # for i in range(num_classes):
        #         #     self.fcs.add_module('head_fc{}'.format(i), nn.Linear(self.in_channels, 1))
        #         self.fc_backbone_out = nn.Linear(self.in_channels, self.num_classes)
        #         self.fc_fusion_weight = nn.Linear(self.in_channels, self.num_classes)


        elif self.use_anfl:
            if self.use_anfl_with_backbone:
                if self.use_fc_fusion_weight or self.use_attn_fusion_weight:
                    self.fc = nn.Linear(self.in_channels, 1) 
                    self.fc_backbone_out = nn.Linear(self.in_channels, self.num_classes)
                    self.fc_fusion_weight = nn.Linear(self.in_channels, self.num_classes)
                # self.fc = nn.Linear(self.in_channels, 1) 
                # self.fc_backbone_out = nn.Linear(self.in_channels, self.num_classes)
                else:
                    self.fc = nn.Linear(self.in_channels, 1) 
                    self.fc_backbone_out = nn.Linear(self.in_channels, self.num_classes)
            elif self.use_anfl_with_csra:
                self.fc = nn.Linear(self.in_channels, 1) 
            else:
                self.fc = nn.Linear(self.in_channels, 1)

        else:
            self.fc = nn.Linear(self.in_channels, self.num_classes) if not self.use_anfl else nn.Linear(self.in_channels, 1) 

        # elif self.use_anfl and self.use_anfl_with_backbone:
        #     self.fc = nn.Linear(self.in_channels, 1) 
        #     self.fc_backbone_out = nn.Linear(self.in_channels, self.num_classes)

        # else:
        #     self.fc = nn.Linear(self.in_channels, self.num_classes) if not self.use_anfl else nn.Linear(self.in_channels, 1) 

    
        self.criterion = WeightedAsymmetricLoss()

    def pre_logits(self, x):
        if isinstance(x, tuple) and self.use_anfl and self.use_anfl_with_backbone:
            pass
        elif isinstance(x, tuple):
            x = x[-1]
        return x

    def forward_train(self, x, gt_label, **kwargs):
        if not self.use_two_backbone:
            x = self.pre_logits(x)
        else: 
            x_anti_person = x[1]
            x = x[0]

        if isinstance(x, tuple): 
            bs = x[0].shape[0]
            gt_label = gt_label.type_as(x[0])
        else:
            bs = x.shape[0]
            gt_label = gt_label.type_as(x)
        

        # if self.use_anfl:
        #     if self.use_anfl_with_csra:
        #         x_, csra_out = x[0], x[1]
        #         cls_score = (1 - self.lam) * self.fc(x_).view(bs, self.num_classes) + self.lam * csra_out
        #     elif self.use_anfl_with_backbone and not self.use_fusion_weight:
        #         x_, backbone_out = x[0], x[1]
        #         # x_out = []
        #         # for i in range(self.num_classes):
        #         #     x_out.append(self.fcs[i](x_[:, i, :]))
                
        #         # cls_score_tmp = torch.stack(x_out, dim=1).view(bs, -1)
        #         # cls_score = (1 - self.lam) * cls_score_tmp + self.lam * self.fc_backbone_out(backbone_out)
        #         cls_score = (1 - self.lam) * self.fc(x_).view(bs, self.num_classes) + self.lam * self.fc_backbone_out(backbone_out)

        #     elif self.use_anfl_with_backbone and self.use_fusion_weight:
        #         x_, backbone_out = x[0], x[1]
        #         # x_out = []
        #         # for i in range(self.num_classes):
        #         #     x_out.append(self.fcs[i](x_[:, i, :]))

        #         # cls_score_tmp = torch.stack(x_out, dim=1).view(bs, -1)
        #         fusion_weight = torch.sigmoid(self.fc_fusion_weight(backbone_out))
        #         # cls_score = (1 - fusion_weight) * cls_score_tmp + fusion_weight * self.fc_backbone_out(backbone_out)
        #         cls_score = (1 - fusion_weight) * self.fc(x_).view(bs, self.num_classes) + fusion_weight * self.fc_backbone_out(backbone_out)

        #     else:   
        #         cls_score = self.fc(x).view(bs, self.num_classes)
        
        if self.use_anfl:
            if self.use_anfl_with_csra:
                x_, csra_out = x[0], x[1]
                cls_score = (1 - self.lam) * self.fc(x_).view(bs, self.num_classes) + self.lam * csra_out
            elif self.use_anfl_with_backbone:
                if self.use_fc_fusion_weight:
                    x_, backbone_out = x[0], x[1]
                    fusion_weight = torch.sigmoid(self.fc_fusion_weight(backbone_out))
                    cls_score = (1 - fusion_weight) * self.fc(x_).view(bs, self.num_classes) + fusion_weight * self.fc_backbone_out(backbone_out)
                # x_, backbone_out = x[0], x[1]
                # cls_score = (1 - self.lam) * self.fc(x_).view(bs, self.num_classes) + self.lam * self.fc_backbone_out(backbone_out)
                elif self.use_attn_fusion_weight:
                    x_, backbone_out, trans_outs = x[0], x[1], x[2]
                    fusion_weight = torch.sigmoid(self.fc_fusion_weight(trans_outs))
                    cls_score = (1 - fusion_weight) * self.fc(x_).view(bs, self.num_classes) + fusion_weight * self.fc_backbone_out(backbone_out)
                else:
                    x_, backbone_out = x[0], x[1]
                    cls_score = (1 - self.lam) * self.fc(x_).view(bs, self.num_classes) + self.lam * self.fc_backbone_out(backbone_out)


            else:   
                cls_score = self.fc(x).view(bs, self.num_classes)
        
            

        elif self.use_transformer_neck:
            x_out = []
            for i in range(self.num_classes):
                x_out.append(self.fcs[i](x[:, i, :]))
            
            cls_score = torch.stack(x_out, dim=1).view(bs, -1)

        elif self.use_featuremap_transformer:
            cls_score = self.fc(torch.flatten(x, 1))

        elif self.use_csra:
            cls_score = x

        else:
            cls_score = self.fc(x)
        
        if self.use_centerloss:
            losses = self.loss_centerLoss(x, gt_label, cls_score)
        elif self.use_bceloss:
            losses = self.loss(cls_score, gt_label, **kwargs)
            # losses = self.criterion(cls_score_, gt_label)

        else:
            losses = {}

        # circle_loss with mixup old2
        if self.use_circleloss & self.use_mixup:
            pos_idx = torch.where(gt_label >= self.circleloss_thr_pos)
            pos_other_idx = torch.where(gt_label < self.circleloss_thr_pos)
            neg_other_idx = torch.where(gt_label >= self.circleloss_thr_neg)
            
            y_pred_neg = cls_score.clone()
            y_pred_pos = cls_score.clone()
            y_pred_pos[pos_idx] = -y_pred_pos[pos_idx]
            # cls_score[pos_idx] = -cls_score[pos_idx]
            # cls_score[neg_idx] = cls_score[neg_idx]
            # tmp = cls_score.clone()
            
            y_pred_neg[neg_other_idx] = y_pred_neg[neg_other_idx] - 1e12
            y_pred_pos[pos_other_idx] = y_pred_pos[pos_other_idx] - 1e12
            

            zeros = torch.zeros_like(cls_score[..., :1])
            y_pred_neg = torch.cat([y_pred_neg, zeros], axis=-1)
            y_pred_pos = torch.cat([y_pred_pos, zeros], axis=-1)

            neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
            pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
            circle_loss = torch.mean(neg_loss + pos_loss)
            losses['circle_loss'] = circle_loss

        # circle_loss without mixup
        if self.use_circleloss & (not self.use_mixup) :
            cls_score = (1 - 2 * gt_label) * cls_score
            y_pred_neg = cls_score - gt_label * 1e12
            y_pred_pos = cls_score - (1 - gt_label) * 1e12
            zeros = torch.zeros_like(cls_score[..., :1])
            y_pred_neg = torch.cat([y_pred_neg, zeros], axis=-1)
            y_pred_pos = torch.cat([y_pred_pos, zeros], axis=-1)

            neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
            pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
            circle_loss = torch.mean(neg_loss + pos_loss)
            losses['circle_loss'] = circle_loss

        if self.use_anti_person_loss:
            anti_target = torch.full((x.shape[0],), -1).cuda(x.device.index)
            anti_person_loss = F.cosine_embedding_loss(x, x_anti_person, anti_target, margin=-0.9)
            losses['anti_person_loss'] = anti_person_loss

        # ## circle_loss with mixup new
        # if self.use_circleloss & self.use_mixup:
        #     eps = 1e-7
        #     gt_label[gt_label < eps] = eps
        #     gt_label[gt_label > 1- eps] = 1 - eps

        #     y_pred_pos = cls_score + torch.log(1 - gt_label)
        #     y_pred_neg = -cls_score + torch.log(gt_label)

        #     zeros = torch.zeros_like(cls_score[..., :1])
        #     y_pred_pos = torch.cat([y_pred_pos, zeros], axis=-1)
        #     y_pred_neg = torch.cat([y_pred_pos, zeros], axis=-1)

        #     neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
        #     pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
        #     circle_loss = torch.mean(neg_loss + pos_loss)
        #     losses['circle_loss'] = circle_loss


        return losses

    def simple_test(self, x, sigmoid=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            sigmoid (bool): Whether to sigmoid the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        if not self.use_two_backbone:
            x = self.pre_logits(x)
        else: 
            x = x[0]

        if isinstance(x, tuple): 
            bs = x[0].shape[0]
        else:
            bs = x.shape[0]

        if self.use_anfl:
            if self.use_anfl_with_csra:
                x_, csra_out = x[0], x[1]
                cls_score = (1 - self.lam) * self.fc(x_).view(bs, self.num_classes) + self.lam * csra_out
            elif self.use_anfl_with_backbone:
                if self.use_fc_fusion_weight:
                    x_, backbone_out = x[0], x[1]
                    fusion_weight = torch.sigmoid(self.fc_fusion_weight(backbone_out))
                    cls_score = (1 - fusion_weight) * self.fc(x_).view(bs, self.num_classes) + fusion_weight * self.fc_backbone_out(backbone_out)
                elif self.use_attn_fusion_weight:
                    x_, backbone_out, trans_outs = x[0], x[1], x[2]
                    fusion_weight = torch.sigmoid(self.fc_fusion_weight(trans_outs))
                    cls_score = (1 - fusion_weight) * self.fc(x_).view(bs, self.num_classes) + fusion_weight * self.fc_backbone_out(backbone_out)
                else:
                    x_, backbone_out = x[0], x[1]
                    cls_score = (1 - self.lam) * self.fc(x_).view(bs, self.num_classes) + self.lam * self.fc_backbone_out(backbone_out)


            else:   
                cls_score = self.fc(x).view(bs, self.num_classes)

        elif self.use_transformer_neck:
            x_out = []
            for i in range(self.num_classes):
                x_out.append(self.fcs[i](x[:, i, :]))
            
            cls_score = torch.stack(x_out, dim=1).view(bs, -1)

        elif self.use_featuremap_transformer:
            cls_score = self.fc(torch.flatten(x, 1))

        elif self.use_csra:
            cls_score = x
        
        else:
            cls_score = self.fc(x) if not self.use_csra else self.fc(x).view(bs, -1)

        if self.use_anfl:
            pred = torch.sigmoid(cls_score) if cls_score is not None else None
        elif sigmoid & self.use_circleloss & self.use_mixup:
            pred = torch.sigmoid(cls_score) if cls_score is not None else None
        elif sigmoid:
            pred = torch.sigmoid(cls_score) if cls_score is not None else None
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

