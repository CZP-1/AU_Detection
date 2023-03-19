from sklearn.utils import compute_class_weight
import torch
import torch.nn as nn
from ..builder import LOSSES
from .utils import weight_reduce_loss
import torch.nn.functional as F

def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         pos_weight=None):
    r"""Calculate the binary CrossEntropy loss with logits.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        label (torch.Tensor): The gt label with shape (N, \*).
        weight (torch.Tensor, optional): Element-wise weight of loss with shape
            (N, ). Defaults to None.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
            is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (torch.Tensor, optional): The positive weight for each
            class with shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # Ensure that the size of class_weight is consistent with pred and label to
    # avoid automatic boracast,
    assert pred.dim() == label.dim()

    if class_weight is not None:
        N = pred.size()[0]
        class_weight = class_weight.repeat(N, 1)
    loss = F.binary_cross_entropy_with_logits(
        pred,
        label,
        weight=class_weight,
        pos_weight=pos_weight,
        reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss

@LOSSES.register_module()
class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True, 
        reduction='mean', loss_weight=1.0, alpha=0.001, lr=0.1, lr_cent=0.5, 
        multi_label=False, pos_weight=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.cls_criterion = cross_entropy if multi_label == False else binary_cross_entropy

        self.alpha = alpha
        self.lr = lr
        self.lr_cent = lr_cent

        self.multi_label = multi_label
        self.pos_weight = pos_weight
        
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda(0))
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels, pred, avg_factor=None):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
            pred: (batch_size, num_cls)
        """
        batch_size = x.size(0)

        # tmp = torch.pow(x, 2).sum(dim=1, keepdim=True)
        # tmp2 = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
        # tmp3 = torch.pow(self.centers, 2).sum(dim=1, keepdim=True)
        # tmp4 = torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # distmat.addmm_(1, -2, x.float(), self.centers.t())
        distmat.addmm_(x.float(), self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda(x.device.index)

        

        if self.multi_label:
            dist = distmat * labels.float()
            loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        else:
            labels_orig = labels
            labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
            mask = labels.eq(classes.expand(batch_size, self.num_classes))

            dist = distmat * mask.float()
            loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        weight = None
        class_weight = None
        if self.multi_label:
            loss_cls = self.loss_weight * self.cls_criterion(
            pred,
            labels,
            weight,
            class_weight=class_weight,
            reduction=self.reduction,
            avg_factor=avg_factor,
            pos_weight=self.pos_weight
            )
        else:
            loss_cls = self.loss_weight * self.cls_criterion(
            pred,
            labels_orig,
            weight,
            class_weight=class_weight,
            reduction=self.reduction,
            avg_factor=avg_factor,
            )
        # loss_total = loss * self.alpha + loss_cls

        # for param in self.parameters():
        #     # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
        #     param.grad.data *= (self.lr_cent / (self.alpha * self.lr))

        return loss_cls, loss * self.alpha