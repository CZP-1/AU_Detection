# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch


def average_performance(pred, target, thrs=None, stage='train', k=None):
    """Calculate CP, CR, CF1, OP, OR, OF1, where C stands for per-class
    average, O stands for overall average, P stands for precision, R stands for
    recall and F1 stands for F1-score.

    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.
        thrs (float | list | array): The confidence threshold. Defaults to None.
        k (int): Top-k performance. Note that if thrs and k are both given, k
            will be ignored. Defaults to None.

    Returns:
        tuple: (CP, CR, CF1, OP, OR, OF1)
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')
    if thrs is None and k is None:
        thrs = [0.5]
        warnings.warn('Neither thr nor k is given, set thr as 0.5 by '
                      'default.')
    elif thrs is not None and k is not None:
        warnings.warn('Both thr and k are given, use threshold in favor of '
                      'top-k.')

    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'

    eps = np.finfo(np.float32).eps
    target[target == -1] = 0

    
    if stage == 'train':
        ACF1s = []
        PCF1s = []
        if thrs is not None:
            # a label is predicted positive if the confidence is no lower than thr
            for thr in thrs:
                pos_inds = pred >= thr
                tp = (pos_inds * target) == 1
                fp = (pos_inds * (1 - target)) == 1
                fn = ((1 - pos_inds) * target) == 1

                precision_class = tp.sum(axis=0) / np.maximum(
                    tp.sum(axis=0) + fp.sum(axis=0), eps)
                recall_class = tp.sum(axis=0) / np.maximum(
                    tp.sum(axis=0) + fn.sum(axis=0), eps)
                CP = precision_class.mean() * 100.0
                CR = recall_class.mean() * 100.0
                CF1 = 2 * CP * CR / np.maximum(CP + CR, eps)

                ACF1 = 2 * precision_class * recall_class / np.maximum(precision_class + recall_class, eps) 
                PCF1 = ACF1 * 100.0
                ACF1 = ACF1.mean() * 100.0

                OP = tp.sum() / np.maximum(tp.sum() + fp.sum(), eps) * 100.0
                OR = tp.sum() / np.maximum(tp.sum() + fn.sum(), eps) * 100.0
                OF1 = 2 * OP * OR / np.maximum(OP + OR, eps)
                
                ACF1s.append(ACF1)
                PCF1s.append(PCF1)

        else:
            # top-k labels will be predicted positive for any example
            sort_inds = np.argsort(-pred, axis=1)
            sort_inds_ = sort_inds[:, :k]
            inds = np.indices(sort_inds_.shape)
            pos_inds = np.zeros_like(pred)
            pos_inds[inds[0], sort_inds_] = 1

            tp = (pos_inds * target) == 1
            fp = (pos_inds * (1 - target)) == 1
            fn = ((1 - pos_inds) * target) == 1

            precision_class = tp.sum(axis=0) / np.maximum(
                tp.sum(axis=0) + fp.sum(axis=0), eps)
            recall_class = tp.sum(axis=0) / np.maximum(
                tp.sum(axis=0) + fn.sum(axis=0), eps)
            CP = precision_class.mean() * 100.0
            CR = recall_class.mean() * 100.0
            CF1 = 2 * CP * CR / np.maximum(CP + CR, eps)

            ACF1 = 2 * precision_class * recall_class / np.maximum(precision_class + recall_class, eps) 
            PCF1 = ACF1 * 100.0
            ACF1 = ACF1.mean() * 100.0

            OP = tp.sum() / np.maximum(tp.sum() + fp.sum(), eps) * 100.0
            OR = tp.sum() / np.maximum(tp.sum() + fn.sum(), eps) * 100.0
            OF1 = 2 * OP * OR / np.maximum(OP + OR, eps)

            ACF1s.append(ACF1)
            PCF1s.append(PCF1)
    else:
        if thrs is not None:
            # a label is predicted positive if the confidence is no lower than thr
            thrs = np.array(torch.Tensor(thrs).expand(pred.shape[0], 12))
            pos_inds = pred >= thrs
            tp = (pos_inds * target) == 1
            fp = (pos_inds * (1 - target)) == 1
            fn = ((1 - pos_inds) * target) == 1

            precision_class = tp.sum(axis=0) / np.maximum(
                tp.sum(axis=0) + fp.sum(axis=0), eps)
            recall_class = tp.sum(axis=0) / np.maximum(
                tp.sum(axis=0) + fn.sum(axis=0), eps)
            
            CP = precision_class.mean() * 100.0
            CR = recall_class.mean() * 100.0
            CF1 = 2 * CP * CR / np.maximum(CP + CR, eps)

            ACF1 = 2 * precision_class * recall_class / np.maximum(precision_class + recall_class, eps) 
            F1_score_test = ACF1.mean() * 100.0
    
    if stage == 'train':
        ACF1s = np.array(ACF1s)
        PCF1s = np.array(PCF1s)
        aus_maxF1_idx = PCF1s.argmax(axis=0)
        aus_maxF1 = PCF1s.max(axis=0)
        ACF1_theory_max = PCF1s[aus_maxF1_idx, range(PCF1s.shape[1])].mean()
        ACF1_max = np.max(ACF1s)

        aus_maxF1_thr = (aus_maxF1_idx + 1) * 0.05

        return CP, CR, CF1, OP, OR, OF1,  ACF1_theory_max, ACF1_max, aus_maxF1_thr, ACF1s, aus_maxF1, PCF1s
    else:
        return F1_score_test