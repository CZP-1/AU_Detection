import numpy as np
import mmcv
import torch

def get_video_list(anno_path):
    with open(anno_path, 'r') as f1:
        annos = f1.readlines()
    video_list = []
    for item in annos:
        video_list.append(item.split(' ')[0].split('/')[-2])
    return video_list

def get_path_list(anno_path):
    with open(anno_path, 'r') as f1:
        annos = f1.readlines()
    path_list = []
    for item in annos:
        path_list.append(item.split(' ')[0])
    return path_list

def cal_f1_from_mmclsInfer(gt, pred, thrs, eps):
    pos_inds = pred >= thrs
    tp = (pos_inds * gt) == 1
    fp = (pos_inds * (1 - gt)) == 1
    fn = ((1 - pos_inds) * gt) == 1

    precision_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fp.sum(axis=0), eps)
    recall_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fn.sum(axis=0), eps)

    ACF1 = 2 * precision_class * recall_class / np.maximum(precision_class + recall_class, eps) 
    F1_score_test = ACF1.mean() * 100.0

    return F1_score_test

def cal_f1_from_submitLabel(gt, pred, eps):
    
    tp = (pred * gt) == 1
    fp = (pred * (1 - gt)) == 1
    fn = ((1 - pred) * gt) == 1

    precision_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fp.sum(axis=0), eps)
    recall_class = tp.sum(axis=0) / np.maximum(
        tp.sum(axis=0) + fn.sum(axis=0), eps)

    ACF1 = 2 * precision_class * recall_class / np.maximum(precision_class + recall_class, eps) 
    F1_score_test = ACF1.mean() * 100.0

    return F1_score_test

def make_submit_from_mmclsInfer(pred, thrs):
    
    pred_label = np.zeros_like(pred, dtype=np.int64)

    flag = pred >= thrs
    idx = np.where(flag)
    
    pred_label[idx] = 1
    return pred_label

def post_process(window_size, pred_label, video_list):
    dealed_pred_labels = []
    for i in range(len(video_list) // window_size):
        cur_window_label = pred_label[window_size*i: (i+1)*window_size, :]

        if video_list[window_size*i] != video_list[(i+1)*window_size - 1]:
            for label in cur_window_label:
                dealed_pred_labels.append(label)
            continue

        cnt_dict = {}
        for item in cur_window_label:
            # item = ",".join(item)
            item = str(item) 
            # item = item[1: -1].split(' ')
            if item in cnt_dict:
                cnt_dict[item] += 1
            else:
                cnt_dict[item] = 1
        
        max_num = 0
        max_label = ''
        for k, v in cnt_dict.items():
            if v > max_num:
                max_num = v
                max_label = k
        
        dealed_pred_label = np.zeros(12, dtype=np.int64)
        max_label = np.array(max_label[1: -1].split(' '))
        index = np.where(max_label == '1')
        dealed_pred_label[index] = 1
        # dealed_pred_label = np.array(torch.Tensor(dealed_pred_label).repeat(10, 1), dtype=np.int64)
        # dealed_pred_labels.append(dealed_pred_label)
        for _ in range(window_size):
            dealed_pred_labels.append(dealed_pred_label)

    rest_num =  len(video_list) % window_size
    for i in range(rest_num):
        dealed_pred_labels.append(pred_label[-rest_num + i, :])
    
    return np.array(dealed_pred_labels, dtype=np.int64)

def post_process1(window_size, pred_label, video_list):
    dealed_pred_labels = []
    for i in range(len(video_list) // window_size):
        cur_window_label = pred_label[window_size*i: (i+1)*window_size, :]

        if video_list[window_size*i] != video_list[(i+1)*window_size - 1]:
            for label in cur_window_label:
                dealed_pred_labels.append(label)
            continue
        
        dealed_pred_label = np.zeros(12, dtype=np.int64)
        cnt = torch.Tensor(cur_window_label).sum(0)
        flag = cnt > window_size // 2
        idx = np.where(flag)
        dealed_pred_label[idx] = 1
        
        for _ in range(window_size):
            dealed_pred_labels.append(dealed_pred_label)

    rest_num =  len(video_list) % window_size
    for i in range(rest_num):
        dealed_pred_labels.append(pred_label[-rest_num + i, :])
    
    return np.array(dealed_pred_labels, dtype=np.int64)  


    dealed_pred_labels = []
    for idx, item in enumerate(pred_label):
        
        if idx > len(video_list) - window_size:
            break

        cur_window_label = pred_label[idx: idx + window_size, :]
        if video_list[idx] != video_list[idx + window_size - 1]:
            dealed_pred_labels.append(item)
            continue

        dealed_pred_label = np.zeros(12, dtype=np.int64)
        cnt = torch.Tensor(cur_window_label).sum(0)
        flag = cnt > window_size // 2
        idx = np.where(flag)
        dealed_pred_label[idx] = 1

        dealed_pred_labels.append(dealed_pred_label)

    rest_num = window_size - 1
    for i in range(rest_num):
        dealed_pred_labels.append(pred_label[-rest_num + i, :])
    
    return np.array(dealed_pred_labels, dtype=np.int64) 

def post_process3(window_size, pred_label, video_list):
    dealed_pred_labels = []
    for i in range(window_size):
        dealed_pred_labels.append(pred_label[i, :])

    for idx, item in enumerate(pred_label[window_size:, ...]):
        idx += window_size
        
        if idx > len(video_list) - window_size - 1:
            break

        cur_window_label = pred_label[idx - window_size: idx + window_size + 1, :]
        if video_list[idx - window_size] != video_list[idx + window_size]:
            dealed_pred_labels.append(item)
            continue

        dealed_pred_label = np.zeros(12, dtype=np.int64)
        cnt = torch.Tensor(cur_window_label).sum(0)
        flag = cnt > (window_size * 2 + 1) // 2
        idx = np.where(flag)
        dealed_pred_label[idx] = 1

        dealed_pred_labels.append(dealed_pred_label)

    rest_num = window_size
    for i in range(rest_num):
        dealed_pred_labels.append(pred_label[-rest_num + i, :])
    return np.array(dealed_pred_labels, dtype=np.int64)  

def probability_post_process(window_size, pred, video_list):
    dealed_pred_labels = []
    for i in range(window_size):
        dealed_pred_labels.append(pred[i, :])

    for idx, item in enumerate(pred[window_size:, ...]):
        idx += window_size
        
        if idx > len(video_list) - window_size - 1:
            break

        cur_window_label = pred[idx - window_size: idx + window_size + 1, :]
        if video_list[idx - window_size] != video_list[idx + window_size]:
            dealed_pred_labels.append(item)
            continue

        dealed_pred_label = np.array(torch.Tensor(cur_window_label).mean(0))
        dealed_pred_labels.append(dealed_pred_label)

    rest_num = window_size
    for i in range(rest_num):
        dealed_pred_labels.append(pred[-rest_num + i, :])
    return np.array(dealed_pred_labels)

def model_fusion(pred_to, pred_from=None, ex_class=None):
    for cls in ex_class:
        pred_to[:, cls] = pred_from[:, cls]
    return pred_to

def thrs_fusion(thrs_to, thrs_from=None, ex_class=None):
    if thrs_from==None or ex_class==None:
        return thrs_to 
    for cls in ex_class:
        thrs_to[cls] = thrs_from[cls]
    return thrs_to

def npLabel2rawSubmit(pred_label, path_list, target_txt_path):
    annos = []
    for path, label in zip(path_list, pred_label):
        pseudo_label = np.array(['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'])
        idx = np.where(label == 1)
        pseudo_label[idx] = '1'
        str_pseudo_label = ','.join(pseudo_label)

        path_frag = path.split('/')
        annos.append(path_frag[-2] + '/' + path_frag[-1] + ' ' + str_pseudo_label + '\n')

    with open(target_txt_path, 'w') as f: 
        f.writelines(annos)

def npLabel2pesudoLabel(pred_label, path_list, target_txt_path):
    annos = []
    for path, label in zip(path_list, pred_label):
        pseudo_label = np.array(['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'])
        idx = np.where(label == 1)
        pseudo_label[idx] = '1'
        str_pseudo_label = ','.join(pseudo_label)

        annos.append(path + ' ' + str_pseudo_label + '\n')

    final = annos[::10]
    with open(target_txt_path, 'w') as f: 
        f.writelines(final)

def vote_fusion_pred(preds, method):
    pred_offcial, pred_fold1, pred_fold2, pred_fold3, pred_fold4 = preds[0], preds[1], preds[2], preds[3], preds[4]
    if method == 'mean':
        fusion_pred = (pred_offcial+pred_fold1+pred_fold2+pred_fold3+pred_fold4) / 5
    elif method == 'weighted_mean':
        fusion_pred = 0.2013 * pred_offcial + 0.2075 * pred_fold1 + 0.2037 * pred_fold2 + 0.1940 * pred_fold3 + 0.1935 * pred_fold4
    elif method == 'max':
        fusion_pred = np.stack(preds, 0)
        fusion_pred = np.max(preds, 0)
    return fusion_pred

def vote_fusion_thrs(method):
    _thrs_offcial = np.array(thrs_offcial_l4).copy()
    _thrs_fold1 = np.array(thrs_fold1).copy()
    _thrs_fold2 = np.array(thrs_fold2).copy()
    _thrs_fold3 = np.array(thrs_fold3).copy()
    _thrs_fold4 = np.array(thrs_fold4).copy()

    if method == 'mean':
        fusion_thrs = (_thrs_offcial + _thrs_fold1 + _thrs_fold2 + _thrs_fold3 + _thrs_fold4) / 5
    elif method == 'weighted_mean':
        fusion_thrs = 0.2013 * _thrs_offcial + 0.2075 * _thrs_fold1 + 0.2037 * _thrs_fold2 + 0.1940 * _thrs_fold3 + 0.1935 * _thrs_fold4
    elif method == 'fixed':
        fusion_thrs = _thrs_offcial
    return fusion_thrs

eps = np.finfo(np.float32).eps
thrs_offcial_l4 = [0.3, 0.2, 0.2, 0.3, 0.4, 0.4, 0.4, 0.15, 0.15, 0.1, 0.45, 0.15]  # official layer4 w9
thrs_official_l2 = [0.45, 0.15, 0.55, 0.35, 0.5, 0.5, 0.35, 0.35, 0.1, 0.1, 0.45, 0.2]  # official layer2 w7
thrs_offcial_test = [0.35, 0.2, 0.35, 0.4, 0.45, 0.5, 0.4, 0.05, 0.15, 0.15, 0.5, 0.2]
# thrs_offcial_l4 thrs_official_l2 fusion w7
thrs_fold1 = [0.5, 0.3, 0.35 ,0.35, 0.35, 0.4, 0.45, 0.3, 0.15, 0.15, 0.4, 0.2]  # fold1  w7
thrs_fold2 = [0.4, 0.25, 0.3, 0.25, 0.3, 0.3, 0.4, 0.3, 0.1, 0.1, 0.35, 0.1]  # fold2  w7
thrs_fold3 = [0.4, 0.2, 0.15, 0.15, 0.35, 0.4, 0.4, 0.25, 0.2, 0.15, 0.4, 0.2 ]  # fold3  w7
thrs_fold4 = [0.4, 0.35, 0.2, 0.3, 0.25, 0.25, 0.4, 0.1, 0.2, 0.05, 0.35, 0.3]  # fold4  w7
# thrs_test = []

# mode = 'train'
mode = 'test'
# mode = 'generate'
# mode = 'vote'

fusion_flag = False

gt_npy_path = '/home/data/lrd/mmclassification_custom/abaw_gt/Official.npy'
# gt_npy_path = '/home/data/lrd/mmclassification_custom/abaw_gt/fold_4.npy'

if mode != 'vote':
    pred_to_json_path = '/home/data/lrd/mmclassification_custom/abaw_test/final/56.82_official.json'
    pred_from_json_path = ''
    pred_to = np.array(mmcv.load(pred_to_json_path)['class_scores'])

# anno_path = '/home/data/lrd/data/abaw/annotations/val.txt'
# anno_path = '/home/data/lrd/data/abaw/annotations_k_folds/fold4/val.txt'
anno_path = '/home/data/lrd/data/abaw/test_related/official_real_test_fake.txt'
video_list = get_video_list(anno_path)

if mode == 'train':
    gt = np.load(gt_npy_path)


if mode == 'train':
    if fusion_flag:
        pred_from = np.array(mmcv.load(pred_from_json_path)['class_scores'])
        pred_to = model_fusion(pred_to, pred_from, [1, 7, 11])

        thrs = thrs_fusion(thrs_offcial_l4, thrs_official_l2, [1, 7, 11])
        thrs = np.array(torch.Tensor(thrs).expand(pred_to.shape[0], 12))

        F1_score_test = cal_f1_from_mmclsInfer(gt, pred_to, thrs, eps)

        pred_to = probability_post_process(8, pred_to, video_list)
            
        pred_label = make_submit_from_mmclsInfer(pred_to, thrs)
        dealed_pred_labels = post_process3(6, pred_label, video_list)
        # F1_score_test_postprocess = cal_f1_from_submitLabel(gt, pred_label, eps)
        
        F1_score_test_postprocess = cal_f1_from_submitLabel(gt, dealed_pred_labels, eps)

        print(F1_score_test, F1_score_test_postprocess)
    else:
        thrs = np.array(torch.Tensor(thrs_fold4).expand(pred_to.shape[0], 12))

        F1_score_test = cal_f1_from_mmclsInfer(gt, pred_to, thrs, eps)

        pred_to = probability_post_process(8, pred_to, video_list)

        pred_label = make_submit_from_mmclsInfer(pred_to, thrs)
        dealed_pred_labels = post_process3(6, pred_label, video_list)

        F1_score_test_postprocess = cal_f1_from_submitLabel(gt, dealed_pred_labels, eps)
        # F1_score_test_postprocess = cal_f1_from_submitLabel(gt, pred_label, eps)

        print(F1_score_test, F1_score_test_postprocess)
elif mode == 'test':
    target_txt_path = ''
    path_list = get_path_list(anno_path)

    if fusion_flag:
        pred_from = np.array(mmcv.load(pred_from_json_path)['class_scores'])
        pred_to = model_fusion(pred_to, pred_from, [1, 7, 11])

        thrs = thrs_fusion(thrs_offcial_l4, thrs_official_l2, [1, 7, 11])
        thrs = np.array(torch.Tensor(thrs).expand(pred_to.shape[0], 12))

        # F1_score_test = cal_f1_from_mmclsInfer(gt, pred_to, thrs, eps)

        pred_to = probability_post_process(8, pred_to, video_list)
            
        pred_label = make_submit_from_mmclsInfer(pred_to, thrs)
        dealed_pred_labels = post_process3(6, pred_label, video_list)
        
        npLabel2rawSubmit(dealed_pred_labels, path_list, target_txt_path)

        # F1_score_test_postprocess = cal_f1_from_submitLabel(gt, dealed_pred_labels, eps)
        # print(F1_score_test, F1_score_test_postprocess)

    else:
        thrs = np.array(torch.Tensor(thrs_offcial_test).expand(pred_to.shape[0], 12))

        pred_to = probability_post_process(8, pred_to, video_list)

        pred_label = make_submit_from_mmclsInfer(pred_to, thrs)
        dealed_pred_labels = post_process3(6, pred_label, video_list)

        npLabel2rawSubmit(dealed_pred_labels, path_list, target_txt_path)

        print('debug')

elif mode == 'generate':
    target_txt_path = '/home/data/lrd/data/abaw/extra_data_txt/test_vote86_postprocess.txt'
    path_list = get_path_list(anno_path)

    if fusion_flag:

        pred_official_json = '/home/data/lrd/mmclassification_custom/abaw_test/test_55.35.json'
        pred_fold1_json = '/home/data/lrd/mmclassification_custom/abaw_test/vote/test_fold1Model.json'
        pred_fold2_json = '/home/data/lrd/mmclassification_custom/abaw_test/vote/test_fold2Model.json'
        pred_fold3_json = '/home/data/lrd/mmclassification_custom/abaw_test/vote/test_fold3Model.json'
        pred_fold4_json = '/home/data/lrd/mmclassification_custom/abaw_test/vote/test_fold4Model.json'

        pred_official = np.array(mmcv.load(pred_official_json)['class_scores'])
        pred_fold1 = np.array(mmcv.load(pred_fold1_json)['class_scores'])
        pred_fold2 = np.array(mmcv.load(pred_fold2_json)['class_scores'])
        pred_fold3 = np.array(mmcv.load(pred_fold3_json)['class_scores'])
        pred_fold4 = np.array(mmcv.load(pred_fold4_json)['class_scores'])

        vote_pred = vote_fusion_pred([pred_official, pred_fold1, pred_fold2, pred_fold3, pred_fold4], 'weighted_mean')

        vote_thrs = vote_fusion_thrs('weighted_mean')
        vote_thrs = np.array(torch.Tensor(vote_thrs).expand(vote_pred.shape[0], 12))

        vote_pred = probability_post_process(8, vote_pred, video_list)
        # tmp
        # F1_score_test = cal_f1_from_mmclsInfer(gt, vote_pred, vote_thrs, eps)


        vote_label = make_submit_from_mmclsInfer(vote_pred, vote_thrs)
        dealed_vote_labels = post_process3(6, vote_label, video_list)

        npLabel2pesudoLabel(dealed_vote_labels, path_list, target_txt_path)

        print('generate fusion')

        # npLabel2rawSubmit(dealed_vote_labels, path_list, target_txt_path)

    else:
        thrs = np.array(torch.Tensor(thrs_offcial_l4).expand(pred_to.shape[0], 12))

        pred_to = probability_post_process(8, pred_to, video_list)

        pred_label = make_submit_from_mmclsInfer(pred_to, thrs)
        dealed_pred_labels = post_process3(6, pred_label, video_list)

        npLabel2pesudoLabel(dealed_pred_labels, path_list, target_txt_path)

        print('debug')

elif mode == 'vote':
    # tmp
    # gt_npy_path = '/home/data/lrd/mmclassification_custom/abaw_gt/Official.npy'
    # gt = np.load(gt_npy_path)
    target_txt_path = '/home/data/lrd/mmclassification_custom/abaw_submit/wo_missing_fig/vote5_postprocess_hp2.txt'
    path_list = get_path_list(anno_path)

    pred_official_json = '/home/data/lrd/mmclassification_custom/abaw_test/test_55.35.json'
    pred_fold1_json = '/home/data/lrd/mmclassification_custom/abaw_test/vote/test_fold1Model.json'
    pred_fold2_json = '/home/data/lrd/mmclassification_custom/abaw_test/vote/test_fold2Model.json'
    pred_fold3_json = '/home/data/lrd/mmclassification_custom/abaw_test/vote/test_fold3Model.json'
    pred_fold4_json = '/home/data/lrd/mmclassification_custom/abaw_test/vote/test_fold4Model.json'

    pred_official = np.array(mmcv.load(pred_official_json)['class_scores'])
    pred_fold1 = np.array(mmcv.load(pred_fold1_json)['class_scores'])
    pred_fold2 = np.array(mmcv.load(pred_fold2_json)['class_scores'])
    pred_fold3 = np.array(mmcv.load(pred_fold3_json)['class_scores'])
    pred_fold4 = np.array(mmcv.load(pred_fold4_json)['class_scores'])

    vote_pred = vote_fusion_pred([pred_official, pred_fold1, pred_fold2, pred_fold3, pred_fold4], 'weighted_mean')

    vote_thrs = vote_fusion_thrs('weighted_mean')
    vote_thrs = np.array(torch.Tensor(vote_thrs).expand(vote_pred.shape[0], 12))

    vote_pred = probability_post_process(8, vote_pred, video_list)
    # tmp
    # F1_score_test = cal_f1_from_mmclsInfer(gt, vote_pred, vote_thrs, eps)


    vote_label = make_submit_from_mmclsInfer(vote_pred, vote_thrs)
    dealed_vote_labels = post_process3(6, vote_label, video_list)

    npLabel2rawSubmit(dealed_vote_labels, path_list, target_txt_path)


