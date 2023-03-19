import torch
import numpy as np
import mmcv

mm_infer_res_path = '/home/data/lrd/mmclassification_custom/abaw_test/55.06.json'
fake_txt_path = '/home/data/lrd/data/abaw/extra_data_txt/affect_fake.txt'
target_txt_path = '/home/data/lrd/data/abaw/extra_data_txt/affect_pseudo_labels_thr1.txt'

with open(fake_txt_path, 'r') as f1:
    fake_txt = f1.readlines()
mm_infer_res = mmcv.load(mm_infer_res_path)['class_scores']

# thr = np.array([0.55, 0.25, 0.65, 0.45, 0.6, 0.6, 0.45, 0.45,
#                 0.2, 0.2, 0.55, 0.3])
thr = np.array([0.8, 0.25, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
                0.2, 0.2, 0.8, 0.5])


pseudo_labels = []
for item in mm_infer_res:
    pseudo_label = np.array(['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'])

    item = np.array(item)
    flag = item >= thr
    idx = np.where(flag)
    
    pseudo_label[idx] = '1'
    str_pseudo_label = ','.join(pseudo_label)
    pseudo_labels.append(str_pseudo_label)

final_txt = []
empty_cnt = 0
for image_path, pseudo_label in zip(fake_txt, pseudo_labels):
    # if pseudo_label == '0,0,0,0,0,0,0,0,0,0,0,0':
    #     empty_cnt += 1
    #     continue
    # else:
    #     anno = image_path.split(' ')[0] + ' ' + pseudo_label + '\n'
    #     final_txt.append(anno)
    anno = image_path.split(' ')[0] + ' ' + pseudo_label + '\n'
    final_txt.append(anno)
print(empty_cnt)

with open(target_txt_path, 'w') as f2:
    f2.writelines(final_txt)


