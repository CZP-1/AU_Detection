# add colorjitter
# add mixup
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='IResNet',
        layers=[3, 13, 30, 3], 
        with_pooling_fc=False,
        with_pooling_fc_out=True,
        use_anfl=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/data/lrd/mmclassification_custom/pretrained_ckp/backbone_addprefix.pth',
            prefix='backbone',
        ),
        ),
    neck=dict(type='Anfl_Neck',
              in_channels=512, 
              num_classes=12, 
              neighbor_num=4, 
              metric='dots', 
              use_lanet=True,
              p=1.0,
              use_csra=False, lam=1.0, use_gap=True, num_heads=8,
              with_backbone_logit=True,
              use_self_attn_fusion=True,
              encoder_layers_num=4, 
              multi_heads_num=1,
              init_cfg=None
              ),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=12,
        use_centerloss=False,
        use_circleloss=True,
        use_mixup=False,
        use_bceloss=True,
        use_csra=False,
        use_anfl=True,

        use_anfl_with_backbone=True,
        use_fc_fusion_weight=False,
        use_attn_fusion_weight=True,
        use_anfl_with_csra=False,
        in_channels=512,
        circleloss_thr_pos=0.9,
        circleloss_thr_neg=0.1,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        )

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(type='ColorJitter', 
    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.3)    
]


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=112),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
        dict(
        type='Albu',
        transforms=albu_train_transforms,
        keymap={
            'img': 'image',
            # 'gt_bboxes': 'bboxes'
        },
        ),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(128, -1)),
    dict(type='CenterCrop', crop_size=112),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

dataset_type = 'CustomMultiLabelDataset'
classes =['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='/home/data/lrd/data/abaw/images_train_val/train',
        ann_file='/home/data/lrd/data/abaw/annotations_tiny/train_clean_bkp.txt',

        # data_prefix='/home/data/lrd/data/abaw/images_train_val/train_val',
        # ann_file='/home/data/lrd/data/abaw/annotations_k_folds/fold4/train.txt',

        # data_prefix='/home/data/lrd/data/abaw/images_train_val/train',
        # ann_file='/home/data/lrd/data/abaw/annotations/train_tiny.txt',
        pipeline=train_pipeline,
        classes=classes
        ),
    val=dict(
        type=dataset_type,

        # data_prefix='/home/data/lrd/data/abaw/images_train_val/train_val',
        # ann_file='/home/data/lrd/data/abaw/annotations_k_folds/fold4/val.txt',
        
        # data_prefix='/home/data/lrd/data/abaw/images_train_val/val',
        # ann_file='/home/data/lrd/data/abaw/annotations/val_tiny.txt',

        data_prefix='/home/data/lrd/data/abaw/images_train_val/val',
        ann_file='/home/data/lrd/data/abaw/annotations/val.txt',
        pipeline=test_pipeline,
        classes=classes
        ),
    test=dict(
        type=dataset_type,
        
        # data_prefix='/home/data/lrd/data/abaw/images_train_val/train_val',
        # ann_file='/home/data/lrd/data/abaw/annotations_k_folds/fold4/val.txt',

        data_prefix='/home/data/lrd/data/abaw/images_train_val/val',
        ann_file='/home/data/lrd/data/abaw/annotations/val.txt',
        
        # data_prefix='/home/data/lrd/data/abaw/images_train_val/val',
        # ann_file='/home/data/lrd/data/abaw/annotations/val_tiny.txt',

        pipeline=test_pipeline,
        classes=classes
        ))
evaluation = dict(save_best='auto', interval=1, metric=[ 'ACF1_theory_max', 'ACF1_max',
                                                        'aus_maxF1_thr','ACF1s', 'aus_maxF1', 'PCF1s'], 
                  metric_options={'thrs': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                                           0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]})

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# lr_config = dict(policy='step', step=[30, 60, 90])
# runner = dict(type='EpochBasedRunner', max_epochs=100)

lr_config = dict(policy='step', step=[4, 6, 8])
runner = dict(type='EpochBasedRunner', max_epochs=15)


checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None

fp16 = dict(loss_scale="dynamic")
work_dir = '/home/data/lrd/mmclassification_custom/work_dirs/abaw/final/r100_ms1v3_pretrain_womixup_Usecircleloss_4x64_15e_newStep_newThr_anfl_fc_lanet_mad1.0_selfattn_fusion_weight_hp1_layernum4_head1'
workflow = [('train', 1), ('val', 1)]