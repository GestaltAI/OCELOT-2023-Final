checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_base_p16_384_20220308-96dfe169.pth'
backbone_norm_cfg = dict(type='LN', eps=1e-06, requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512)),
    pretrained=
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_base_p16_384_20220308-96dfe169.pth',
    backbone=dict(
        type='VisionTransformer',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        final_norm=True,
        norm_cfg=dict(type='LN', eps=1e-06, requires_grad=True),
        with_cls_token=True,
        interpolate_mode='bicubic',
        init_cfg=None),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=768,
        channels=768,
        num_classes=3,
        num_layers=2,
        num_heads=12,
        embed_dims=768,
        dropout_ratio=0.0,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                loss_weight=1.0,
                class_weight=[1.0, 1.0, 1.0],
                avg_non_ignore=True)
        ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.9, min_kept=100000)),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(480, 480)))
dataset_type = 'OZELOTDataset'
data_root = '/mnt/d/Datasets/OCELOT_Data/MMSegTissue_False'
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type': 'Resize',
            'scale_factor': 1,
            'keep_ratio': True
        }],
                    [{
                        'type': 'RandomFlip',
                        'prob': 0.0,
                        'direction': 'horizontal'
                    }, {
                        'type': 'RandomFlip',
                        'prob': 1.0,
                        'direction': 'horizontal'
                    }, {
                        'type': 'RandomFlip',
                        'prob': 0.0,
                        'direction': 'vertical'
                    }, {
                        'type': 'RandomFlip',
                        'prob': 1.0,
                        'direction': 'vertical'
                    }], [{
                        'type': 'LoadAnnotations'
                    }]])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='OZELOTDataset',
        data_root='/mnt/d/Datasets/OCELOT_Data/MMSegTissue_False',
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='RandomFlip', direction='vertical', prob=0.5),
            dict(type='RandomRotate', degree=(-45, 45), prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs')
        ],
        ann_file='Folds/fold_1/train.txt'))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='OZELOTDataset',
        data_root='/mnt/d/Datasets/OCELOT_Data/MMSegTissue_False',
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ],
        ann_file='Folds/fold_1/valid.txt'))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='OZELOTDataset',
        data_root='/mnt/d/Datasets/OCELOT_Data/MMSegTissue_False',
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ],
        ann_file='Folds/fold_1/valid.txt'))
val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU'], prefix='MMSeg'),
    dict(
        type='F1_OCELOTSegmentationScoreMetric',
        csv_annotation_folder=
        '/mnt/d/Datasets/OCELOT_Data/ocelot2023_v0.1.2/annotations/train/cell',
        metadata_path=
        '/mnt/d/Datasets/OCELOT_Data/ocelot2023_v0.1.2/metadata.json')
]
test_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU'], prefix='MMSeg'),
    dict(
        type='F1_OCELOTSegmentationScoreMetric',
        csv_annotation_folder=
        '/mnt/d/Datasets/OCELOT_Data/ocelot2023_v0.1.2/annotations/train/cell',
        metadata_path=
        '/mnt/d/Datasets/OCELOT_Data/ocelot2023_v0.1.2/metadata.json')
]
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='WandbVisBackend')],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = 'models/segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706-bc533b08.pth'
resume = False
tta_model = dict(type='SegTTAModel')
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005),
    clip_grad=None,
    loss_scale='dynamic',
    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=0.1))))
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0001,
        power=0.9,
        begin=0,
        end=4000,
        by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=4000, val_interval=50)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_best=['MMSeg/mAcc', 'OCELOT/mF1'],
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True))
amp = True
experiment_name = 'segmenter_IU_BS2'
randomness = dict(seed=0)
gpu_ids = range(0, 1)
device = 'cuda'
work_dir = 'logs/tutorial'
