dataset_type = 'BeproDataset'
data_root = 'data/bepro/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(3072, 1728), keep_ratio=True),
    dict(type='Resize', img_scale=(4096, 1200), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(3600, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            # dict(type='ImageToTensor', keys=['img']),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
       ann_file='/home/bepro/mmdetection/data/bepro/ann_file_stitching.txt',
    #    ann_file='/home/dmitriy.khvan/mmdetection/data/bepro/ann_file.txt',
        img_prefix='',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='/home/bepro/mmdetection/data/bepro/val_file_stitching.txt',
        # ann_file='/home/dmitriy.khvan/mmdetection/data/bepro/val_file.txt',
        img_prefix='',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/home/bepro/mmdetection/data/bepro/test_file_stitching.txt',
        img_prefix='',
        pipeline=test_pipeline))
evaluation = dict(interval=1)
