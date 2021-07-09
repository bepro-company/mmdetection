# optimizer
optimizer = dict(type='SGD', lr=0.0012, momentum=0.96, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=16)
