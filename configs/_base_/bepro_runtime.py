checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = '/home/bepro/mmdetection/checkpoint/gfl_r50_bepro_12e_stitch.pth'
load_from = ''
resume_from = None
workflow = [('train', 1), ('val', 1)]
