_base_ = [
    '../_base_/models/pp_mobile.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
checkpoint = './models/pp_mobile_tiny.pth'
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size, test_cfg=dict(size_divisor=32))
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        type='StrideFormer',
        mobileV3_cfg=[
            # k t c, s
            [[3, 16, 16, True, 'ReLU', 1], [3, 64, 32, False, 'ReLU', 2],
             [3, 48, 24, False, 'ReLU', 1]],  # cfg1
            [[5, 96, 32, True, 'hardswish', 2],
             [5, 96, 32, True, 'hardswish', 1]],  # cfg2
            [[5, 160, 64, True, 'hardswish', 2],
             [5, 160, 64, True, 'hardswish', 1]],  # cfg3
            [[3, 384, 128, True, 'hardswish', 2],
             [3, 384, 128, True, 'hardswish', 1]],  # cfg4
        ],
        channels=[16, 24, 32, 64, 128],
        depths=[2, 2],
        embed_dims=[64, 128],
        num_heads=4,
        inj_type='AAM',
        out_feat_chs=[32, 64, 128],
        act_cfg=dict(type='ReLU6'),
    ),
    decode_head=dict(
        num_classes=150,
        in_channels=256,
        use_dw=True,
        act_cfg=dict(type='ReLU'),
        upsample='intepolate'),
)
