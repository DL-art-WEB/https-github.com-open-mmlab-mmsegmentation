# model settings

custom_imports = dict(imports=['projects.pixel_contrast_cross_entropy_loss'])
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='HRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144)))),
    decode_head=dict(
        type='ContrastHead',
        in_channels=[18, 36, 72, 144],
        channels=sum([18, 36, 72, 144]),
        num_classes=19,
        in_index=(0, 1, 2, 3),
        input_transform='resize_concat',
        proj_n=256,
        proj_mode='convmlp',
        drop_p=0.1,
        dropout_ratio=-1,
        norm_cfg=norm_cfg,
        align_corners=False,
        seg_head=dict(
                     type='FCNHead',
                     in_channels=[18, 36, 72, 144],
                     in_index=(0, 1, 2, 3),
                     channels=sum([18, 36, 72, 144]),
                     input_transform='resize_concat',
                     kernel_size=1,
                     num_convs=1,
                     concat_input=False,
                     dropout_ratio=-1,
                     num_classes=19,
                     norm_cfg=norm_cfg,
                     align_corners=False),
        loss_decode=[
            dict(
                type='PixelContrastCrossEntropyLoss',
                base_temperature=0.07,
                temperature=0.1,
                ignore_index=255,
                max_samples=1024,
                max_views=100,
                loss_weight=0.1),
            dict(type='CrossEntropyLoss', loss_weight=1.0)
        ]),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
