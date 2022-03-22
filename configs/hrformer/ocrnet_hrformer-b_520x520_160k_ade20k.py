_base_ = [
    '../_base_/models/ocrnet_hrformer-s.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.1)
model = dict(
    pretrained='pretrain/hrt_base.pth',
    backbone=dict(
        type='HRFormer',
        norm_cfg=norm_cfg,
        norm_eval=False,
        drop_path_rate=0.4,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(2, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='HRFORMERBLOCK',
                window_sizes=(7, 7),
                num_heads=(2, 4),
                mlp_ratios=(4, 4),
                num_blocks=(2, 2),
                num_channels=(78, 156)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='HRFORMERBLOCK',
                window_sizes=(7, 7, 7),
                num_heads=(2, 4, 8),
                mlp_ratios=(4, 4, 4),
                num_blocks=(2, 2, 2),
                num_channels=(78, 156, 312)),
            stage4=dict(
                num_modules=2,
                num_branches=4,
                block='HRFORMERBLOCK',
                window_sizes=(7, 7, 7, 7),
                num_heads=(2, 4, 8, 16),
                mlp_ratios=(4, 4, 4, 4),
                num_blocks=(2, 2, 2, 2),
                num_channels=(78, 156, 312, 624)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[78, 156, 312, 624],
            channels=512,
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            kernel_size=3,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            sampler=dict(type='OHEMPixelSampler', thresh=0.9,
                         min_kept=100000)),
        dict(
            type='OCRHead',
            in_channels=[78, 156, 312, 624],
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            channels=512,
            ocr_channels=256,
            dropout_ratio=-1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            sampler=dict(type='OHEMPixelSampler', thresh=0.9,
                         min_kept=100000)),
    ],
)
# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 4 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (520, 520)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# By default, models are trained on 4 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2, train=dict(pipeline=train_pipeline))
