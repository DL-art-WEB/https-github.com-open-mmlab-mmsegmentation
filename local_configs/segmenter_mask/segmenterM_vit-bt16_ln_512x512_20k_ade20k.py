_base_ = [
    '../_base_/models/segmenterM_vit.py',
    '../_base_/datasets/ade20k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

# bt16 means vit-B seg-T patchsize=16
# in the origin config of segmenter, encoder and decoder transformer's parameter are the same
model = dict(
    backbone=dict(drop_path_rate=0.1, final_norm=True),
    decode_head=dict(
        num_classes=150,
        channels=192,
        num_layers=12,
        num_heads=3),
    auxiliary_head=dict(num_classes=150),
    # ensure image size can be divided by the patch size
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'cls_embed': dict(decay_mult=0.)
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

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
