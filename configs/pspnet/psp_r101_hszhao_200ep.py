_base_ = './psp_r50_tv_200ep.py'
model = dict(
    pretrained='pretrain_model/resnet101c128_hszhao-f9120436.pth',
    backbone=dict(depth=101, deep_stem=True, base_channels=128))
