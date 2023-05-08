_base_ = './van-b2_pre1k_upernet_4xb2-160k_ade20k-512x512.py'

model = dict(
    backbone=dict(
        depths=[2, 2, 4, 2],
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/van_b1.pth')))
