_base_ = '../deeplabv3/deeplabv3_r50-d8_512x1024_80k_cityscapes.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
# fp16 settings
fp16 = dict(loss_scale=512.)
