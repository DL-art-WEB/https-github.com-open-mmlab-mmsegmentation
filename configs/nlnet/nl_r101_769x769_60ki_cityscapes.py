_base_ = './nl_r50_769x769_60ki_cityscapes.py'
model = dict(
    pretrained='pretrain_model/resnet101_v1c_trick-e67eebb6.pth',
    backbone=dict(depth=101))
