_base_ = './fcn_r50_60ki_cityscapes.py'
model = dict(
    pretrained='pretrain_model/resnet101_v1c_trick-fde66a20.pth',
    backbone=dict(depth=101))
