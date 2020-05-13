_base_ = [
    '../_base_/models/fcn_r50.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_220e.py'
]
model = dict(
    pretrained='pretrain_model/resnet50c128_csail-0a46e9a7.pth',
    backbone=dict(stem_channels=128))
