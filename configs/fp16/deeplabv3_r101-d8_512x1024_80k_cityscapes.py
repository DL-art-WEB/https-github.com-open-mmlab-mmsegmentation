_base_ = '../deeplabv3/deeplabv3_r101-d8_512x1024_80k_cityscapes.py'
# fp16 settings
fp16 = dict(loss_scale=512.)
