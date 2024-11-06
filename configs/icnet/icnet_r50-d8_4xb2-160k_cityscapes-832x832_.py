_base_ = [
    '../_base_/models/icnet_r50-d8.py',
    '../_base_/datasets/ucsd_half.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (720, 1280)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
