_base_ = [
    '../_base_/models/lraspp_m-v3-d8.py', '../_base_/datasets/ucsd_half-256x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_320k.py'
]
crop_size = (256, 512)
data_preprocessor = dict(size=crop_size)
# Re-config the data sampler.
model = dict(data_preprocessor=data_preprocessor)
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

runner = dict(type='IterBasedRunner', max_iters=320000)
