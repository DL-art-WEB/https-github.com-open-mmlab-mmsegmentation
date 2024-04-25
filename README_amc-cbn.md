# README_AMC-CBN

## Dataset
- ADE20K: Download ADE20K dataset from http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip 
- Angiography: 
    - `mmseg/datasets/angiography.py`

## Settings
- logger: change `vis_backends` type to `WandbVisBackend` in `configs/_base_/default_runtime.py`
- wandb logging url: https://wandb.ai/amccbn/mmsegmentation-tool
- Adjust the number of epochs depending on the batch size.

## Installation
- Download pretrained vit from https://github.com/open-mmlab/mmpretrain/tree/master/configs/vision_transformer and move it to `pretrain/`
- Used [vit checkpoint](vit_vit-b16_mln_upernet_8xb2-80k_ade20k-512x512.py) for training vit_vit-b16_mln_upernet_8xb2-80k_ade20k-512x512.py 
- `pip install wandb`

## Train
```
tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_8xb2-80k_ade20k-512x512.py [num_gpus] --work-dir logs/vit-upernet-ade20k
tools/dist_train.sh configs/upernet/upernet_r101_4xb4-160k_ade20k-512x512.py [num_gpus] --work-dir logs/res-101-upernet-ade20k

tools/dist_train.sh configs/upernet/upernet_r101_4xb4-160k_cag-512x512.py [num_gpus] --work-dir logs/res-101-upernet-cag
```

## Transforms
- Additional augmentation functions using albumentation can be found in `mmseg/datasets/transforms/transforms.py`

## Issues
- To create instance of `dataset` in the jupyter notebook, `init_default_scope('mmseg')` must be called.
    - https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/datasets.html#main-interfaces
- After adding a custom dataset class, you need to register the class to `datasets/__init__.py` file.
