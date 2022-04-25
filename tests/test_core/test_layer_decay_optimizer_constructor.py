# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.core.optimizers.layer_decay_optimizer_constructor import (
    LayerDecayOptimizerConstructor, LearningRateDecayOptimizerConstructor)

base_lr = 1
decay_rate = 2
base_wd = 0.05
weight_decay = 0.05

stage_wise_gt_lst = [{
    'weight_decay': 0.0,
    'lr_scale': 128
}, {
    'weight_decay': 0.0,
    'lr_scale': 1
}, {
    'weight_decay': 0.05,
    'lr_scale': 64
}, {
    'weight_decay': 0.0,
    'lr_scale': 64
}, {
    'weight_decay': 0.05,
    'lr_scale': 32
}, {
    'weight_decay': 0.0,
    'lr_scale': 32
}, {
    'weight_decay': 0.05,
    'lr_scale': 16
}, {
    'weight_decay': 0.0,
    'lr_scale': 16
}, {
    'weight_decay': 0.05,
    'lr_scale': 8
}, {
    'weight_decay': 0.0,
    'lr_scale': 8
}, {
    'weight_decay': 0.05,
    'lr_scale': 128
}, {
    'weight_decay': 0.05,
    'lr_scale': 1
}]

layer_wise_gt_lst = [{
    'weight_decay': 0.0,
    'lr_scale': 128
}, {
    'weight_decay': 0.0,
    'lr_scale': 1
}, {
    'weight_decay': 0.05,
    'lr_scale': 64
}, {
    'weight_decay': 0.0,
    'lr_scale': 64
}, {
    'weight_decay': 0.05,
    'lr_scale': 32
}, {
    'weight_decay': 0.0,
    'lr_scale': 32
}, {
    'weight_decay': 0.05,
    'lr_scale': 16
}, {
    'weight_decay': 0.0,
    'lr_scale': 16
}, {
    'weight_decay': 0.05,
    'lr_scale': 2
}, {
    'weight_decay': 0.0,
    'lr_scale': 2
}, {
    'weight_decay': 0.05,
    'lr_scale': 128
}, {
    'weight_decay': 0.05,
    'lr_scale': 1
}]

layer_wise_wd_lr = [{
    'weight_decay': 0.0,
    'lr_scale': 16
}, {
    'weight_decay': 0.05,
    'lr_scale': 8
}, {
    'weight_decay': 0.0,
    'lr_scale': 8
}, {
    'weight_decay': 0.05,
    'lr_scale': 4
}, {
    'weight_decay': 0.0,
    'lr_scale': 4
}, {
    'weight_decay': 0.05,
    'lr_scale': 2
}, {
    'weight_decay': 0.0,
    'lr_scale': 2
}]


class ConvNeXtExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(ConvModule(3, 4, kernel_size=1, bias=True))
            self.stages.append(stage)
        self.norm0 = nn.BatchNorm2d(2)

        # add some variables to meet unit test coverate rate
        self.cls_token = nn.Parameter(torch.ones(1))
        self.mask_token = nn.Parameter(torch.ones(1))
        self.pos_embed = nn.Parameter(torch.ones(1))
        self.stem_norm = nn.Parameter(torch.ones(1))
        self.downsample_norm0 = nn.BatchNorm2d(2)
        self.downsample_norm1 = nn.BatchNorm2d(2)
        self.downsample_norm2 = nn.BatchNorm2d(2)
        self.lin = nn.Parameter(torch.ones(1))
        self.lin.requires_grad = False
        self.downsample_layers = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1, bias=True))
            self.downsample_layers.append(stage)

        self.decode_head = nn.Conv2d(2, 2, kernel_size=1, groups=2)


class PseudoDataParallel1(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = ConvNeXtExampleModel()

    def forward(self, x):
        return x


class BEiTExampleModel(nn.Module):

    def __init__(self, depth):
        super().__init__()
        # add some variables to meet unit test coverate rate
        self.cls_token = nn.Parameter(torch.ones(1))
        self.patch_embed = nn.Parameter(torch.ones(1))
        self.layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.Conv2d(3, 3, 1)
            self.layers.append(layer)


class PseudoDataParallel2(nn.Module):

    def __init__(self, depth):
        super().__init__()
        self.backbone = BEiTExampleModel(depth)

    def forward(self, x):
        return x


class ViTExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = nn.ModuleList()
        self.backbone.cls_token = nn.Parameter(torch.ones(1))
        self.backbone.patch_embed = nn.Parameter(torch.ones(1))


def check_convnext_adamw_optimizer(optimizer, gt_lst):
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['weight_decay'] == base_wd
    param_groups = optimizer.param_groups
    assert len(param_groups) == 12
    for i, param_dict in enumerate(param_groups):
        assert param_dict['weight_decay'] == gt_lst[i]['weight_decay']
        assert param_dict['lr_scale'] == gt_lst[i]['lr_scale']
        assert param_dict['lr_scale'] == param_dict['lr']


def check_beit_adamw_optimizer(optimizer, gt_lst):
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == 1
    assert optimizer.defaults['weight_decay'] == 0.05
    param_groups = optimizer.param_groups
    # 1 layer (cls_token and patch_embed) + 3 layers * 2 (w, b) = 9 layers
    assert len(param_groups) == 7
    for i, param_dict in enumerate(param_groups):
        assert param_dict['weight_decay'] == gt_lst[i]['weight_decay']
        assert param_dict['lr_scale'] == gt_lst[i]['lr_scale']
        assert param_dict['lr_scale'] == param_dict['lr']


def test_learning_rate_decay_optimizer_constructor():

    # paramwise_cfg with ConvNeXtExampleModel
    model = PseudoDataParallel1()
    optimizer_cfg = dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05)
    stagewise_paramwise_cfg = dict(
        decay_rate=decay_rate, decay_type='stage_wise', num_layers=6)
    optim_constructor = LearningRateDecayOptimizerConstructor(
        optimizer_cfg, stagewise_paramwise_cfg)
    optimizer = optim_constructor(model)
    check_convnext_adamw_optimizer(optimizer, stage_wise_gt_lst)

    layerwise_paramwise_cfg = dict(
        decay_rate=decay_rate, decay_type='layer_wise', num_layers=6)
    optim_constructor = LearningRateDecayOptimizerConstructor(
        optimizer_cfg, layerwise_paramwise_cfg)
    optimizer = optim_constructor(model)
    check_convnext_adamw_optimizer(optimizer, layer_wise_gt_lst)

    layerwise_paramwise_cfg = dict(
        decay_rate=decay_rate, decay_type='layer_wise', num_layers=3)
    model = PseudoDataParallel2(depth=3)
    optim_constructor = LearningRateDecayOptimizerConstructor(
        optimizer_cfg, layerwise_paramwise_cfg)
    optimizer = optim_constructor(model)
    check_beit_adamw_optimizer(optimizer, layer_wise_wd_lr)

    with pytest.raises(NotImplementedError):
        model = ViTExampleModel()
        optim_constructor = LearningRateDecayOptimizerConstructor(
            optimizer_cfg, layerwise_paramwise_cfg)
        optimizer = optim_constructor(model)

    with pytest.raises(NotImplementedError):
        model = ViTExampleModel()
        optim_constructor = LearningRateDecayOptimizerConstructor(
            optimizer_cfg, stagewise_paramwise_cfg)
        optimizer = optim_constructor(model)


def test_beit_layer_decay_optimizer_constructor():

    # paramwise_cfg with BEiTExampleModel
    model = PseudoDataParallel2(depth=3)
    optimizer_cfg = dict(
        type='AdamW', lr=1, betas=(0.9, 0.999), weight_decay=0.05)
    paramwise_cfg = dict(layer_decay_rate=2, num_layers=3)
    optim_constructor = LayerDecayOptimizerConstructor(optimizer_cfg,
                                                       paramwise_cfg)
    optimizer = optim_constructor(model)
    check_beit_adamw_optimizer(optimizer, layer_wise_wd_lr)
