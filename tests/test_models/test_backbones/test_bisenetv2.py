# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule

from mmseg.models.backbones import BiSeNetV2


def test_bisenetv2_backbone():
    # Test BiSeNetV2 Standard Forward
    model = BiSeNetV2()
    model.init_weights()
    model.train()
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 512, 1024)
    feat = model(imgs)

    assert len(feat) == 5
    # output for segment Head
    assert feat[0].shape == torch.Size([batch_size, 128, 64, 128])
    # for auxiliary head 1
    assert feat[1].shape == torch.Size([batch_size, 16, 128, 256])
    # for auxiliary head 2
    assert feat[2].shape == torch.Size([batch_size, 32, 64, 128])
    # for auxiliary head 3
    assert feat[3].shape == torch.Size([batch_size, 64, 32, 64])
    # for auxiliary head 4
    assert feat[4].shape == torch.Size([batch_size, 128, 16, 32])

    # Test input with rare shape
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 527, 952)
    feat = model(imgs)
    assert len(feat) == 5

    # Add unittests for modules in bisenetv2
    backbone = BiSeNetV2()
    assert isinstance(backbone.detail.detail_branch[0][0], ConvModule)
    assert backbone.semantic.stage1.pool.stride == 2
    assert isinstance(backbone.bga.conv, ConvModule)
