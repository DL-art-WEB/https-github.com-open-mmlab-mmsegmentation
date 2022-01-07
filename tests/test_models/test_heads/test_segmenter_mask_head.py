# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmseg.models.decode_heads import SegmenterMaskTransformerHead
from .utils import _conv_has_norm, to_cuda


def test_uper_head():
    head = SegmenterMaskTransformerHead(
        in_channels=192,
        channels=192,
        num_classes=150,
        num_layers=2,
        num_heads=3,
        embed_dims=192,
        dropout_ratio=0.0)
    assert _conv_has_norm(head, sync_bn=True)

    inputs = [torch.randn(1, 192, 32, 32)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 32, 32)
