import pytest
import torch

from mmseg.models.backbones.vit import VisionTransformer
from .utils import check_norm_state


def test_vit_backbone():
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = VisionTransformer()
        model.init_weights(pretrained=0)

    # Test ViT backbone with input size of 224 and patch size of 16
    model = VisionTransformer()
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(4, 3, 224, 224)
    feat = model(imgs)
    assert feat[0].shape == torch.Size([4, 768, 1, 1])
