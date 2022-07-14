from collections import OrderedDict

import torch
from torchvision import models
from detection.layers import FrozenBatchNorm2d


class ResNet(torch.nn.Sequential):
    def __init__(self, resnet):
        model = OrderedDict([
            (
                'stem', torch.nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool,
                )
            ),
            ('layer1', resnet.layer1),
            ('layer2', resnet.layer2),
            ('layer3', resnet.layer3)
        ])
        super().__init__(model)


def resnet(cfg, pretrained=True):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = models.resnet.__dict__[backbone_name](pretrained=pretrained, norm_layer=FrozenBatchNorm2d)
    backbone = ResNet(backbone)
    backbone.out_channels = 1024

    for param in backbone.stem.parameters():
        param.requires_grad = False

    return backbone
