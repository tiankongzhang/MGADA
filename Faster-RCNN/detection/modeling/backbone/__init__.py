from .resnet import resnet
from .vgg import vgg

BACKBONES = {
    'vgg16': vgg,
    'resnet101': resnet,
}


def build_backbone(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONES[backbone_name](cfg)
    return backbone
