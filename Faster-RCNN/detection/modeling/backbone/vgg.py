from torchvision import models


def vgg(cfg, pretrained=True):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = models.vgg.__dict__[backbone_name](pretrained=pretrained)
    backbone = backbone.features[:-1]
    backbone.out_channels = 512

    for layer in range(10):
        for param in backbone[layer].parameters():
            param.requires_grad = False

    return backbone
