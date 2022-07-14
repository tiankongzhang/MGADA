import math
import torch
import torch.nn.functional as F
from torch import nn

from .loss import make_boxgen_loss_evaluator

from fcos_core.layers import Scale


class BoxGenHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(BoxGenHead, self).__init__()
        # TODO: Implement the sigmoid version first.

        for level in range(cfg.MODEL.GENBOX.NUM):
            bbox_tower = []
            for i in range(cfg.MODEL.GENBOX.NUM_CONVS):
                bbox_tower.append(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )
                bbox_tower.append(nn.GroupNorm(32, in_channels))
                bbox_tower.append(nn.ReLU())

            self.add_module('bbox_tower'+str(level), nn.Sequential(*bbox_tower))
            bbox_pred = [nn.Conv2d(
                in_channels, 4, kernel_size=3, stride=1,
                padding=1
            )]
            self.add_module('bbox_pred'+str(level), nn.Sequential(*bbox_pred))


            # initialization
            bbox_tower = getattr(self, 'bbox_tower'+str(level))
            bbox_pred = getattr(self, 'bbox_pred'+str(level))

            for modules in [bbox_tower, bbox_pred]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)


        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        bbox_reg = []
        for l, feature in enumerate(x):
            bbox_tower = getattr(self, 'bbox_tower'+str(l))
            bbox_pred = getattr(self, 'bbox_pred'+str(l))

            bbox_reg.append(torch.exp(self.scales[l](
                bbox_pred(bbox_tower(feature))
            )))
        return bbox_reg
        
    
class BoxGenModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(BoxGenModule, self).__init__()

        head = BoxGenHead(cfg, in_channels)
        loss_evaluator = make_boxgen_loss_evaluator(cfg)

        self.head = head
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.GENBOX.FPN_STRIDES

    def forward(self, images, features, targets=None, return_maps=False):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_regression = self.head(features)
        locations = self.compute_locations(features)
 
        if self.training:
            #print('level_box---:', self.training)
            return self._forward_train(locations, box_regression, targets, return_maps)
        else:
            return self._forward_test(locations, box_regression, images.image_sizes)

    def _forward_train(self, locations, box_regression, targets, return_maps=False):
        score_maps = {
            "box_regression": box_regression
        }
        losses = {}
        if targets is not None:
            loss_box_reg = self.loss_evaluator(locations, box_regression, targets)
            losses = {
                "loss_reg_box": loss_box_reg
            }
        else:
            losses = {
                "zero_box": 0.0 * sum(0.0 * torch.sum(x) for x in box_regression)
            }

        if return_maps:
            return None, losses, score_maps
        else:
            return None, losses, None

    def _forward_test(self, locations, box_regression, image_sizes):
        score_maps = {
            "box_regression": box_regression
        }
        return None, {}, score_maps

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_boxgen(cfg, in_channels):
    return BoxGenModule(cfg, in_channels)
