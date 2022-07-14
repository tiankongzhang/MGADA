import torch
import torch.nn.functional as F
from torch import nn, cat

from detection.layers import ConvTranspose2d


def mask_rcnn_loss(pred_mask_logits, gt_masks, fg_proposals, fg_labels):
    device = pred_mask_logits.device
    total_num_masks = pred_mask_logits.shape[0]
    mask_size = pred_mask_logits.shape[2]
    gt_mask_tensors = []
    for gt_masks_per_image, proposals in zip(gt_masks, fg_proposals):
        if len(gt_masks_per_image) == 0:
            continue

        gt_masks_per_image = gt_masks_per_image.crop_and_resize(proposals, mask_size).to(device=device)
        gt_mask_tensors.append(gt_masks_per_image)

    if len(gt_mask_tensors) == 0:
        return pred_mask_logits.sum() * 0

    gt_mask_tensors = cat(gt_mask_tensors, dim=0)
    pred_mask_logits = pred_mask_logits[torch.arange(total_num_masks), cat(fg_labels, dim=0)]

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_mask_tensors.to(dtype=torch.float32), reduction='mean')
    return mask_loss


class ConvUpSampleMaskHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        num_conv = 4
        conv_dims = 256

        self.convs = []
        for i in range(num_conv):
            conv = nn.Conv2d(in_channels if i == 0 else conv_dims, conv_dims, kernel_size=3, stride=1, padding=1)
            self.add_module('mask_fcn{}'.format(i + 1), conv)
            self.convs.append(conv)

        self.deconv = ConvTranspose2d(conv_dims if len(self.convs) > 0 else in_channels, conv_dims, kernel_size=2, stride=2, padding=0)
        self.predictor = nn.Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.predictor.weight, std=0.001)
        nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x):
        for layer in self.convs:
            x = F.relu(layer(x))
        x = F.relu(self.deconv(x))
        x = self.predictor(x)
        return x
