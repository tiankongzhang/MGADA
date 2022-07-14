import torch

import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops

from torchvision.ops import boxes as box_ops

from detection.layers import smooth_l1_loss, cat
from .utils import BalancedPositiveNegativeSampler, Matcher, BoxCoder
from .anchor_generator import AnchorGenerator


class RPN(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.cfg = cfg
        # fmt:off
        batch_size          = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        anchor_stride       = cfg.MODEL.RPN.ANCHOR_STRIDE
        anchor_scales       = cfg.MODEL.RPN.ANCHOR_SIZES
        anchor_ratios       = cfg.MODEL.RPN.ASPECT_RATIOS
        num_channels        = cfg.MODEL.RPN.NUM_CHANNELS
        nms_thresh          = cfg.MODEL.RPN.NMS_THRESH
        min_size            = cfg.MODEL.RPN.MIN_SIZE
        num_anchors         = len(anchor_scales) * len(anchor_ratios)
        #iou_th              = cfg.MODEL.RPN.IOU_TH
        # fmt:on
        #self.iou_th = iou_th

        self.pre_nms_top_n = {
            True: cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST,
        }
        self.post_nms_top_n = {
            True: cfg.MODEL.RPN.POST_NMS_TOP_N_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOP_N_TEST,
        }
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        num_channels = in_channels if num_channels is None else num_channels
        self.conv = nn.Conv2d(
            in_channels, num_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(num_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(num_channels, num_anchors * 4, kernel_size=1, stride=1)
        self.anchor_generator = AnchorGenerator(anchor_stride, (anchor_scales,), (anchor_ratios,))
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.matcher = Matcher(high_threshold=0.7, low_threshold=0.3, allow_low_quality_matches=True)
        self.sampler = BalancedPositiveNegativeSampler(batch_size, 0.5)

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, images, features, img_metas, targets=None):
        anchors = self.anchor_generator([features])
        t = F.relu(self.conv(features))
        logits = self.cls_logits(t)
        bbox_reg = self.bbox_pred(t)

        is_target_domain = self.training and targets is None
        with torch.no_grad():
            proposals = self.generate_proposals(anchors, logits, bbox_reg, img_metas, is_target_domain)

        if self.training and targets is not None:
            objectness_loss, box_loss = self.losses(anchors, logits, bbox_reg, img_metas, targets)
            loss = {
                'rpn_cls_loss': objectness_loss,
                'rpn_reg_loss': box_loss,
            }
        else:
            loss = {}

        return proposals, loss, logits

    def generate_proposals(self, anchors, objectness, box_regression, img_metas, is_target_domain=False):
        """
        Args:
            anchors:
            objectness: (N, A, H, W)
            box_regression: (N, A * 4, H, W)
            img_metas:
            is_target_domain:
        Returns:
        """
        pre_nms_top_n = self.pre_nms_top_n[self.training]
        post_nms_top_n = self.post_nms_top_n[self.training]
        if is_target_domain:
            post_nms_top_n = self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        nms_thresh = self.nms_thresh

        device = objectness.device
        N, A, H, W = objectness.shape

        objectness = objectness.permute(0, 2, 3, 1).reshape(N, H * W * A)
        objectness = objectness.sigmoid()

        box_regression = box_regression.permute(0, 2, 3, 1).reshape(N, H * W * A, 4)
        concat_anchors = cat(anchors, dim=0)
        concat_anchors = concat_anchors.reshape(N, A * H * W, 4)

        num_anchors = A * H * W
        pre_nms_top_n = min(pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)
        batch_idx = torch.arange(N, device=device)[:, None]
        box_regression = box_regression[batch_idx, topk_idx]
        concat_anchors = concat_anchors[batch_idx, topk_idx]

        proposals = self.box_coder.decode(
            box_regression.view(-1, 4), concat_anchors.view(-1, 4)
        )
        proposals = proposals.view(N, -1, 4)

        results = []
        for proposal, score, img_meta in zip(proposals, objectness, img_metas):
            img_width, img_height = img_meta['img_shape']
            proposal = box_ops.clip_boxes_to_image(proposal, (img_height, img_width))
            keep = box_ops.remove_small_boxes(proposal, self.min_size)

            proposal = proposal[keep]
            score = score[keep]

            keep = ops.nms(proposal, score, nms_thresh)
            keep = keep[:post_nms_top_n]
            proposal = proposal[keep]
            score = score[keep]

            results.append(proposal)  # (N, 4)
        return results

    def losses(self, anchors, objectness, box_regression, img_metas, targets):
        N, A, H, W = objectness.shape
        labels = []
        regression_targets = []
        real_boxes = []
        for batch_id in range(len(targets)):
            target = targets[batch_id]
            anchors_per_image = anchors[batch_id]
            img_width, img_height = img_metas[batch_id]['img_shape']

            match_quality_matrix = box_ops.box_iou(target['boxes'], anchors_per_image)
            matched_idxs = self.matcher(match_quality_matrix)

            matched_idxs_for_target = matched_idxs.clamp(0)
            
            #print('--00:', target['boxes'].size(), matched_idxs_for_target.size())
            target_boxes = target['boxes'][matched_idxs_for_target]
            real_boxes.append(target_boxes)
            #print('--01:', target_boxes.size())
            labels_per_image = (matched_idxs >= 0).to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1

            straddle_thresh = 0
            visibility = ((anchors_per_image[..., 0] >= -straddle_thresh)
                          & (anchors_per_image[..., 1] >= -straddle_thresh)
                          & (anchors_per_image[..., 2] < img_width + straddle_thresh)
                          & (anchors_per_image[..., 3] < img_height + straddle_thresh))
            labels_per_image[~visibility] = -1

            #print(target_boxes.size(),anchors_per_image.size())
            regression_targets_per_image = self.box_coder.encode(
                target_boxes, anchors_per_image
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        #print('--00:',regression_targets[0].size(), box_regression.size())
        #print('--01:',labels[0].size(), objectness.size())
        sampled_pos_inds, sampled_neg_inds = self.sampler(labels)
        #sampled_pos_inds = cat(sampled_pos_inds, dim=0)
        #sampled_neg_inds = cat(sampled_neg_inds, dim=0)
        #print(cat(sampled_pos_inds, dim=0).size(), cat(sampled_neg_inds, dim=0).size())
        #print(sampled_pos_inds)
        #print(torch.nonzero(cat(sampled_pos_inds, dim=0)).size(), cat(sampled_pos_inds, dim=0).size())

        sampled_pos_inds = torch.nonzero(cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(cat(sampled_neg_inds, dim=0)).squeeze(1)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.permute(0, 2, 3, 1).reshape(-1)
        box_regression = box_regression.permute(0, 2, 3, 1).reshape(-1, 4)

        labels = cat(labels)
        regression_targets = cat(regression_targets)
        real_boxes = cat(real_boxes)
        
        '''
        concat_anchors = cat(anchors, dim=0)
        concat_anchors = concat_anchors.reshape(N, A * H * W, 4)
        
        pred_bbs = self.box_coder.decode(
            box_regression, concat_anchors.view(-1, 4))
        boxes_iou = self.IOU(pred_bbs, real_boxes)
        label_mask = (boxes_iou > self.iou_th).long()
        
        sampled_pos_inds = sampled_pos_inds.mul(label_mask)
        sampled_neg_inds = sampled_neg_inds.mul(label_mask)
        sampled_pos_inds = torch.nonzero(sampled_pos_inds).squeeze(1)
        sampled_neg_inds = torch.nonzero(sampled_neg_inds).squeeze(1)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        '''

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )
        return objectness_loss, box_loss
    
    def IoU(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        g_w_intersect = torch.max(pred_left, target_left) + \
                        torch.max(pred_right, target_right)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                        torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        return gious
