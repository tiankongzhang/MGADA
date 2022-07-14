from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops, models
from torchvision.ops import boxes as box_ops

from .layer import GradientReversal

class OneFCOSDiscriminator(nn.Module):
    def __init__(self, cfg, in_channels=256,  grl_applied_domain='both'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(OneFCOSDiscriminator, self).__init__()
        num_convs = cfg.MODEL.ROI_ONE_DIS.DOM.NUM_CONVS
        grad_reverse_lambda = cfg.MODEL.ROI_ONE_DIS.DOM.GRL_LAMBDA
        
        dis_tower = []
        for i in range(num_convs):
            dis_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            dis_tower.append(nn.GroupNorm(32, in_channels))
            dis_tower.append(nn.ReLU())

        self.add_module('dis_tower', nn.Sequential(*dis_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.dis_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        self.grl_applied_domain = grl_applied_domain
        


    def forward(self, feature, target, domain='source'):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'
        
        if self.grl_applied_domain == 'both':
            feature = self.grad_reverse(feature)
        elif self.grl_applied_domain == 'target':
            if domain == 'target':
                feature = self.grad_reverse(feature)
        
        x = self.dis_tower(feature)
        x = self.cls_logits(x)

        target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
        loss = self.loss_fn(x, target)

        return loss


class OneFCOSDiscriminator_cc(nn.Module):
    def __init__(self, cfg, in_channels=256, grl_applied_domain='both'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(OneFCOSDiscriminator_cc, self).__init__()
        num_convs = cfg.MODEL.ROI_ONE_DIS.CLS.NUM_CONVS
        grad_reverse_lambda = cfg.MODEL.ROI_ONE_DIS.CLS.GRL_LAMBDA
        
        self.loss_direct_w = cfg.MODEL.ROI_ONE_DIS.CLS.LOSS_DIRECT_W
        self.loss_grl_w = cfg.MODEL.ROI_ONE_DIS.CLS.LOSS_GRL_W
        self.samples_thresh = cfg.MODEL.ROI_ONE_DIS.CLS.SAMPLES_THRESH

        self.num_classes = cfg.MODEL.ROI_ONE_DIS.CLS.NUM_CLASSES
        self.out_classes = cfg.MODEL.ROI_ONE_DIS.CLS.NUM_CLASSES * 2

        dis_tower = []
        for i in range(num_convs):
            dis_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            dis_tower.append(nn.GroupNorm(32, in_channels))
            dis_tower.append(nn.ReLU())

        self.add_module('dis_tower', nn.Sequential(*dis_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, self.out_classes, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.dis_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.loss_direct_f = nn.CrossEntropyLoss()#
        self.loss_grl_f = nn.BCELoss()#

        assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        self.grl_applied_domain = grl_applied_domain


    def forward(self, feature, target, pred_dict, groundtruth, domain='source'):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        pred_dict = pred_dict.sigmoid()
        scores_mx = pred_dict.max()
        
        loss_direct = self.forward_direct(feature, target, pred_dict, groundtruth, domain, scores_mx)
        
        loess_grl = self.forward_grl(feature, target, pred_dict, groundtruth, domain, scores_mx)
        
        loss = self.loss_direct_w * loss_direct + self.loss_grl_w * loess_grl
        
        return loss
    

    def forward_direct(self, feature, target, pred_cls, groundtruth, domain='source', scores_mx = 1.0):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        x = self.dis_tower(feature)
        x = self.cls_logits(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes, 2).sum(dim=2)


        nb, nc, nh, nw = pred_cls.size()
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().view(-1, nc)
        pred_cls_v, pred_cls_index = pred_cls.max(dim=1)

        gt_mask = (pred_cls_v > scores_mx * self.samples_thresh).long()

        loss = self.loss_direct_f(x, gt_mask)
        return loss
    

    def forward_grl(self, feature, target, pred_cls, groundtruth, domain='source', scores_mx = 1.0):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        if self.grl_applied_domain == 'both':
            feature = self.grad_reverse(feature)
        elif self.grl_applied_domain == 'target':
            if domain == 'target':
                feature = self.grad_reverse(feature)
        x = self.dis_tower(feature)
        x = self.cls_logits(x)
        x = F.softmax(x.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes, 2), 2).view(-1, self.out_classes)
        
        nb, nc, nh, nw = pred_cls.size()
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().view(-1, nc)
        pred_cls_v, pred_cls_index = pred_cls.max(dim=1)
        pred_cls_index = (pred_cls_v > scores_mx * self.samples_thresh).long()

        loss = 0.0 * torch.sum(x)
        for ii in range(1, self.num_classes):
            cls_idxs = pred_cls_index == ii
            pred_cls_idx = pred_cls_v[cls_idxs]
            if pred_cls_idx.size(0) == 0:
                continue

            dx_cls_idx = x[cls_idxs,:]
            
            cls_idxs = pred_cls_idx > self.samples_thresh * scores_mx
            pred_cls_idx = pred_cls_idx[cls_idxs]
            if pred_cls_idx.size(0) == 0:
                continue
            if domain == 'target':
                dx_cls_idx = dx_cls_idx[cls_idxs,ii*2]
            elif domain == 'source':
                dx_cls_idx = dx_cls_idx[cls_idxs,ii*2+1]

            target_idx = torch.full(dx_cls_idx.shape, 1.0, dtype=torch.float, device=dx_cls_idx.device)
            loss += self.loss_grl_f(dx_cls_idx, target_idx)
                
        return loss

