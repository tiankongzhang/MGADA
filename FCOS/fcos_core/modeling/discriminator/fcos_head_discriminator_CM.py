import torch
import torch.nn.functional as F
from torch import nn

from .layer import GradientReversal


class FCOSDiscriminator_CM(nn.Module):
    def __init__(self, cfg, num_convs=2, in_channels=256, grad_reverse_lambda=-1.0, grl_applied_domain='both'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSDiscriminator_CM, self).__init__()
        self.fpn_strides = cfg.MODEL.GENBOX.FPN_STRIDES
        self.layer_levels = {'P3':0, 'P4':0, 'P5':0, 'P6':0, 'P7':0}
        
        self.loss_direct_w = cfg.MODEL.CC.LOSS_DIRECT_W
        self.loss_grl_w = cfg.MODEL.CC.LOSS_GRL_W
        self.samples_thresh = cfg.MODEL.CC.SAMPLES_THRESH

        self.num_class = cfg.MODEL.FCOS.NUM_CLASSES
        self.out_class = cfg.MODEL.FCOS.NUM_CLASSES * 2

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
            in_channels, self.out_class, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.dis_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.loss_direct_f = nn.CrossEntropyLoss()
        self.loss_grl_f = nn.BCELoss()

        assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        self.grl_applied_domain = grl_applied_domain


    def forward(self, feature, target, pred_dict, groundtruth, layer, domain='source'):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'
        level = self.layer_levels[layer]
        
        scores_mx = 0
        layer_index = 0
        
        for layer_idx in pred_dict.keys():
            if layer_index == 0:
                scores_mx = pred_dict[layer_idx]['box_cls'].sigmoid().max()
            else:
                lmx = pred_dict[layer_idx]['box_cls'].sigmoid().max()
                scores_mx = scores_mx.max(lmx)
            layer_index += 1

        
        loss_direct = self.forward_direct(feature, target, pred_dict[layer]['box_cls'], groundtruth, layer, domain, scores_mx)
        loess_grl = self.forward_grl(feature, target, pred_dict[layer]['box_cls'], groundtruth, layer, domain, scores_mx)
        
        loss = self.loss_direct_w * loss_direct + self.loss_grl_w * loess_grl
        return loss
    

    def forward_direct(self, feature, target, pred_cls, groundtruth, layer, domain='source', scores_mx = 1.0):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        x = self.dis_tower(feature)
        x = self.cls_logits(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.num_class, 2).sum(dim=2)

        nb, nc, nh, nw = pred_cls.size()
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().view(-1, nc)
        pred_cls_v, pred_cls_index = pred_cls.sigmoid().max(dim=1)

        fg_mask = (pred_cls_v > scores_mx * self.samples_thresh).long()
        bg_mask = 1 - fg_mask
        pred_cls_index = fg_mask.mul(pred_cls_index) + bg_mask * (self.num_class - 1)

        loss = self.loss_direct_f(x, pred_cls_index)
        return loss
    

    def forward_grl(self, feature, target, pred_cls, groundtruth, layer, domain='source', scores_mx = 1.0):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'
        
        if self.grl_applied_domain == 'both':
            feature = self.grad_reverse(feature)
        elif self.grl_applied_domain == 'target':
            if domain == 'target':
                feature = self.grad_reverse(feature)
                    
        x = self.dis_tower(feature)
        x = self.cls_logits(x)
        x = F.softmax(x.permute(0, 2, 3, 1).contiguous().view(-1, self.num_class, 2), 2).view(-1, self.out_class)
        
        nb, nc, nh, nw = pred_cls.size()
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().view(-1, nc)
        pred_cls_v, pred_cls_index = pred_cls.sigmoid().max(dim=1)

        loss = 0.0 * torch.sum(x)
        for ii in range(nc):
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
