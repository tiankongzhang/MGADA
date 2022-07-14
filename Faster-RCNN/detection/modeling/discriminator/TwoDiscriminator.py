from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops, models
from torchvision.ops import boxes as box_ops
from .layer import GradientReversal

from detection.layers import FrozenBatchNorm2d, smooth_l1_loss

class VGG16TwoDiscriminator(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        
        self.sampel_weights = cfg.MODEL.ROI_TWO_DIS.DOM.SAMPLE_WEIGHTS
        self.grad_reverse = GradientReversal(cfg.MODEL.ROI_TWO_DIS.DOM.GRL_LAMBDA)
        
        self.extractor = nn.Sequential(
            nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1
                ),
            nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1
                ),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
        )

        self.dis_score = nn.Linear(in_channels, 2)
        nn.init.normal_(self.dis_score.weight, std=0.01)
        nn.init.constant_(self.dis_score.bias, 0)
        
        self.loss_fn = nn.NLLLoss()
        

    def forward(self, dis_features, target, domain='source'):
        dis_features = self.grad_reverse(dis_features)
        dis_features = self.extractor(dis_features)
        dis_features = torch.mean(dis_features, dim=(2, 3))

        dis_features = self.classifier(dis_features)
        dis_logits = self.dis_score(dis_features)
        dis_logits = F.log_softmax(dis_logits, dim=1)
        
        target = torch.full((dis_logits.shape[0],), target, dtype=torch.long, device=dis_logits.device)
        loss = self.loss_fn(dis_logits, target)
        
        if domain=='source':
            loss *= self.sampel_weights
        
        return loss


class ResNetTwoDiscriminator(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        
        resnet = models.resnet.__dict__[cfg.MODEL.BACKBONE.NAME](pretrained=True, norm_layer=FrozenBatchNorm2d)
        self.extractor = resnet.layer4
        del resnet

        in_channels = self.extractor[-1].conv3.out_channels
        
        self.classifiers = nn.Sequential(
            nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1
                )
        )


        self.dis_score = nn.Linear(in_channels, 2)
        nn.init.normal_(self.dis_score.weight, std=0.01)
        nn.init.constant_(self.dis_score.bias, 0)
        
        self.grad_reverse = GradientReversal(cfg.MODEL.ROI_TWO_DIS.DOM.GRL_LAMBDA)
        
        self.loss_fn = nn.NLLLoss()
        self.sampel_weights = cfg.MODEL.ROI_TWO_DIS.DOM.SAMPLE_WEIGHTS
    
    def forward(self, dis_features, target, domain='source'):
        dis_features = self.grad_reverse(dis_features)
        dis_features = self.extractor(dis_features)
        dis_features = self.classifiers(dis_features)

        dis_features = torch.mean(dis_features, dim=(2, 3))
        
        dis_logits = self.dis_score(dis_features)
        dis_logits = F.log_softmax(dis_logits, dim=1)
        
        target = torch.full((dis_logits.shape[0],), target, dtype=torch.long, device=dis_logits.device)
        loss = self.loss_fn(dis_logits, target)
        
        if domain=='source':
            loss *= self.sampel_weights
        
        return loss

class VGG16TwoDiscriminator_cc(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.sampel_weights = cfg.MODEL.ROI_TWO_DIS.CLS.SAMPLE_WEIGHTS
        self.num_classes = cfg.MODEL.ROI_TWO_DIS.CLS.NUM_CLASSES
        self.out_classes = cfg.MODEL.ROI_TWO_DIS.CLS.NUM_CLASSES * 2
        
        self.grad_reverse = GradientReversal(cfg.MODEL.ROI_TWO_DIS.CLS.GRL_LAMBDA)
        self.extractor = nn.Sequential(
            nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1
                ),
            nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1
                ),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
        )
    

        self.dis_score = nn.Linear(in_channels, self.num_classes*2)
        nn.init.normal_(self.dis_score.weight, std=0.01)
        nn.init.constant_(self.dis_score.bias, 0)
        
        self.loss_direct_f = nn.CrossEntropyLoss()
        self.loss_grl_f = nn.BCELoss()
        
        self.loss_direct_w = cfg.MODEL.ROI_TWO_DIS.CLS.LOSS_DIRECT_W
        self.loss_grl_w = cfg.MODEL.ROI_TWO_DIS.CLS.LOSS_GRL_W
        self.samples_thresh = cfg.MODEL.ROI_TWO_DIS.CLS.SAMPLES_THRESH

    def forward(self, dis_features, pred_cls, target, domain='source'):
        pred_cls = F.softmax(pred_cls, 1)
        score_mx = pred_cls.max()
        loss_direct = self.forward_direct(dis_features, pred_cls, target, domain, score_mx)
        loss_grl = self.forward_grl(dis_features, pred_cls, target, domain, score_mx)
        
        loss = self.loss_direct_w * loss_direct + self.loss_grl_w * loss_grl
        
        if domain=='source':
            loss *= self.sampel_weights
        
        return loss
    
    def forward_direct(self, dis_features, pred_cls, target, domain='source', score_mx=1.0):
        ex_features = self.extractor(dis_features)
        ex_features = torch.mean(ex_features, dim=(2, 3))
        ex_features = self.classifier(ex_features)
        dis_logits = self.dis_score(ex_features)
        dis_logits = dis_logits.view(-1, self.num_classes, 2).sum(dim=2)
        
        pred_cls_v, pred_cls_index = pred_cls.max(dim=1)
        
        th_mask = pred_cls_v > score_mx * self.samples_thresh
        
        pred_cls_v_part = pred_cls_v[th_mask]
        pred_cls_index_part = pred_cls_index[th_mask]
        dis_logits_part = dis_logits[th_mask,:]
        
        fg_mask = pred_cls_index_part > 0
        bg_mask = pred_cls_index_part == 0
        
        pred_cls_v_part_fg = pred_cls_v_part[fg_mask]
        pred_cls_index_part_fg = pred_cls_index_part[fg_mask]
        dis_logits_part_fg = dis_logits_part[fg_mask,:]
        
        pred_cls_v_part_bg = pred_cls_v_part[bg_mask]
        pred_cls_index_part_bg = pred_cls_index_part[bg_mask]
        dis_logits_part_bg = dis_logits_part[bg_mask,:]
        
        if domain == 'source':
            smaples_pos_num = 30
        elif domain == 'target':
            smaples_pos_num = 300
            
        if pred_cls_v_part_fg.size(0)>smaples_pos_num:
            vs, ins = pred_cls_v_part_fg.sort(descending=True)
            ins = ins[0:smaples_pos_num]
            pred_cls_index_part_fg = pred_cls_index_part_fg[ins]
            dis_logits_part_fg = dis_logits_part_fg[ins,:]
        
        smaples_neg_num = 2 * pred_cls_index_part_fg.size(0)
        if pred_cls_v_part_bg.size(0)>smaples_neg_num:
            vs, ins = pred_cls_v_part_bg.sort(descending=True)
            if smaples_neg_num==0:
               smaples_neg_num = 1
            ins = ins[0:smaples_neg_num]
            pred_cls_index_part_bg = pred_cls_index_part_bg[ins]
            dis_logits_part_bg = dis_logits_part_bg[ins,:]
        
        if pred_cls_index_part_fg.size(0)>0 and pred_cls_index_part_bg.size(0)>0:
             pred_cls_index_part_m = torch.cat((pred_cls_index_part_fg, pred_cls_index_part_bg),0)
             dis_logits_part_m = torch.cat((dis_logits_part_fg, dis_logits_part_bg),0)
        elif pred_cls_index_part_fg.size(0)>0:
             pred_cls_index_part_m = pred_cls_index_part_fg
             dis_logits_part_m = dis_logits_part_fg
        elif pred_cls_index_part_bg.size(0)>0:
             pred_cls_index_part_m = pred_cls_index_part_bg
             dis_logits_part_m = dis_logits_part_bg
        
        loss = self.loss_direct_f(dis_logits_part_m, pred_cls_index_part_m)
        
        return loss
    
    def forward_grl(self, dis_features, pred_cls, target, domain='source', score_mx=1.0):
        dis_features = self.grad_reverse(dis_features)
        
        ex_features = self.extractor(dis_features)
        ex_features = torch.mean(ex_features, dim=(2, 3))
        ex_features = self.classifier(ex_features)
        dis_logits = self.dis_score(ex_features)

        dis_logits = F.softmax(dis_logits.view(-1, self.num_classes, 2), 2).view(-1, self.out_classes)
        
        pred_cls_v, pred_cls_index = pred_cls.max(dim=1)
        
        loss = 0.0 * torch.sum(dis_logits)
        
        sample_pos_num = 0
        for ii in range(1, self.num_classes):
            cls_idxs = pred_cls_index == ii
            pred_cls_idx = pred_cls_v[cls_idxs]
            if pred_cls_idx.size(0) == 0:
                continue

            dx_cls_idx = dis_logits[cls_idxs,:]
            
            cls_idxs = pred_cls_idx > self.samples_thresh * score_mx
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


class ResNetTwoDiscriminator_cc(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.sampel_weights = cfg.MODEL.ROI_TWO_DIS.CLS.SAMPLE_WEIGHTS
        self.num_classes = cfg.MODEL.ROI_TWO_DIS.CLS.NUM_CLASSES
        self.out_classes = cfg.MODEL.ROI_TWO_DIS.CLS.NUM_CLASSES * 2
        
        resnet = models.resnet.__dict__[cfg.MODEL.BACKBONE.NAME](pretrained=True, norm_layer=FrozenBatchNorm2d)
        self.extractor = resnet.layer4
        del resnet
        
        in_channels = self.extractor[-1].conv3.out_channels
        self.classifiers = nn.Sequential(
            nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1
                )
        )


        self.dis_score = nn.Linear(in_channels, self.out_classes)
        nn.init.normal_(self.dis_score.weight, std=0.01)
        for l in [self.dis_score]:
            nn.init.constant_(l.bias, 0)
        
        self.grad_reverse = GradientReversal(cfg.MODEL.ROI_TWO_DIS.CLS.GRL_LAMBDA)
        
        self.loss_direct_f = nn.CrossEntropyLoss()
        self.loss_grl_f = nn.BCELoss()
        
        self.loss_direct_w = cfg.MODEL.ROI_TWO_DIS.CLS.LOSS_DIRECT_W
        self.loss_grl_w = cfg.MODEL.ROI_TWO_DIS.CLS.LOSS_GRL_W
        self.samples_thresh = cfg.MODEL.ROI_TWO_DIS.CLS.SAMPLES_THRESH
    
    def forward(self, dis_features, pred_cls, target, domain='source'):
        pred_cls = F.softmax(pred_cls, 1)
        score_mx = pred_cls.max()
        loss_direct = self.forward_direct(dis_features, pred_cls, target, domain, score_mx)
        loss_grl = self.forward_grl(dis_features, pred_cls, target, domain, score_mx)
        
        loss = self.loss_direct_w * loss_direct + self.loss_grl_w * loss_grl
        
        if domain=='source':
           loss *= self.sampel_weights
        
        return loss
    
    def forward_direct(self, dis_features, pred_cls, target, domain='source', score_mx=1.0):
        ex_features = self.extractor(dis_features)
        ex_features = self.classifiers(ex_features)

        ex_features = torch.mean(ex_features, dim=(2, 3))
        dis_logits = self.dis_score(ex_features)
        dis_logits = dis_logits.view(-1, self.num_classes, 2).sum(dim=2)
        
        pred_cls_v, pred_cls_index = pred_cls.max(dim=1)
        
        th_mask = pred_cls_v > score_mx * self.samples_thresh
        
        pred_cls_v_part = pred_cls_v[th_mask]
        pred_cls_index_part = pred_cls_index[th_mask]
        dis_logits_part = dis_logits[th_mask,:]
        
        loss = self.loss_direct_f(dis_logits_part, pred_cls_index_part)
        
        return loss
    
    def forward_grl(self, dis_features, pred_cls, target, domain='source', score_mx=1.0):
        dis_features = self.grad_reverse(dis_features)
        
        ex_features = self.extractor(dis_features)
        ex_features = self.classifiers(ex_features)
        ex_features = torch.mean(ex_features, dim=(2, 3))
        dis_logits = self.dis_score(ex_features)

        dis_logits = F.softmax(dis_logits.view(-1, self.num_classes, 2), 2).view(-1, self.out_classes)
        
        pred_cls_v, pred_cls_index = pred_cls.max(dim=1)
        
        loss = 0.0 * torch.sum(dis_logits)
        for ii in range(1, self.num_classes):
            cls_idxs = pred_cls_index == ii
            pred_cls_idx = pred_cls_v[cls_idxs]
            if pred_cls_idx.size(0) == 0:
                continue

            dx_cls_idx = dis_logits[cls_idxs,:]
            
            cls_idxs = pred_cls_idx > self.samples_thresh * score_mx
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
