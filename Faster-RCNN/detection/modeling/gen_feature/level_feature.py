import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

#simple global feature
class getTwoStageFeature(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(getTwoStageFeature, self).__init__()
        # TODO: Implement the sigmoid version first.
        self.zero_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, \
            stride=1, bias=False)
        )
        

        self.conv_two = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
            )
        
        self.conv_three = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        
        
    def forward(self, xs):
        #zero stage
        znx = self.zero_conv(xs)
        
        #one stage
        onx_s = self.conv_two(xs)
        onx = F.interpolate(onx_s, size=[xs.size(-2), xs.size(-1)], mode="bilinear")
        
        tnx_s = self.conv_three(xs)
        tnx = F.interpolate(tnx_s, size=[xs.size(-2), xs.size(-1)], mode="bilinear")
            
        return znx, onx, tnx



class getGlobal(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(getGlobal, self).__init__()
        # TODO: Implement the sigmoid version first.
        levels = len(cfg.MODEL.GENFEATURE.FPN_STRIDES)
        self.fpn_strides = cfg.MODEL.GENFEATURE.FPN_STRIDES
        
        self.msf = getTwoStageFeature(cfg, in_channels)
        
        self.in_channels = in_channels
        
        out_channels_new = in_channels // 5
        out_channels_old = in_channels - out_channels_new
        
        self.upsample_new = nn.Sequential(
            nn.Conv2d(
                    in_channels,
                    out_channels_new,
                    kernel_size=1,
                    stride=1
                )
        )
        
        self.upsample_old = nn.Sequential(
            nn.Conv2d(
                    in_channels,
                    out_channels_old,
                    kernel_size=1,
                    stride=1
                )
        )
        
        
    def forward(self, x):
        #two stage
        znx, onx, tnx = self.msf(x)
        nx = torch.cat((znx, onx, tnx), 1)
        return nx
    
    def merge_feature(self, features, proposals):
        nb, nc, nh, nw = features.size()
        features = features.view(nb, nc, -1)
        proposals_im = torch.cat(proposals, dim=0) * self.fpn_strides[0]
        
        proposals_im[:,0] = proposals_im[:,0].clamp(min=0, max=nw-1)
        proposals_im[:,2] = proposals_im[:,2].clamp(min=0, max=nw-1)
        proposals_im[:,1] = proposals_im[:,1].clamp(min=0, max=nh-1)
        proposals_im[:,3] = proposals_im[:,3].clamp(min=0, max=nh-1)
        
        proposals_im_cx = (proposals_im[:, 0] + proposals_im[:, 2]) /2
        proposals_im_cy = (proposals_im[:, 1] + proposals_im[:, 3]) /2
        proposals_im_cw = proposals_im[:, 2] - proposals_im[:, 0]
        proposals_im_ch = proposals_im[:, 3] - proposals_im[:, 1]

        proposals_im_cw = proposals_im_cw.clamp(min=0, max=nw-1)
        proposals_im_ch = proposals_im_ch.clamp(min=0, max=nh-1)
        
        wh = torch.sqrt(proposals_im_cw.mul(proposals_im_ch))
        
        ##one kernel area
        bz = 3 #
        proposals_im[:,2] = proposals_im[:,2] - proposals_im[:,0]
        proposals_im[:,3] = proposals_im[:,3] - proposals_im[:,1]
        proposals_im[:,0] = 0
        proposals_im[:,1] = 0
        
        proposals_one = self.generate_conv_box(proposals_im, 1, [bz,bz])
        proposals_two = self.generate_conv_box(proposals_im, 2, [bz,bz])
        proposals_three = self.generate_conv_box(proposals_im, 4, [bz,bz])
        
        ious_one = self.IoU(proposals_im, proposals_one)
        ious_two = self.IoU(proposals_im, proposals_two)
        ious_three = self.IoU(proposals_im, proposals_three)
        
        iou_weights = torch.stack((ious_one, ious_two, ious_three), dim=1)
        iou_weights = (iou_weights - iou_weights.max(dim=1)[0].view(nb, 1)) / (iou_weights.max(dim=1)[0].view(nb, 1) + 1e-5)
        
        #train
        iou_weights = F.softmax(4.0 * iou_weights, dim=1)
        
        #test
        #iou_weights = F.softmax(9 * iou_weights, dim=1) #
        
        #zero
        lv = 0
        hv = self.in_channels
        zero_ft = features[:, lv:hv, :]
        
        #one
        lv = self.in_channels
        hv = self.in_channels * 2
        one_ft = features[:, lv:hv, :]
        
        #two
        lv = self.in_channels * 2
        hv = self.in_channels * 3
        two_ft = features[:, lv:hv, :]
        
        gl_ft = torch.stack((zero_ft, one_ft, two_ft), dim=1)
        gl_ft = gl_ft.mul(iou_weights.view(nb, -1, 1, 1)).sum(dim=1)
        
        gl_ft = gl_ft.view(nb, -1, nh, nw)
        return gl_ft
        
    def fusion(self, ofeatures, nfeatures):
        lofeatures = self.upsample_old(ofeatures)
        lnfeatures = self.upsample_new(nfeatures)
        features = torch.cat((lofeatures, lnfeatures), dim=1)
        return features
    
    def getBox(self, proposals, stride, padding):
        proposals[:,0] = proposals[:,0] - padding[0]
        proposals[:,2] = proposals[:,2] - padding[1]
        proposals[:,1] = proposals[:,1] + padding[0]
        proposals[:,3] = proposals[:,3] + padding[1]
        
        proposals_cx = (proposals[:, 0] + proposals[:, 2]) /2
        proposals_cy = (proposals[:, 1] + proposals[:, 3]) /2
        proposals_cw = (proposals[:, 2] - proposals[:, 0]) * stride
        proposals_ch = (proposals[:, 3] - proposals[:, 1]) * stride
        
        proposals[:,0] = proposals_cx - proposals_cw // 2
        proposals[:,1] = proposals_cy - proposals_ch // 2
        proposals[:,2] = proposals_cx + proposals_cw // 2
        proposals[:,3] = proposals_cy + proposals_ch // 2
        
        return proposals
    
    def generate_conv_box(self, box_regression, strides, bbs):
        conv_bbs = torch.zeros_like(box_regression)
        conv_bbs[:, 0] += 0 * strides
        conv_bbs[:, 2] += (bbs[1] // 2 + box_regression[:,2]) * strides
        conv_bbs[:, 1] += 0 * strides
        conv_bbs[:, 3] += (bbs[0] // 2 + box_regression[:,3]) * strides

        return conv_bbs
    
    def IoU(self, proposals_im, proposals_ch):
        
        pred_x1 = proposals_im[:, 0]
        pred_y1 = proposals_im[:, 1]
        pred_x2 = proposals_im[:, 2]
        pred_y2 = proposals_im[:, 3]
        
        pred_x1 = pred_x1 - 1
        pred_x2 = pred_x2 + 1
        
        pred_y1 = pred_y1 - 1
        pred_y2 = pred_y2 + 1
           
        
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_y2 = torch.max(pred_y1, pred_y2)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        target_x1 = proposals_ch[:, 0]
        target_y1 = proposals_ch[:, 1]
        target_x2 = proposals_ch[:, 2]
        target_y2 = proposals_ch[:, 3]
            
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        x1_intersect = torch.max(pred_x1, target_x1)
        y1_intersect = torch.max(pred_y1, target_y1)
        x2_intersect = torch.min(pred_x2, target_x2)
        y2_intersect = torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size()).to(proposals_im)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

        x1_enclosing = torch.min(pred_x1, target_x1)
        y1_enclosing = torch.min(pred_y1, target_y1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        y2_enclosing = torch.max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        
        return ious
        

def build_featuregen(cfg, in_channels):
    return getGlobal(cfg, in_channels)
