import torch
from torch import nn
import math

class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()

class CIOULoss(nn.Module):
    def forward(self, bboxes1, bboxes2, weight=None):
        '''
        w1 = pred[:, 2] - pred[:, 0]
        h1 = pred[:, 3] - pred[:, 1]
        w2 = target[:, 2] - target[:, 0]
        h2 = target[:, 3] - target[:, 1]
        area1 = w1 * h1
        area2 = w2 * h2
        
        center_x1 = (pred[:, 2] + pred[:, 0]) / 2
        center_y1 = (pred[:, 3] + pred[:, 1]) / 2
        center_x2 = (target[:, 2] + target[:, 0]) / 2
        center_y2 = (target[:, 3] + target[:, 1]) / 2

        inter_l = torch.max(center_x1 - w1 / 2,center_x2 - w2 / 2)
        inter_r = torch.min(center_x1 + w1 / 2,center_x2 + w2 / 2)
        inter_t = torch.max(center_y1 - h1 / 2,center_y2 - h2 / 2)
        inter_b = torch.min(center_y1 + h1 / 2,center_y2 + h2 / 2)
        inter_area = torch.clamp((inter_r - inter_l),min=0) * torch.clamp((inter_b - inter_t),min=0)

        c_l = torch.min(center_x1 - w1 / 2,center_x2 - w2 / 2)
        c_r = torch.max(center_x1 + w1 / 2,center_x2 + w2 / 2)
        c_t = torch.min(center_y1 - h1 / 2,center_y2 - h2 / 2)
        c_b = torch.max(center_y1 + h1 / 2,center_y2 + h2 / 2)

        inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
        c_diag = torch.clamp((c_r - c_l),min=0)**2 + torch.clamp((c_b - c_t),min=0)**2

        union = area1+area2-inter_area
        u = (inter_diag) / (c_diag+0.0001)
        iou = inter_area / (union + 0.0001)

        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        with torch.no_grad():
            S = (iou>0.5).float()
            alpha= S*v/(1-iou+v)
        cious = iou - u - alpha * v
        cious = torch.clamp(cious,min=-1.0,max = 1.0)
        #print(cious)

        #losses = -torch.log((cious + 1)/2)
        losses = 1 - cious

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()
        '''
        '''
        pred_cx = (pred[:, 0] + pred[:,2]) / 2
        pred_cy = (pred[:, 1] + pred[:,3]) / 2
        pred_w = pred[:, 2] - pred[:,0]
        pred_h = pred[:, 3] - pred[:,1]
        bboxes1 = torch.stack([pred_cx, pred_cy, pred_w, pred_h],-1)

        target_cx = (target[:, 0] + target[:,2]) / 2
        target_cy = (target[:, 1] + target[:,3]) / 2
        target_w = target[:, 2] - target[:,0]
        target_h = target[:, 3] - target[:,1]
        bboxes2 = torch.stack([target_cx, target_cy, target_w, target_h],-1)
        '''
        
        rows = bboxes1.shape[0]
        cols = bboxes2.shape[0]
        cious = torch.zeros((rows, cols))
        if rows * cols == 0:
            return cious
        exchange = False
        if bboxes1.shape[0] > bboxes2.shape[0]:
            bboxes1, bboxes2 = bboxes2, bboxes1
            cious = torch.zeros((cols, rows))
            exchange = True

        w1 = bboxes1[:, 2] - bboxes1[:, 0]
        h1 = bboxes1[:, 3] - bboxes1[:, 1]
        w2 = bboxes2[:, 2] - bboxes2[:, 0]
        h2 = bboxes2[:, 3] - bboxes2[:, 1]

        area1 = w1 * h1
        area2 = w2 * h2

        center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
        center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
        center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
        center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

        inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
        inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
        out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
        out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

        inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
        inter_area = inter[:, 0] * inter[:, 1]
        inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
        outer = torch.clamp((out_max_xy - out_min_xy), min=0)
        outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
        union = area1+area2-inter_area
        u = (inter_diag) / outer_diag
        iou = inter_area / union
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        with torch.no_grad():
            S = 1 - iou
            alpha = v / (S + v)
        cious = iou - (u + alpha * v)
        cious = torch.clamp(cious,min=-1.0,max = 1.0)
        if exchange:
            cious = cious.T
        return torch.mean(1-cious)
