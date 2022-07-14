import  torch

def box_union(box1,box2,x1y1x2y2=True):
    """
    :param bbox1:
    :param bbox2:
    :return: a union b
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    box1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    box2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    intersect = (torch.min(b2_x2,b1_x2) - torch.max(b1_x1,b2_x1)) * (torch.min(b2_y2,b1_y2) - torch.max(b1_y1,b2_y1))
    union_item = (box1_area + box2_area) - intersect + 1e-16
    return union_item

def box_c(box1,box2,x1y1x2y2=True):
    """
    :param bbox1:
    :param bbox2:
    :return: the smallest convex object c
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    xmin = torch.min(b1_x1,b2_x1)
    ymin = torch.min(b1_y1,b2_y1)
    xmax = torch.max(b1_x2,b2_x2)
    ymax = torch.max(b1_y2,b2_y2)

    return (xmin,ymin,xmax,ymax)

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 , min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 , min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 ) * (b1_y2 - b1_y1 )
    b2_area = (b2_x2 - b2_x1 ) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def giou(bbox1,bbox2, x1y1x2y2=True):
    iou_item  = bbox_iou(bbox1,bbox2,x1y1x2y2)
    boxc_item = box_c(bbox1,bbox2,x1y1x2y2)
    boxc_area = (boxc_item[2]-boxc_item[0])*(boxc_item[3]-boxc_item[1])+ 1e-7

    if boxc_area.eq(0).sum() > 0 :
        print("8888888888888888888888888888888888888888888888888888")
        return iou_item
    u = box_union(bbox1,bbox2,x1y1x2y2)
    giou_item = iou_item - (boxc_area - u)/boxc_area
    return giou_item


if __name__ == "__main__":

    print("aaaaa")

    box1 = torch.Tensor([[2,  2,  2,  2]])
    box2 = torch.Tensor([[4,  4, 2,  2]])

    m = giou(box1,box2,x1y1x2y2=False)
    print(m)
