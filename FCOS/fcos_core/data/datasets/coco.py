# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.segmentation_mask import SegmentationMask
from fcos_core.structures.keypoint import PersonKeypoints
import math
import numpy.random as npr

import cv2
from PIL import Image
import numpy as np
 
min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None, is_sample=False, domain=0):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        self.is_sample = is_sample
        self.domain = domain

        # filter images without detection annotations
        #self.is_train = is_train
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.ntransforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)
        scale_p = npr.randint(0,10) / 10.0
        if scale_p < 0.4:
            scale_p = 1.0
        else:
            scale_p = 0.0
        scale_mark = npr.randint(0,2)*2 - 1
        scale_mark = math.pow(npr.randint(2,6), scale_mark * scale_p)
        if not self.is_sample:
            scale_p = 0.0
        #print(not self.is_sample)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        
        ##
        im_w = img.size[0] * scale_mark
        im_h = img.size[1] * scale_mark
        im_cx = im_w / 2
        im_cy = im_h / 2

        crop_im_ex = img.size[0] / 2 - im_cx
        crop_im_ey = img.size[1] / 2 - im_cy

        rl_num_boxes = 0
        bboxes_s = []
        gt_classes_s = []
        
        if scale_p > 0:
            for idx_b in range(len(boxes)):
                bb = boxes[idx_b] * 1
                clsl = classes[idx_b]

                bb[2] = bb[0] + bb[2]
                bb[3] = bb[1] + bb[3]
                if self.domain == 0:
                    if float(bb[2]) - float(bb[0]) > 0 and float(bb[3]) - float(bb[1]) > 0:
                        bb[0] = int(bb[0] * scale_mark + crop_im_ex)
                        bb[1] = int(bb[1] * scale_mark + crop_im_ey)
                        bb[2] = int(bb[2] * scale_mark + crop_im_ex)
                        bb[3] = int(bb[3] * scale_mark + crop_im_ey)

                        if float(bb[3]) - float(bb[1]) < 16 or float(bb[2]) - float(bb[0]) < 16 or \
                            float(bb[0]) < 0 or float(bb[1]) < 0 or bb[2] > img.size[0] or bb[3] > img.size[1]:
                            continue
                        rl_num_boxes += 1
                        bboxes_s.append(bb)
                        gt_classes_s.append(clsl)
                else:
                     bb[0] = int(bb[0] * scale_mark + crop_im_ex)
                     bb[1] = int(bb[1] * scale_mark + crop_im_ey)
                     bb[2] = int(bb[2] * scale_mark + crop_im_ex)
                     bb[3] = int(bb[3] * scale_mark + crop_im_ey)

                     rl_num_boxes += 1
                     bboxes_s.append(bb)
                     gt_classes_s.append(clsl)
                     
        if rl_num_boxes > 1:
            center = (img.size[0] // 2, img.size[1] // 2)
            scale_mat = cv2.getRotationMatrix2D(center, 0, scale_mark)
            imgv = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
            padding = np.mean(imgv, axis=(0, 1))

            imgv = cv2.warpAffine(imgv, scale_mat, (img.size[0], img.size[1]),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=padding)
            img = Image.fromarray(cv2.cvtColor(imgv,cv2.COLOR_BGR2RGB))
            boxes = bboxes_s
            classes = gt_classes_s
        
            
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        classes = torch.tensor(classes)
        
        if rl_num_boxes > 1:
            target = BoxList(boxes, img.size, mode="xyxy")
        else:
            target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        
        target.add_field("labels", classes)

        # masks = [obj["segmentation"] for obj in anno]
        # masks = SegmentationMask(masks, img.size, mode='poly')
        # target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)
        #print(target)

        if self.ntransforms is not None:
            img, target = self.ntransforms(img, target)
        

        #return imgt, cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR), idx,
        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
