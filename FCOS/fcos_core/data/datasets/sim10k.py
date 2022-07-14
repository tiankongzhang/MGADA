import os

import torch
import torch.utils.data
from PIL import Image
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import math
import numpy.random as npr

import cv2
import numpy as np


from fcos_core.structures.bounding_box import BoxList


class Sim10kDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "car",
    )

    def __init__(self, data_dir, split, use_difficult=False, transforms=None, is_sample=False, domain=0):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms
        self.is_sample = is_sample
        self.domain = domain

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = Sim10kDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")
        anno = self.get_groundtruth(index)
        boxes = anno["boxes"]
        classes = anno["labels"]
        diffs = anno["difficult"]

        scale_p = npr.randint(0,10) / 10.0
        if scale_p < 0.4:
            scale_p = 1.0
        else:
            scale_p = 0.0
        

        scale_mark = npr.randint(0,2)*2 - 1
        scale_mark = math.pow(npr.randint(2,6), scale_mark * scale_p)
        if not self.is_sample:
            scale_p = 0.0

        im_w = img.size[0] * scale_mark
        im_h = img.size[1] * scale_mark
        im_cx = im_w / 2
        im_cy = im_h / 2

        crop_im_ex = img.size[0] / 2 - im_cx
        crop_im_ey = img.size[1] / 2 - im_cy

        rl_num_boxes = 0
        bboxes_s = []
        gt_classes_s = []
        gt_diff_s = []
        
        
        if scale_p > 0:
            for idx_b in range(len(boxes)):
                bb = boxes[idx_b] * 1
                clsl = classes[idx_b]
                dffl = diffs[idx_b]

                if float(bb[2]) - float(bb[0]) > 0 and float(bb[3]) - float(bb[1]) > 0:
                    #bb = (bb.float() * scale_mark).int()

                    bb[0] = int(bb[0] * scale_mark + crop_im_ex)
                    bb[1] = int(bb[1] * scale_mark + crop_im_ey)
                    bb[2] = int(bb[2] * scale_mark + crop_im_ex)
                    bb[3] = int(bb[3] * scale_mark + crop_im_ey)
                    
                    if self.domain == 0:
                        if float(bb[3]) - float(bb[1]) < 16 or float(bb[2]) - float(bb[0]) < 16 or \
                            float(bb[0]) < 0 or float(bb[1]) < 0 or bb[2] > img.size[0] or bb[3] > img.size[1]:
                            continue
                            
                    rl_num_boxes += 1
                    bboxes_s.append(bb)
                    gt_classes_s.append(clsl)
                    gt_diff_s.append(dffl)

        if rl_num_boxes > 1:
            center = (img.size[0] // 2, img.size[1] // 2)
            scale_mat = cv2.getRotationMatrix2D(center, 0, scale_mark)
            imgv = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
            padding = np.mean(imgv, axis=(0, 1))

            imgv = cv2.warpAffine(imgv, scale_mat, (img.size[0], img.size[1]),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=padding)
            img = Image.fromarray(cv2.cvtColor(imgv,cv2.COLOR_BGR2RGB))
            
            boxes = torch.stack(bboxes_s)
            classes = torch.stack(gt_classes_s)
            diffs = torch.stack(gt_diff_s)
            
            
        #print(boxes, classes, diffs)
        height, width = anno["im_info"]
        target = BoxList(boxes, (width, height), mode="xyxy")
        target.add_field("labels", classes)
        target.add_field("difficult", diffs)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)
        
        return anno

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1
        
        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            # ignore if not car
            if not name == "car":
                continue
                
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text, 
                bb.find("ymin").text, 
                bb.find("xmax").text, 
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return Sim10kDataset.CLASSES[class_id]
