from collections import defaultdict

import numpy as np
from .dataset import COCODataset


class CityscapeDatasetMixin:
    def filter_ids(self, betas=None):
        if betas is not None:
            if not isinstance(betas, (list, tuple)):
                betas = (betas,)
            img_ids = []
            for img_id in self.ids:
                img_info = self.coco.loadImgs(img_id)[0]
                file_name = img_info['file_name']
                for beta in betas:
                    if '_beta_{}.png'.format(beta) in file_name:
                        img_ids.append(img_id)
            return img_ids
        else:
            return self.ids


class CityscapeDataset(COCODataset, CityscapeDatasetMixin):
    CLASSES = ('__background__', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')

    def __init__(self, train, betas=None, **kwargs):
        super().__init__(remove_empty=train, **kwargs)
        #print('---:', train)
        self.ids = self.filter_ids(betas=betas)
        
        if train:
            self.is_sample = True
        else:
            self.is_sample = False
        


class CityscapeCarDataset(COCODataset, CityscapeDatasetMixin):
    CLASSES = ('__background__', 'car')

    def __init__(self, train, betas=None, **kwargs):
        super().__init__(remove_empty=train, **kwargs)
        self.ids = self.filter_ids(betas=betas)
        coco = self.coco
        car_id = coco.getCatIds(catNms='car')[0]
        #print('---city:',train)
        self.is_sample = False
        if train:
            self.is_sample = train
            img_ids = []
            origin_size = len(self.ids)
            for img_id in self.ids:
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                anns = [obj for obj in anns if obj["iscrowd"] == 0]
                labels = [obj["category_id"] for obj in anns if obj["category_id"] == car_id]
                if len(labels) > 0:
                    img_ids.append(img_id)
            self.ids = img_ids
            print('({})Only images containing car are kept, from {} to {}'.format(self.dataset_name, origin_size, len(self.ids)))

        self.cat2label = {
            1: 1
        }
        self.label2cat = {
            v: k for k, v in self.cat2label.items()
        }
        self.car_id = car_id

    def get_annotations_by_image_id(self, img_id):
        coco = self.coco
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        anns = [obj for obj in anns if obj["iscrowd"] == 0]

        boxes = []
        labels = []
        for obj in anns:
            x, y, w, h = obj["bbox"]
            box = [x, y, x + w - 1, y + h - 1]
            if obj["category_id"] == self.car_id:
                boxes.append(box)
                labels.append(1)
        boxes = np.array(boxes).reshape((-1, 4))
        labels = np.array(labels).reshape((-1,))

        return {'img_info': img_info, 'boxes': boxes, 'labels': labels}
