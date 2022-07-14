import torch
from torch.utils.data import ConcatDataset, DataLoader

from . import collate_fn
from .datasets import *

cityscapes_images_dir = '/data7/lufficc/cityscapes/leftImg8bit'
foggy_cityscapes_images_dir = '/data7/lufficc/cityscapes/leftImg8bit_foggy'

DATASETS = {
    'cityscapes_train': {
        'ann_file': '/data7/lufficc/cityscapes/cityscapes_coco_train.json',
        'root': cityscapes_images_dir,
    },

    'cityscapes_val': {
        'ann_file': '/data7/lufficc/cityscapes/cityscapes_coco_val.json',
        'root': cityscapes_images_dir,
    },

    'cityscapes_test': {
        'ann_file': '/data7/lufficc/cityscapes/cityscapes_coco_test.json',
        'root': cityscapes_images_dir,
    },

    'foggy_cityscapes_train': {
        'ann_file': '/data7/lufficc/cityscapes/foggy_cityscapes_coco_train.json',
        'root': foggy_cityscapes_images_dir,
    },

    'foggy_cityscapes_val': {
        'ann_file': '/data7/lufficc/cityscapes/foggy_cityscapes_coco_val.json',
        'root': foggy_cityscapes_images_dir,
    },

    'foggy_cityscapes_train_0.02': {
        'ann_file': '/data7/lufficc/cityscapes/foggy_cityscapes_coco_train.json',
        'root': foggy_cityscapes_images_dir,
        'betas': 0.02,
    },

    'foggy_cityscapes_val_0.02': {
        'ann_file': '/data7/lufficc/cityscapes/foggy_cityscapes_coco_val.json',
        'root': foggy_cityscapes_images_dir,
        'betas': 0.02,
    },

    'foggy_cityscapes_test': {
        'ann_file': '/data7/lufficc/cityscapes/foggy_cityscapes_coco_test.json',
        'root': foggy_cityscapes_images_dir,
    },
    "coco_2017_train": {
        "ann_file": "/data7/lufficc/coco/annotations/instances_train2017.json",
        "root": "/data7/lufficc/coco/train2017",
    },
    "coco_2017_val": {
        "ann_file": "/data7/lufficc/coco/annotations/instances_val2017.json",
        "root": "/data7/lufficc/coco/val2017",
    },

    'voc_2007_trainval': {
        'root': '/data7/lufficc/voc/VOCdevkit/VOC2007',
        'split': 'trainval',
    },

    'voc_2012_trainval': {
        'root': '/data7/lufficc/voc/VOCdevkit/VOC2012',
        'split': 'trainval',
    },

    'voc_2007_test': {
        'root': '/data7/lufficc/voc/VOCdevkit/VOC2007',
        'split': 'test',
    },

    # -----------watercolor----------
    'watercolor_voc_2012_trainval': {
        'root': '/data7/lufficc/voc/VOCdevkit/VOC2012',
        'split': 'trainval',
    },

    'watercolor_voc_2007_trainval': {
        'root': '/data7/lufficc/voc/VOCdevkit/VOC2007',
        'split': 'trainval',
    },
    'watercolor_train': {
        'root': '/data7/lufficc/cross_domain_detection/watercolor',
        'split': 'train',
    },
    'watercolor_test': {
        'root': '/data7/lufficc/cross_domain_detection/watercolor',
        'split': 'test',
    },

    # -----------clipart----------
    'voc_clipart_train': {
        'root': '/data7/lufficc/cross_domain_detection/clipart',
        'split': 'train',
    },
    'voc_clipart_test': {
        'root': '/data7/lufficc/cross_domain_detection/clipart',
        'split': 'test',
    },
    'voc_clipart_traintest': {
        'root': '/data7/lufficc/cross_domain_detection/clipart',
        'split': 'traintest',
    },

    # -----------sim10k----------
    'sim10k': {
        'root': '/data7/lufficc/cross_domain_detection/sim10k/repro_10k_images/',
        'split': 'all',
    },
    'cityscapes_car_train': {
        'ann_file': '/data7/lufficc/cityscapes/cityscapes_coco_train.json',
        'root': cityscapes_images_dir,
    },

    'cityscapes_car_val': {
        'ann_file': '/data7/lufficc/cityscapes/cityscapes_coco_val.json',
        'root': cityscapes_images_dir,
    },

    'car_city_val': {
        'ann_file': '/data7/lufficc/cityscapes/annotations/instances_car_only_filtered_gtFine_val.json',
        'root': cityscapes_images_dir + '/val',
    },

    '6cats_city_val': {
        'ann_file': '/data7/lufficc/cityscapes/cityscapes_6cats_coco_val.json',
        'root': cityscapes_images_dir,
    },

    'foggy_cityscapes_car_train': {
        'ann_file': '/data7/lufficc/cityscapes/foggy_cityscapes_coco_train.json',
        'root': foggy_cityscapes_images_dir,
    },

    'foggy_cityscapes_car_val': {
        'ann_file': '/data7/lufficc/cityscapes/foggy_cityscapes_coco_val.json',
        'root': foggy_cityscapes_images_dir,
    },

    # -----------kitti----------
    'kitti_train': {
        'root': '/data7/lufficc/cross_domain_detection/kitti/VOC2012/',
        'split': 'train',
    },

    'vkitti': {
        'ann_file': '/data8/lufficc/datasets/VirtualKITTI-InstanceSeg-COCO.json',
        'root': '/data8/lufficc/datasets/VKITTI/vkitti_1.3.1_rgb',
    },

    'SYNTHIA_mask': {
        'ann_file': '/data8/lufficc/datasets/RAND_CITYSCAPES-COCO.json',
        'root': '/data8/lufficc/datasets/SYNTHIA/RAND_CITYSCAPES/RGB',
    },
}


def build_datasets(names, transforms, is_train=True):
    assert len(names) > 0
    datasets = []
    for name in names:
        cfg = DATASETS[name].copy()
        cfg['dataset_name'] = name
        cfg['train'] = is_train
        cfg['transforms'] = transforms
        if 'watercolor' in name:
            dataset = WatercolorDataset(**cfg)
        elif 'cityscapes_car' in name:
            dataset = CityscapeCarDataset(**cfg)
        elif 'sim10k' in name:
            dataset = Sim10kDataset(**cfg)
        elif 'vkitti' in name:
            dataset = VKITTI(**cfg)
        elif 'kitti' in name:
            dataset = KITTIDataset(**cfg)
        elif 'cityscapes' in name:
            dataset = CityscapeDataset(**cfg)
        elif 'coco' in name:
            dataset = MSCOCODataset(**cfg)
        elif 'voc' in name:
            dataset = CustomVocDataset(**cfg)
        elif 'car_city_val' in name:
            dataset = CityscapeDataset(**cfg)
        elif '6cats_city_val' in name:
            dataset = CityscapeDataset(**cfg)
        elif 'SYNTHIA_mask' in name:
            dataset = SYNTHIAMask(**cfg)
        else:
            raise NotImplementedError
        datasets.append(dataset)
    if is_train:
        return datasets if len(datasets) == 1 else [ConcatDataset(datasets)]
    return datasets


def build_data_loaders(names, transforms, is_train=True, distributed=False, batch_size=1, num_workers=8):
    datasets = build_datasets(names, transforms=transforms, is_train=is_train)
    data_loaders = []
    for dataset in datasets:
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        elif is_train:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        if is_train:
            batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)
            loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn)
        else:
            loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=num_workers, collate_fn=collate_fn)

        data_loaders.append(loader)

    if is_train:
        assert len(data_loaders) == 1, 'When training, only support one dataset.'
        return data_loaders[0]
    return data_loaders
