from .dataset import COCODataset


class MSCOCODataset(COCODataset):
    CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def __init__(self, train, **kwargs):
        super().__init__(remove_empty=train, **kwargs)


class VKITTI(COCODataset):
    CLASSES = ('__background__', 'car')

    def __init__(self, train, **kwargs):
        super().__init__(remove_empty=train, **kwargs)


class SYNTHIAMask(COCODataset):
    CLASSES = ('__background__', 'Car', 'Bus', 'Motorcycle', 'Bicycle', 'Person', 'Rider')

    def __init__(self, train, **kwargs):
        super().__init__(remove_empty=train, **kwargs)
