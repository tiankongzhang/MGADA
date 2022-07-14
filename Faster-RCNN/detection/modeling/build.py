from .faster_rcnn import FasterRCNN


def build_detectors(cfg):
    model = FasterRCNN(cfg)
    return model
