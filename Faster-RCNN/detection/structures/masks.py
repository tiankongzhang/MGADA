import copy

import numpy as np
import pycocotools.mask as mask_utils
import torch


def polygons_to_bitmask(polygons, height, width):
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height (int)
        width (int)
    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    assert len(polygons) > 0, 'COCOAPI does not support empty polygons'
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)

    return mask_utils.decode(rle).astype(np.bool)


def rasterize_polygons_within_box(polygons, box, mask_size):
    """
    Rasterize the polygons into a mask image and
    crop the mask content in the given box.
    The cropped mask is resized to (mask_size, mask_size).

    This function is used when generating training targets for mask head in Mask R-CNN.
    Given original ground-truth masks for an image, new ground-truth mask
    training targets in the size of `mask_size x mask_size`
    must be provided for each predicted box. This function will be called to
    produce such targets.

    Args:
        polygons (list[ndarray[float]]): a list of polygons, which represents an instance.
        box: 4-element numpy array
        mask_size (int):

    Returns:
        Tensor: BoolTensor of shape (mask_size, mask_size)
    """
    # 1. Shift the polygons w.r.t the boxes
    w, h = box[2] - box[0], box[3] - box[1]

    polygons = copy.deepcopy(polygons)
    for p in polygons:
        p[0::2] = p[0::2] - box[0]
        p[1::2] = p[1::2] - box[1]

    # 2. Rescale the polygons to the new box size
    ratio_h = mask_size / max(h, 0.1)
    ratio_w = mask_size / max(w, 0.1)

    if ratio_h == ratio_w:
        for p in polygons:
            p *= ratio_h
    else:
        for p in polygons:
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h

    # 3. Rasterize the polygons with coco api
    mask = polygons_to_bitmask(polygons, mask_size, mask_size)
    mask = torch.from_numpy(mask)
    return mask


class PolygonMasks:
    def __init__(self, polygons):
        """
        Arguments:
            polygons (list[list[Tensor[float]]]): The first
                level of the list correspond to individual instances,
                the second level to all the polygons that compose the
                instance, and the third level to the polygon coordinates.
                The third level Tensor should have the format of
                torch.Tensor([x0, y0, x1, y1, ..., xn, yn]) (n >= 3).
        """
        assert isinstance(polygons, list)

        def _make_array(t):
            # Use float64 for higher precision, because why not?
            # Always put polygons on CPU (self.to is a no-op) since they
            # are supposed to be small tensors.
            # May need to change this assumption if GPU placement becomes useful
            if isinstance(t, torch.Tensor):
                t = t.cpu().numpy()
            return np.asarray(t).astype("float64")

        def process_polygons(polygons_per_instance):
            assert isinstance(polygons_per_instance, list), type(polygons_per_instance)
            # transform the polygon to a tensor
            polygons_per_instance = [_make_array(p) for p in polygons_per_instance]
            for polygon in polygons_per_instance:
                assert len(polygon) % 2 == 0 and len(polygon) >= 6
            return polygons_per_instance

        self.polygons = [process_polygons(polygons_per_instance) for polygons_per_instance in polygons]

    def __getitem__(self, item):
        """
        Support indexing over the instances and return a `PolygonMasks` object.
        `item` can be:

        1. An integer. It will return an object with only one instance.
        2. A slice. It will return an object with the selected instances.
        3. A list[int]. It will return an object with the selected instances,
           corresponding to the indices in the list.
        4. A vector mask of type BoolTensor, whose length is num_instances.
           It will return an object with the instances whose mask is nonzero.
        """
        if isinstance(item, int):
            selected_polygons = [self.polygons[item]]
        elif isinstance(item, slice):
            selected_polygons = self.polygons[item]
        elif isinstance(item, list):
            selected_polygons = [self.polygons[i] for i in item]
        elif isinstance(item, torch.Tensor):
            # Polygons is a list, so we have to move the indices back to CPU.
            if item.dtype == torch.bool:
                assert item.dim() == 1, 'bool tensor as indices should be 1-dim'
                item = item.nonzero().squeeze(1).tolist()
            elif item.dtype in [torch.int32, torch.int64]:
                item = item.tolist()
            else:
                raise ValueError('Unsupported tensor dtype={} for indexing!'.format(item.dtype))
            selected_polygons = [self.polygons[i] for i in item]
        else:
            raise ValueError('Unsupported indexing type {}'.format(type(item)))
        return PolygonMasks(selected_polygons)

    def crop_and_resize(self, boxes, mask_size):
        """
        Crop each mask by the given box, and resize results to (mask_size, mask_size).
        This can be used to prepare training targets for Mask R-CNN.

        Args:
            boxes (Tensor): Nx4 tensor storing the boxes for each mask
            mask_size (int): the size of the rasterized mask.

        Returns:
            Tensor: A bool tensor of shape (N, mask_size, mask_size), where
            N is the number of predicted boxes for this image.
        """
        assert len(boxes) == len(self) and len(boxes) != 0, "boxes len should == mask len, got {} != {}".format(len(boxes), len(self))

        device = boxes.device
        # Put boxes on the CPU, as the polygon representation is not efficient GPU-wise
        # (several small tensors for representing a single instance mask)
        boxes = boxes.to(torch.device('cpu'))

        results = [
            rasterize_polygons_within_box(poly, box.numpy(), mask_size)
            for poly, box in zip(self.polygons, boxes)
        ]
        """
        poly: list[list[float]], the polygons for one instance
        box: a tensor of shape (4,)
        """
        if len(results) == 0:
            return torch.empty(0, mask_size, mask_size, dtype=torch.bool, device=device)
        return torch.stack(results, dim=0).to(device=device)

    def __iter__(self):
        return iter(self.polygons)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.polygons))
        return s

    def __len__(self) -> int:
        return len(self.polygons)
