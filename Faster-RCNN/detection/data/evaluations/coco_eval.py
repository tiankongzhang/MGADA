import copy
import json
import os
from datetime import datetime


def coco_evaluation(dataset, predictions, output_dir, iteration=None):
    """
    Args:
        dataset: Dataset
        predictions: dict, key is image_id, value is (boxes, scores, labels)
        output_dir:
        iteration:
    Returns: metrics
    """
    assert len(dataset) == len(predictions)
    metrics = {}
    coco_results = []
    for image_id in predictions:
        det = predictions[image_id]
        boxes, scores, labels = det['boxes'], det['scores'], det['labels']
        for k, box in enumerate(boxes):
            result = {
                "image_id": image_id,
                "category_id": labels[k],
                "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # to xywh format
                "score": scores[k],
            }
            if 'masks' in det:
                result['segmentation'] = det['masks'][k]

            coco_results.append(result)

    if len(coco_results) == 0:
        return metrics

    os.makedirs(output_dir, exist_ok=True)
    iou_type = 'segm'

    json_result_file = os.path.join(output_dir, iou_type + ".json")
    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)
    from pycocotools.cocoeval import COCOeval
    coco_gt = copy.deepcopy(dataset.coco)
    coco_dt = coco_gt.loadRes(json_result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    result_strings = []
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    for i, key in enumerate(keys):
        metrics[key] = coco_eval.stats[i]
        result_strings.append('{:<10}: {}'.format(key, round(coco_eval.stats[i], 3)))

    if iteration is not None:
        result_path = os.path.join(output_dir, 'coco_result_{:05d}.txt'.format(iteration))
    else:
        result_path = os.path.join(output_dir, 'coco_result_{}.txt'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    with open(result_path, "w") as f:
        f.write('\n'.join(result_strings))
    return metrics
