import copy
from typing import Tuple, Mapping, Sequence, List

import tensorflow as tf

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _COCO_result(image_id: int,
                 labels: tf.Tensor,
                 bboxes: tf.Tensor,
                 scores: tf.Tensor) -> Sequence[dict]:

    b_h = bboxes[:, 3] - bboxes[:, 1]
    b_w = bboxes[:, 2] - bboxes[:, 0]
    coco_bboxes = tf.stack([bboxes[:, 0], bboxes[:, 1], b_w, b_h])
    coco_bboxes = tf.transpose(coco_bboxes).numpy().tolist()

    labels = labels.numpy().tolist()
    scores = scores.numpy().tolist()

    return [dict(image_id=image_id, category_id=l, bbox=b,score=s) 
            for l, b, s in zip(labels, coco_bboxes, scores)]


def _COCO_gt_annot(image_id: int,
                   annot_id: int,
                   image_shape: Tuple[int, int], 
                   labels: tf.Tensor, 
                   bboxes: tf.Tensor) -> Tuple[dict, Sequence[dict]]:
    
    im_h, im_w = image_shape
    
    b_h = bboxes[:, 3] - bboxes[:, 1]
    b_w = bboxes[:, 2] - bboxes[:, 0]
    areas = tf.reshape(b_h * b_w, [-1])
    areas = areas.numpy().tolist()

    coco_bboxes = tf.stack([bboxes[:, 0], bboxes[:, 1], b_w, b_h])
    coco_bboxes = tf.transpose(coco_bboxes).numpy().tolist()

    labels = labels.numpy().tolist()

    image = dict(id=image_id, height=im_h, width=im_w)

    it = zip(coco_bboxes, areas, labels)
    annotations = [
        dict(id=id_, image_id=image_id, bbox=bbox, iscrowd=0, 
             area=a, category_id=l)
        for id_, (bbox, a, l) in enumerate(it, start=annot_id)]

    return image, annotations
    

def tf_data_to_COCO(ds: tf.data.Dataset,
                    class2idx: Mapping[str, int]) -> COCO:

    gt_coco: dict = dict(images=[], annotations=[])
    image_id = 1
    annot_id = 1

    # Create COCO categories
    categories = [dict(supercategory='instance', id=i, name=n)
                  for n, i in class2idx.items()]
    gt_coco['categories'] = categories

    for image, (labels, bbs) in ds.unbatch():
        h, w = image.shape[0: 2]
        im_annot, annots = _COCO_gt_annot(image_id, annot_id, 
                                          (h, w), labels, bbs)
        gt_coco['annotations'].extend(annots)
        gt_coco['images'].append(im_annot)
        
        annot_id += len(annots)
        image_id += 1

    gtCOCO = COCO()
    gtCOCO.dataset = gt_coco
    gtCOCO.createIndex()

    return gtCOCO


def evaluate(model: tf.keras.Model, 
             dataset: tf.data.Dataset,
             gtCOCO: COCO,
             steps: int,
             print_every: int = 10) -> None:

    results_coco: List[dict] = []
    image_id = 1
    
    for i, (images, _) in enumerate(dataset):
        
        bboxes, categories, scores = model(images, training=False)

        for batch_idx in range(len(bboxes)):
            preds = categories[batch_idx], bboxes[batch_idx], scores[batch_idx]
            pred_labels, pred_boxes, pred_scores = preds

            if pred_labels.shape[0] > 0:
                results = _COCO_result(image_id, 
                                       pred_labels, 
                                       pred_boxes, 
                                       pred_scores)
                results_coco.extend(results)
            
            image_id += 1

        if i % print_every == 0:
            print(f'Validating[{i}/{steps}]...')

    resCOCO = COCO()
    resCOCO.dataset['images'] = gtCOCO.dataset['images']
    resCOCO.dataset['categories'] = copy.deepcopy(gtCOCO.dataset['categories'])

    for i, ann in enumerate(results_coco):
        bb = ann['bbox']
        x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
        if not 'segmentation' in ann:
            ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
        ann['area'] = bb[2]*bb[3]
        ann['id'] = i + 1
        ann['iscrowd'] = 0

    resCOCO.dataset['annotations'] = results_coco
    resCOCO.createIndex()

    coco_eval = COCOeval(gtCOCO, resCOCO, 'bbox')
    coco_eval.params.imgIds = sorted(gtCOCO.getImgIds())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
