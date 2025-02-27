from typing import Mapping

import tensorflow as tf

from . import coco
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCOmAPCallback(tf.keras.callbacks.Callback):

    def __init__(self,
                 validation_data: tf.data.Dataset,
                 class2idx: Mapping[str, int], 
                 validate_every: int = 1,
                 print_freq: int = 10) -> None:
        self.validation_data = validation_data
        self.gtCOCO = coco.tf_data_to_COCO(validation_data, class2idx)

        self.class2idx = class2idx
        self.validate_every = validate_every
        self.print_freq = print_freq

def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
    if (epoch + 1) % self.validate_every == 0:
        self.model.training_mode = False
        coco_eval = coco.evaluate(self.model,  # coco_eval에 반환값 저장
                                  self.validation_data, 
                                  self.gtCOCO,
                                  sum(1 for _ in self.validation_data),
                                  self.print_freq)
        self.print_class_mAP(coco_eval, self.class2idx)  # 클래스별 mAP 출력
        self.model.training_mode = True


    def print_class_mAP(self, coco_eval: COCOeval, class2idx: Mapping[str, int]) -> None:
        idx2class = {v: k for k, v in class2idx.items()}
        for i, cId in enumerate(coco_eval.params.catIds):
            mAP = coco_eval.eval['precision'][0, :, i, 0, 2].mean()
            print(f'{idx2class[cId]}: {mAP}')       