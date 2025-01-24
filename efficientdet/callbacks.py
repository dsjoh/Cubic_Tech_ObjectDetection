from typing import Mapping

import tensorflow as tf

from . import coco


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
            coco.evaluate(self.model, 
                          self.validation_data, 
                          self.gtCOCO,
                          sum(1 for _ in self.validation_data),
                          self.print_freq)

            self.model.training_mode = True
