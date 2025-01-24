import math

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class WarmupCosineDecayLRScheduler(LearningRateSchedule):

    def __init__(self, 
                 max_lr: float,
                 warmup_steps: int,
                 decay_steps: int,
                 alpha: float = 0.) -> None:
        super(WarmupCosineDecayLRScheduler, self).__init__()

        self.name = 'WarmupCosineDecayLRScheduler'
        self.alpha = alpha

        self.max_lr = max_lr
        self.last_step = 0

        self.warmup_steps = int(warmup_steps)
        self.linear_increase = self.max_lr / float(self.warmup_steps)

        self.decay_steps = int(decay_steps)

    def _decay(self) -> tf.Tensor:
        rate = tf.subtract(self.last_step, self.warmup_steps) 
        rate = tf.divide(rate, self.decay_steps)
        rate = tf.cast(rate, tf.float32)

        cosine_decayed = tf.multiply(tf.constant(math.pi), rate)
        cosine_decayed = tf.add(1., tf.cos(cosine_decayed))
        cosine_decayed = tf.multiply(.5, cosine_decayed)

        decayed = tf.subtract(1., self.alpha)
        decayed = tf.multiply(decayed, cosine_decayed)
        decayed = tf.add(decayed, self.alpha)
        return tf.multiply(self.max_lr, decayed)

    @property
    def current_lr(self) -> tf.Tensor:
        return tf.cond(
            tf.less(self.last_step, self.warmup_steps),
            lambda: tf.multiply(self.linear_increase, self.last_step),
            lambda: self._decay())

    def __call__(self, step: int) -> tf.Tensor:
        self.last_step = step
        return self.current_lr

    def get_config(self) -> dict:
        config = {
            "max_lr": self.max_lr,
            "warmup_steps": self.warmup_steps,
            'decay_steps': self.decay_steps,
            'alpha': self.alpha
        }
        return config

class WarmupCosineAnnealingLRScheduler(LearningRateSchedule):

    def __init__(self, 
                 max_lr: float,
                 warmup_steps: int,
                 total_steps: int,
                 annealing_steps: int,
                 alpha: float = 0.) -> None:
        super(WarmupCosineAnnealingLRScheduler, self).__init__()

        self.name = 'WarmupCosineAnnealingLRScheduler'
        self.alpha = alpha

        self.max_lr = max_lr
        self.last_step = 0

        self.warmup_steps = int(warmup_steps)
        self.linear_increase = self.max_lr / float(self.warmup_steps)

        self.annealing_steps = int(annealing_steps)
        self.total_steps = int(total_steps)

    def _annealing(self) -> tf.Tensor:
        rate = tf.subtract(self.last_step, self.warmup_steps) 
        rate = tf.divide(rate, self.annealing_steps)
        rate = tf.cast(rate, tf.float32)

        cosine_annealed = tf.multiply(tf.constant(math.pi), rate)
        cosine_annealed = tf.add(1., tf.cos(cosine_annealed))
        cosine_annealed = tf.multiply(.5, cosine_annealed)

        annealed = tf.subtract(1., self.alpha)
        annealed = tf.multiply(annealed, cosine_annealed)
        annealed = tf.add(annealed, self.alpha)
        return tf.multiply(self.max_lr, annealed)

    @property
    def current_lr(self) -> tf.Tensor:
        return tf.cond(
            tf.less(self.last_step, self.warmup_steps),
            lambda: tf.multiply(self.linear_increase, self.last_step),
            lambda: self._annealing())

    def __call__(self, step: int) -> tf.Tensor:
        self.last_step = step
        return self.current_lr

    def get_config(self) -> dict:
        config = {
            "max_lr": self.max_lr,
            "warmup_steps": self.warmup_steps,
            'total_steps': self.total_steps,
            'annealing_steps': self.annealing_steps,
            'alpha': self.alpha
        }
        return config