"""Learning rate schedules."""

from __future__ import annotations

import math

import tensorflow as tf
import keras


class CosineWithWarmup(keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup followed by cosine annealing to min_lr.

    peak_lr  ──/‾warmup‾\\──cos──► min_lr
    """

    def __init__(
        self,
        peak_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        name: str = "cosine_with_warmup",
    ):
        super().__init__()
        self.peak_lr      = float(peak_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)
        self.min_lr       = float(peak_lr * min_lr_ratio)
        self._name        = name

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.float32)

        warmup_lr = self.peak_lr * (step / tf.maximum(self.warmup_steps, 1.0))

        progress = (step - self.warmup_steps) / tf.maximum(
            self.total_steps - self.warmup_steps, 1.0
        )
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * progress)
        )

        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self) -> dict:
        return {
            "peak_lr":      self.peak_lr,
            "warmup_steps": int(self.warmup_steps),
            "total_steps":  int(self.total_steps),
            "min_lr_ratio": self.min_lr / self.peak_lr,
            "name":         self._name,
        }


class LinearWithWarmup(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup → linear decay."""

    def __init__(self, peak_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.peak_lr      = float(peak_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        warmup = self.peak_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        decay  = self.peak_lr * tf.maximum(
            0.0,
            (self.total_steps - step) / tf.maximum(self.total_steps - self.warmup_steps, 1.0),
        )
        return tf.where(step < self.warmup_steps, warmup, decay)

    def get_config(self) -> dict:
        return {
            "peak_lr":      self.peak_lr,
            "warmup_steps": int(self.warmup_steps),
            "total_steps":  int(self.total_steps),
        }
