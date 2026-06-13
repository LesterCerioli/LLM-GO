from __future__ import annotations

from pathlib import Path

import tensorflow as tf


class GoDataset:
    """Factory for train/val/test tf.data.Dataset objects."""

    def __init__(
        self,
        processed_dir: str | Path,
        seq_len: int = 2048,
        batch_size: int = 32,
        shuffle_buffer: int = 50_000,
        seed: int = 42,
    ):
        self.processed_dir  = Path(processed_dir)
        self.seq_len        = seq_len
        self.batch_size     = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.seed           = seed

    def build(self, split: str) -> tf.data.Dataset:
        split_dir = self.processed_dir / split
        tfrecords = sorted(split_dir.glob("*.tfrecord"))
        if not tfrecords:
            raise FileNotFoundError(f"No TFRecords found in {split_dir}")

        fileset = tf.data.Dataset.from_tensor_slices([str(p) for p in tfrecords])
        if split == "train":
            fileset = fileset.shuffle(len(tfrecords), seed=self.seed, reshuffle_each_iteration=True)

        raw = fileset.interleave(
            lambda p: tf.data.TFRecordDataset(p, compression_type=""),
            cycle_length=16,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=(split != "train"),
        )

        feature_spec = {
            "input_ids": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
        }

        def _parse(serialised: tf.Tensor):
            parsed = tf.io.parse_single_example(serialised, feature_spec)
            ids = tf.cast(parsed["input_ids"], tf.int32)
            return ids[:-1], ids[1:]     # (input_ids, labels)

        ds = raw.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

        if split == "train":
            ds = ds.shuffle(self.shuffle_buffer, seed=self.seed)

        ds = (
            ds.batch(self.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        return ds

    def train(self) -> tf.data.Dataset:
        return self.build("train")

    def val(self) -> tf.data.Dataset:
        return self.build("val")

    def test(self) -> tf.data.Dataset:
        return self.build("test")

    def steps_per_epoch(self, split: str = "train") -> int:
        """Estimate steps; requires at least one pass through the dataset."""
        count = sum(
            1 for _ in self.build(split).unbatch().batch(self.batch_size)
        )
        return count
