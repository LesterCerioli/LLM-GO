"""CLI: run the full training loop."""

import click
import tensorflow as tf
from llm_go.config import ModelConfig, TrainingConfig
from llm_go.data.dataset import GoDataset
from llm_go.training.trainer import Trainer


@click.command()
@click.option("--model-size",   default="medium",       type=click.Choice(["small","medium","large","xl"]))
@click.option("--data-dir",     default="data/processed", show_default=True)
@click.option("--ckpt-dir",     default="checkpoints",  show_default=True)
@click.option("--log-dir",      default="logs",         show_default=True)
@click.option("--batch-size",   default=32,             show_default=True)
@click.option("--max-steps",    default=100_000,        show_default=True)
@click.option("--lr",           default=3e-4,           show_default=True)
@click.option("--warmup-steps", default=2000,           show_default=True)
@click.option("--grad-accum",   default=4,              show_default=True)
@click.option("--precision",    default="bfloat16",     type=click.Choice(["float32","float16","bfloat16"]))
@click.option("--gpus",         default=-1,             help="-1 = all GPUs")
def main(model_size, data_dir, ckpt_dir, log_dir, batch_size, max_steps,
         lr, warmup_steps, grad_accum, precision, gpus):
    """Train GoLLM from scratch."""

    mc = {"small": ModelConfig.small, "medium": ModelConfig.medium,
          "large": ModelConfig.large, "xl": ModelConfig.xl}[model_size]()

    tc = TrainingConfig(
        learning_rate=lr,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        mixed_precision=precision,
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir,
    )

    ds = GoDataset(data_dir, seq_len=mc.max_seq_len, batch_size=batch_size)

    if gpus == 1:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
    elif gpus == -1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        devices = [f"/gpu:{i}" for i in range(gpus)]
        strategy = tf.distribute.MirroredStrategy(devices=devices)

    trainer = Trainer(mc, tc, ds.train(), ds.val(), strategy=strategy)
    trainer.train()
