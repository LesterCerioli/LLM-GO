"""
Training loop with:
  - Mixed-precision (bfloat16)
  - Gradient accumulation
  - Gradient clipping
  - TensorBoard logging
  - Step-based checkpointing
  - Distributed training via tf.distribute.MirroredStrategy
"""

from __future__ import annotations

import time
from pathlib import Path

import tensorflow as tf
import keras
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from llm_go.config import ModelConfig, TrainingConfig
from llm_go.model.transformer import GoLLM
from llm_go.training.lr_schedule import CosineWithWarmup, LinearWithWarmup

console = Console()


class Trainer:
    """Full training loop for GoLLM."""

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainingConfig,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        strategy: tf.distribute.Strategy | None = None,
    ):
        self.mc = model_config
        self.tc = train_config
        self.train_ds = train_dataset
        self.val_ds   = val_dataset

        # Multi-GPU strategy; fall back to single-device
        self.strategy = strategy or tf.distribute.MirroredStrategy()
        console.print(
            f"[cyan]Devices: {self.strategy.num_replicas_in_sync} × "
            f"{[d.name for d in self.strategy.extended.worker_devices]}"
        )

        # Build model and optimizer inside strategy scope
        with self.strategy.scope():
            self.model = GoLLM(model_config, name="go_llm")
            self.optimizer = self._build_optimizer()

        # Mixed precision
        if train_config.mixed_precision != "float32":
            keras.mixed_precision.set_global_policy(train_config.mixed_precision)

        # TensorBoard
        self.tb_writer = tf.summary.create_file_writer(train_config.log_dir)

        # Checkpoint
        self.ckpt_dir = Path(train_config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="global_step")
        self._accum_grads: list[tf.Variable] | None = None

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> keras.optimizers.Optimizer:
        schedule_cls = (
            CosineWithWarmup
            if self.tc.lr_schedule == "cosine"
            else LinearWithWarmup
        )
        schedule = schedule_cls(
            peak_lr=self.tc.learning_rate,
            warmup_steps=self.tc.warmup_steps,
            total_steps=self.tc.max_steps,
            **({"min_lr_ratio": self.tc.min_lr_ratio} if self.tc.lr_schedule == "cosine" else {}),
        )
        return keras.optimizers.AdamW(
            learning_rate=schedule,
            weight_decay=self.tc.weight_decay,
            beta_1=self.tc.beta1,
            beta_2=self.tc.beta2,
            epsilon=self.tc.epsilon,
            clipnorm=self.tc.gradient_clip,
            global_clipnorm=None,
        )

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def _train_step(
        self, input_ids: tf.Tensor, labels: tf.Tensor
    ) -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            logits = self.model(input_ids, training=True)     # [B, T, V]
            loss   = self._loss_fn(logits, labels)
            # Scale for gradient accumulation
            scaled = loss / tf.cast(self.tc.gradient_accumulation_steps, tf.float32)

        grads = tape.gradient(scaled, self.model.trainable_variables)
        return {"loss": loss, "grads": grads}

    def _loss_fn(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(labels, tf.int32),
            logits=tf.cast(logits, tf.float32),
        )
        mask = tf.cast(tf.not_equal(labels, self.mc.pad_token_id), tf.float32)
        return tf.reduce_sum(loss * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        console.print(
            f"[bold green]Training GoLLM ({self.mc.model_type}) "
            f"for {self.tc.max_steps:,} steps"
        )
        params = self.model.count_params()
        console.print(f"  Parameters: {params['total']:,} total | {params['non_embedding']:,} non-emb")

        step    = 0
        accum   = self.tc.gradient_accumulation_steps
        acc_grads: list[tf.Variable] | None = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} steps"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Training…", total=self.tc.max_steps)

            for input_ids, labels in self.train_ds.repeat():
                if step >= self.tc.max_steps:
                    break

                result = self._train_step(input_ids, labels)
                loss   = result["loss"]
                grads  = result["grads"]

                # Gradient accumulation
                if acc_grads is None:
                    acc_grads = [tf.Variable(g, trainable=False) for g in grads]
                else:
                    for ag, g in zip(acc_grads, grads):
                        ag.assign_add(g)

                micro_step = (step + 1) % accum

                if micro_step == 0:
                    # Apply accumulated gradients
                    self.optimizer.apply_gradients(
                        zip(acc_grads, self.model.trainable_variables)
                    )
                    # Reset accumulators
                    for ag in acc_grads:
                        ag.assign(tf.zeros_like(ag))

                    self.global_step.assign_add(1)
                    opt_step = int(self.global_step.numpy())

                    if opt_step % self.tc.log_every_n_steps == 0:
                        lr = float(self.optimizer.learning_rate(self.global_step))
                        self._log(opt_step, float(loss), lr)
                        progress.update(
                            task,
                            completed=opt_step,
                            description=f"loss={float(loss):.4f} lr={lr:.2e}",
                        )

                    if opt_step % self.tc.eval_every_n_steps == 0:
                        val_loss = self._evaluate()
                        console.print(
                            f"  [cyan]step={opt_step:,} val_loss={val_loss:.4f} "
                            f"val_ppl={tf.exp(val_loss).numpy():.2f}"
                        )
                        with self.tb_writer.as_default():
                            tf.summary.scalar("val/loss", val_loss, step=opt_step)
                            tf.summary.scalar("val/perplexity", tf.exp(val_loss), step=opt_step)

                    if opt_step % self.tc.save_every_n_steps == 0:
                        self._save_checkpoint(opt_step)

                step += 1

        console.print("[bold green]Training complete.")
        self._save_checkpoint(int(self.global_step.numpy()), final=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def _eval_step(self, input_ids: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        logits = self.model(input_ids, training=False)
        return self._loss_fn(logits, labels)

    def _evaluate(self, max_batches: int = 50) -> float:
        total, count = 0.0, 0
        for input_ids, labels in self.val_ds.take(max_batches):
            total += float(self._eval_step(input_ids, labels))
            count += 1
        return total / max(count, 1)

    # ------------------------------------------------------------------
    # Logging & checkpointing
    # ------------------------------------------------------------------

    def _log(self, step: int, loss: float, lr: float) -> None:
        with self.tb_writer.as_default():
            tf.summary.scalar("train/loss",       loss,         step=step)
            tf.summary.scalar("train/perplexity", tf.exp(loss), step=step)
            tf.summary.scalar("train/lr",         lr,           step=step)
        self.tb_writer.flush()

    def _save_checkpoint(self, step: int, final: bool = False) -> None:
        tag  = "final" if final else f"step_{step:07d}"
        dest = self.ckpt_dir / tag
        self.model.save_pretrained(dest)
        console.print(f"  [green]Checkpoint saved → {dest}")
