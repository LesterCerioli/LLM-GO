"""
Evaluation metrics for GoLLM:
  - Perplexity on held-out Go code
  - pass@k (HumanEval-Go style) — requires gofmt + go vet
  - BLEU / ROUGE-L for documentation generation
  - Exact-match for fill-in-the-middle
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf
from rich.console import Console

console = Console()


class CodeEvaluator:
    """Evaluation harness for GoLLM."""

    def __init__(self, model, tokenizer, device: str = "CPU:0"):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------

    def perplexity(self, dataset: tf.data.Dataset, max_batches: int = 200) -> float:
        """Token-level perplexity on a tf.data.Dataset."""
        total_loss, total_tokens = 0.0, 0

        for input_ids, labels in dataset.take(max_batches):
            with tf.device(self.device):
                logits = self.model(input_ids, training=False)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.cast(labels, tf.int32),
                logits=tf.cast(logits, tf.float32),
            )
            mask         = tf.cast(tf.not_equal(labels, 0), tf.float32)
            total_loss  += float(tf.reduce_sum(loss * mask))
            total_tokens += int(tf.reduce_sum(mask))

        return math.exp(total_loss / max(total_tokens, 1))

    # ------------------------------------------------------------------
    # pass@k (functional correctness)
    # ------------------------------------------------------------------

    def pass_at_k(
        self,
        problems: list[dict],
        k: int = 1,
        n_samples: int = 10,
        temperature: float = 0.8,
    ) -> float:
        """
        Estimate pass@k using the unbiased estimator from Chen et al. 2021:
            pass@k = 1 - C(n-c, k) / C(n, k)

        `problems` is a list of dicts with keys:
            - "prompt"      : str  (Go function signature + docstring)
            - "test"        : str  (Go test code asserting correctness)
            - "entry_point" : str  (function name to test)
        """
        results: list[float] = []
        for problem in problems:
            prompt_ids = self.tokenizer.encode(problem["prompt"])
            passed = 0

            for _ in range(n_samples):
                completion_ids = self.model.generate(
                    tf.constant([prompt_ids], dtype=tf.int32),
                    max_new_tokens=512,
                    temperature=temperature,
                )
                completion = self.tokenizer.decode(
                    completion_ids[0].numpy().tolist(), skip_special_tokens=True
                )
                code = problem["prompt"] + completion
                if self._go_test_passes(code, problem["test"]):
                    passed += 1

            results.append(self._estimator(n_samples, passed, k))

        return float(np.mean(results))

    @staticmethod
    def _estimator(n: int, c: int, k: int) -> float:
        """Unbiased pass@k estimator."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod([(n - c - i) / (n - i) for i in range(k)])

    def _go_test_passes(self, implementation: str, test_code: str) -> bool:
        """Write a temporary Go file, run `go test`, return True if passes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "solution.go").write_text(implementation)
            (p / "solution_test.go").write_text(test_code)
            # Minimal go.mod
            (p / "go.mod").write_text("module llmgoeval\ngo 1.22\n")
            try:
                result = subprocess.run(
                    ["go", "test", "-timeout", "10s", "./..."],
                    cwd=tmpdir,
                    capture_output=True,
                    timeout=15,
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False

    # ------------------------------------------------------------------
    # go vet / gofmt quality
    # ------------------------------------------------------------------

    def syntax_pass_rate(self, completions: list[str]) -> float:
        """Fraction of completions that pass `gofmt` without errors."""
        passed = 0
        for code in completions:
            with tempfile.NamedTemporaryFile(suffix=".go", mode="w", delete=False) as f:
                f.write(code)
                fname = f.name
            try:
                r = subprocess.run(
                    ["gofmt", "-e", fname], capture_output=True, timeout=5
                )
                if r.returncode == 0:
                    passed += 1
            except Exception:
                pass
        return passed / max(len(completions), 1)

    # ------------------------------------------------------------------
    # BLEU / ROUGE-L (for doc-gen tasks)
    # ------------------------------------------------------------------

    def bleu(self, references: list[str], hypotheses: list[str]) -> float:
        from evaluate import load as hf_load
        metric = hf_load("sacrebleu")
        score  = metric.compute(predictions=hypotheses, references=[[r] for r in references])
        return float(score["score"])

    def rouge_l(self, references: list[str], hypotheses: list[str]) -> float:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, hypotheses)]
        return float(np.mean(scores))

    # ------------------------------------------------------------------
    # Full eval report
    # ------------------------------------------------------------------

    def full_report(
        self,
        val_dataset: tf.data.Dataset,
        problems: list[dict] | None = None,
        sample_completions: list[str] | None = None,
    ) -> dict[str, float]:
        report: dict[str, float] = {}

        console.print("[cyan]Computing perplexity…")
        report["perplexity"] = self.perplexity(val_dataset)
        console.print(f"  perplexity = {report['perplexity']:.2f}")

        if problems:
            console.print("[cyan]Computing pass@1…")
            report["pass@1"] = self.pass_at_k(problems, k=1, n_samples=10)
            report["pass@5"] = self.pass_at_k(problems, k=5, n_samples=10)
            console.print(f"  pass@1={report['pass@1']:.3f}  pass@5={report['pass@5']:.3f}")

        if sample_completions:
            console.print("[cyan]Computing syntax pass rate…")
            report["syntax_pass"] = self.syntax_pass_rate(sample_completions)
            console.print(f"  syntax_pass = {report['syntax_pass']:.3f}")

        return report
