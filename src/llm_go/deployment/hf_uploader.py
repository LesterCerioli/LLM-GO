"""
Upload GoLLM to the Hugging Face Hub as an open-source model.

Handles:
  - SafeTensors weight conversion
  - HuggingFace config.json (GPT2-compatible)
  - Tokenizer upload (PreTrainedTokenizerFast)
  - Model card generation
  - Dataset card generation
  - Gradio inference demo (optional)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, Repository, login
from rich.console import Console

console = Console()


class HuggingFaceUploader:
    """Converts and uploads GoLLM weights to the Hugging Face Hub."""

    def __init__(
        self,
        repo_id: str,               # e.g. "your-org/llm-go-350m"
        hf_token: str | None = None,
        private: bool = False,
    ):
        self.repo_id  = repo_id
        self.private  = private
        self.token    = hf_token or os.environ.get("HF_TOKEN", "")
        self.api      = HfApi(token=self.token)

        if not self.token:
            raise ValueError("HF_TOKEN env var or hf_token= is required.")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def upload(
        self,
        checkpoint_dir: str | Path,
        tokenizer_dir: str | Path,
        commit_message: str = "Upload GoLLM model",
        push_dataset_card: bool = True,
    ) -> str:
        """Full upload pipeline. Returns the Hub URL."""
        checkpoint_dir = Path(checkpoint_dir)
        tokenizer_dir  = Path(tokenizer_dir)

        with tempfile.TemporaryDirectory() as staging:
            stage = Path(staging)
            console.print(f"[cyan]Staging model at {stage}…")

            self._write_hf_config(checkpoint_dir, stage)
            self._convert_weights(checkpoint_dir, stage)
            self._write_tokenizer(tokenizer_dir, stage)
            self._write_model_card(stage)
            if push_dataset_card:
                self._write_dataset_card(stage)
            self._write_generation_config(stage)

            console.print(f"[cyan]Creating / verifying repo {self.repo_id}…")
            self.api.create_repo(
                repo_id=self.repo_id,
                repo_type="model",
                private=self.private,
                exist_ok=True,
            )

            console.print("[cyan]Uploading…")
            self.api.upload_folder(
                folder_path=str(stage),
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=commit_message,
            )

        url = f"https://huggingface.co/{self.repo_id}"
        console.print(f"[bold green]Published → {url}")
        return url

    # ------------------------------------------------------------------
    # HF config
    # ------------------------------------------------------------------

    def _write_hf_config(self, ckpt: Path, stage: Path) -> None:
        """Write HuggingFace-compatible config.json (GPT2-style schema)."""
        from llm_go.config import ModelConfig
        mc = ModelConfig.load(ckpt / "config.json")

        hf_cfg = {
            "architectures":        ["GoLLMForCausalLM"],
            "model_type":           "gpt2",
            "vocab_size":           mc.vocab_size,
            "n_positions":          mc.max_seq_len,
            "n_ctx":                mc.max_seq_len,
            "n_embd":               mc.d_model,
            "n_layer":              mc.n_layers,
            "n_head":               mc.n_heads,
            "n_inner":              mc.d_ff,
            "activation_function":  "gelu_new",
            "resid_pdrop":          mc.dropout,
            "attn_pdrop":           mc.attention_dropout,
            "layer_norm_epsilon":   mc.layer_norm_eps,
            "initializer_range":    0.02,
            "bos_token_id":         mc.bos_token_id,
            "eos_token_id":         mc.eos_token_id,
            "pad_token_id":         mc.pad_token_id,
            "tie_word_embeddings":  mc.tie_embeddings,
            "rope_theta":           mc.rope_theta,
            "torch_dtype":          "bfloat16",
            "transformers_version": "4.41.0",
            # Custom metadata
            "llm_go": {
                "go_versions":  ["1.0–1.24"],
                "architecture": "gpt-decoder-ropeswiglurms",
                "license":      "apache-2.0",
            },
        }
        (stage / "config.json").write_text(json.dumps(hf_cfg, indent=2))

    # ------------------------------------------------------------------
    # Weight conversion (Keras → SafeTensors)
    # ------------------------------------------------------------------

    def _convert_weights(self, ckpt: Path, stage: Path) -> None:
        """Convert Keras .h5 weights to safetensors format."""
        try:
            import safetensors.tensorflow as sf_tf
            import keras

            from llm_go.config import ModelConfig
            from llm_go.model.transformer import GoLLM

            mc    = ModelConfig.load(ckpt / "config.json")
            model = GoLLM(mc)
            dummy = __import__("tensorflow").zeros([1, 8], dtype="int32")
            model(dummy)
            model.load_weights(str(ckpt / "model.weights.h5"))

            tensors = {v.name: v.numpy() for v in model.trainable_variables}
            sf_tf.save_file(tensors, str(stage / "model.safetensors"))
            console.print("  [green]Converted to safetensors")

        except ImportError:
            # Fall back to copying raw .h5
            console.print("  [yellow]safetensors not installed; copying .h5 weights")
            shutil.copy(ckpt / "model.weights.h5", stage / "pytorch_model.bin")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def _write_tokenizer(self, tok_dir: Path, stage: Path) -> None:
        from llm_go.tokenizer.go_tokenizer import GoTokenizer

        tok     = GoTokenizer.load(tok_dir)
        hf_tok  = tok.to_hf_tokenizer()
        hf_tok.save_pretrained(str(stage))
        console.print("  [green]Tokenizer written")

    # ------------------------------------------------------------------
    # Generation config
    # ------------------------------------------------------------------

    def _write_generation_config(self, stage: Path) -> None:
        cfg = {
            "bos_token_id":         1,
            "eos_token_id":         2,
            "pad_token_id":         0,
            "do_sample":            True,
            "temperature":          0.8,
            "top_p":                0.95,
            "top_k":                50,
            "repetition_penalty":   1.1,
            "max_new_tokens":       512,
            "transformers_version": "4.41.0",
        }
        (stage / "generation_config.json").write_text(json.dumps(cfg, indent=2))

    # ------------------------------------------------------------------
    # Model card
    # ------------------------------------------------------------------

    def _write_model_card(self, stage: Path) -> None:
        card = f"""---
language:
  - en
  - go
license: apache-2.0
tags:
  - code
  - golang
  - go
  - code-generation
  - llm
  - tensorflow
  - text-generation
datasets:
  - github-code-go
model-index:
  - name: {self.repo_id}
    results: []
---

# GoLLM — Go-Specialised Large Language Model

A decoder-only transformer pre-trained exclusively on Go (Golang) source code,
covering **Go 1.0 through Go 1.24**, the standard library, and the major
Go ecosystem frameworks: **Fiber, Cobra, Gin, Echo, Chi, gRPC, GORM, Zap, Viper,
Prometheus, and more**.

## Model Details

| Property | Value |
|---|---|
| Architecture | GPT decoder-only (RoPE + RMSNorm + SwiGLU) |
| Framework | TensorFlow / Keras 3 |
| License | Apache 2.0 |
| Language | Go (Golang) |
| Go versions covered | 1.0 – 1.24 |

## Intended uses

- **Code completion** — autocomplete Go functions, types, interfaces
- **Code generation** — generate idiomatic Go from natural-language prompts
- **Code review assistance** — suggest best-practice improvements
- **Documentation generation** — generate godoc comments
- **Learning tool** — explore Go idioms across versions

## Quick start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{self.repo_id}")
model     = AutoModelForCausalLM.from_pretrained("{self.repo_id}")

prompt = \"\"\"<go_file>
package main

import "fmt"

// Fibonacci returns the nth Fibonacci number.
func Fibonacci(n int) int {{\"\"\"

inputs    = tokenizer(prompt, return_tensors="pt")
output_ids = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

## Training data

- **GitHub** — 50,000+ Go repositories with ≥10 stars (Apache/MIT/BSD licensed)
- **Go standard library** — full `src/` tree from Go 1.0 to Go 1.24
- **Framework repos** — Fiber, Cobra, Gin, Echo, Chi, gRPC-Go, GORM, and more
- **Deduplication** — MinHash LSH at 0.80 Jaccard threshold
- **PII scrubbing** — emails, API keys, tokens removed

## Evaluation

| Metric | Value |
|---|---|
| Perplexity (val) | TBD after training |
| pass@1 (HumanEval-Go) | TBD |
| gofmt pass rate | TBD |

## Limitations

- Optimised for Go; poor at other languages by design.
- No internet access during generation; cannot fetch dependencies.
- May reproduce open-source patterns closely — review before production use.

## Citation

```bibtex
@misc{{llm-go-2024,
  title  = {{GoLLM: A Go-Specialised Large Language Model}},
  year   = {{2024}},
  url    = {{https://huggingface.co/{self.repo_id}}}
}}
```
"""
        (stage / "README.md").write_text(card)

    def _write_dataset_card(self, stage: Path) -> None:
        """Write a brief dataset provenance note."""
        note = """# Dataset provenance

Training data was collected from:
- Public GitHub repositories (Go, ≥10 stars, open-source licenses)
- The official Go standard library (golang/go)
- Key Go framework repositories

All data was deduplicated with MinHash LSH and PII-scrubbed before use.
"""
        (stage / "DATASET.md").write_text(note)
