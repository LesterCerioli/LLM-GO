# LLM-GO

A Go-specialized large language model built with TensorFlow 2 and Python 3.12. Trained on all Golang versions (1.0–1.24), the Fiber and Cobra ecosystems, real-world project patterns, and Go best practices. Published to Hugging Face as an open-source model under the Apache 2.0 license.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Model Sizes](#model-sizes)
- [Training Data](#training-data)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Pipeline](#pipeline)
- [Go Layout Rule](#go-layout-rule)
- [Supported Frameworks](#supported-frameworks)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Deploying to Hugging Face](#deploying-to-hugging-face)
- [Development](#development)
- [License](#license)

---

## Overview

**llm-go** is a decoder-only transformer model designed exclusively for Go code generation, completion, and explanation. It understands Go idioms, project layout conventions, the standard library across all major versions, and the most widely used frameworks in the Go ecosystem.

Key goals:

- Complete coverage of Go 1.0 through 1.24
- Deep knowledge of Fiber, Cobra, GORM, Gin, Echo, gRPC, and more
- Enforces canonical Go project layout (`cmd/` always at the repo root)
- Trained on real-world patterns extracted from production Go projects
- Fully open-source and deployable via the Hugging Face Hub

---

## Architecture

GoLLM is a GPT-style decoder-only transformer with modern improvements from LLaMA/Mistral:

| Component | Implementation |
|---|---|
| Attention | Multi-head causal self-attention |
| Positional encoding | RoPE (Rotary Position Embedding) |
| Normalization | RMSNorm (pre-norm, before each sub-layer) |
| Feed-forward | SwiGLU activation (`silu(gate(x)) * up(x)`) |
| Embeddings | Tied input/output embeddings |
| Tokenizer | BPE via HuggingFace `tokenizers` (Rust-backed) |
| Training precision | bfloat16 mixed precision |
| Multi-GPU | TensorFlow `MirroredStrategy` |
| Optimizer | AdamW + cosine LR schedule with warmup |

### Special Tokens

The tokenizer uses structural tags so the model understands Go file anatomy:

```
<go_file>   <go_func>   <go_type>   <go_pkg>   <go_version>
<go_test>   <go_comment>
<task:generate>   <task:complete>   <task:fix>   <task:explain>   <task:optimize>
```

---

## Model Sizes

| Variant | Parameters | d_model | Layers | Heads | Context | Use case |
|---|---|---|---|---|---|---|
| `small` | ~125 M | 768 | 12 | 12 | 2 048 | CPU / fast iteration |
| `medium` | ~350 M | 1 024 | 24 | 16 | 2 048 | Single GPU (default) |
| `large` | ~760 M | 1 280 | 36 | 20 | 4 096 | Multi-GPU |
| `xl` | ~1.5 B | 1 600 | 48 | 25 | 4 096 | Near state-of-the-art |

The default training target is `medium`. Override with `MODEL_SIZE=large make train`.

---

## Training Data

### Real-world corpus

- Up to 50 000 Go repositories from GitHub (≥10 stars)
- Go standard library source across all versions (1.0–1.24)
- Official documentation and release notes

### Synthetic patterns (oversampled)

Patterns extracted from real production Go projects and rendered across multiple Go versions, business domains, and application types:

| Category | Examples | Source |
|---|---|---|
| Fiber controllers | ~36 | Struct-based handlers, constructor injection, Swagger |
| GORM repositories | ~52 | UUID PKs, soft delete, repo interface pattern |
| Service layer | ~32 | `errgroup`, DI container, RabbitMQ consumer |
| JWT / Auth | ~16 | HS256, bcrypt, Bearer middleware, CPF/CNPJ validators |
| Tests | ~20 | `go-sqlmock`, testify, `fiber.App.Test()`, table-driven |
| Docker / CI | ~40 | Multi-stage Dockerfile, docker-compose, Jenkinsfile |
| **Total** | **~196** | |

Layout examples are oversampled **5×** and pattern examples **3×** to reinforce correct conventions.

### Deduplication

MinHash LSH with 128 permutations, 32 bands, and a 0.80 Jaccard similarity threshold removes near-duplicate files before tokenization.

### Dataset format

Preprocessed data is stored as sharded TFRecord files in `data/processed/{train,val,test}/`.

---

## Project Structure

```
llm-go/
├── cmd/                          # (Go convention — always at root)
├── configs/
│   ├── small.yaml
│   ├── medium.yaml
│   └── large.yaml
├── data/
│   ├── raw/                      # downloaded Go source files
│   ├── processed/                # TFRecord shards
│   └── tokenizer/                # trained BPE tokenizer
├── scripts/
│   ├── setup_env.sh
│   ├── collect_data.sh
│   ├── build_tokenizer.sh
│   ├── preprocess.sh
│   ├── train.sh
│   ├── evaluate.sh
│   ├── generate.sh
│   └── deploy_huggingface.sh
├── src/llm_go/
│   ├── config.py                 # ModelConfig, TrainingConfig, DataConfig
│   ├── model/
│   │   ├── attention.py          # RoPE + MultiHeadAttention
│   │   └── transformer.py        # RMSNorm, SwiGLU, TransformerBlock, GoLLM
│   ├── tokenizer/
│   │   └── go_tokenizer.py       # BPE + structural tag injection
│   ├── data/
│   │   ├── collector.py          # GitHub + stdlib scraper
│   │   ├── preprocessor.py       # filter → dedup → tokenize → TFRecord
│   │   ├── go_best_practices.py  # GoProjectTemplates + GoLayoutValidator
│   │   ├── templates/
│   │   │   ├── loader.py
│   │   │   └── go_project/       # canonical cmd/ layout examples
│   │   └── patterns/
│   │       ├── fiber_patterns.py
│   │       ├── gorm_patterns.py
│   │       ├── service_patterns.py
│   │       ├── auth_patterns.py
│   │       ├── test_patterns.py
│   │       ├── docker_patterns.py
│   │       └── registry.py       # PatternRegistry (~196 examples)
│   ├── training/
│   │   ├── trainer.py            # gradient accumulation, MirroredStrategy
│   │   └── lr_schedule.py        # CosineWithWarmup
│   ├── evaluation/
│   │   └── metrics.py            # perplexity, pass@k, gofmt rate, BLEU, ROUGE-L
│   ├── deployment/
│   │   └── hf_uploader.py        # safetensors + model card → HF Hub
│   └── scripts/                  # CLI entry points
│       ├── collect.py
│       ├── tokenize.py
│       ├── train.py
│       ├── evaluate.py
│       ├── generate.py
│       └── deploy.py
├── tests/
│   ├── conftest.py               # shared pytest fixtures
│   ├── test_model.py
│   ├── test_tokenizer.py
│   └── test_best_practices.py
├── checkpoints/                  # saved during training
├── logs/                         # TensorBoard event files
├── Makefile
├── pyproject.toml
├── requirements.txt
└── requirements-gpu.txt
```

---

## Requirements

- Python 3.12
- TensorFlow 2.17.1 (CPU) or `tensorflow[and-cuda]` for GPU
- CUDA 12.x + cuDNN 8.x (optional, GPU only)

### Python 3.12 compatibility notes

| Package | Version | Note |
|---|---|---|
| `tensorflow` | 2.17.1 | cp312 wheel confirmed (manylinux) |
| `keras` | 3.5.0 | compatible with TF 2.17.x |
| `numpy` | 1.26.4 | TF 2.17.x requires numpy < 2 |
| `tensorboard` | 2.17.1 | must match TF version |
| `tensorflow-text` | — | skipped 2.17.x release; not used (tokenization via HF `tokenizers`) |
| `tree-sitter` | optional | core pipeline uses regex tagging; see `requirements.txt` comments |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-org/llm-go.git
cd llm-go

# CPU
bash scripts/setup_env.sh

# GPU (NVIDIA CUDA 12)
bash scripts/setup_env.sh --gpu
```

Or manually:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 2. Generate code (from a pre-trained checkpoint)

```bash
# using the Makefile
make generate

# or directly
llm-go-generate \
  --model-dir checkpoints/final \
  --tok-dir   data/tokenizer \
  --prompt    "package main\n\nimport \"github.com/gofiber/fiber/v2\"\n\nfunc main() {"
```

### 3. Generate with a Python script

```python
from llm_go.model.transformer import GoLLM
from llm_go.tokenizer.go_tokenizer import GoTokenizer

tok   = GoTokenizer.load("data/tokenizer")
model = GoLLM.from_pretrained("checkpoints/final")

prompt = """<go_version>1.24</go_version>
<go_file>cmd/server/main.go</go_file>
package main

import "github.com/gofiber/fiber/v2"

func main() {"""

ids    = tok.encode(prompt)
output = model.generate(ids, max_new_tokens=256, temperature=0.8, top_p=0.95)
print(tok.decode(output))
```

---

## Pipeline

Run each stage individually or all at once with `make pipeline`.

### Stage 1 — Collect data

```bash
export GITHUB_TOKEN=ghp_...
make collect
# or
bash scripts/collect_data.sh
```

Downloads Go repositories (≥10 stars, configurable) and the standard library into `data/raw/`.

### Stage 2 — Build tokenizer

```bash
make tokenize
# or
bash scripts/build_tokenizer.sh
```

Trains a 32 000-token BPE vocabulary on the raw corpus with Go keywords, builtins, and packages seeded as the initial alphabet.

### Stage 3 — Preprocess

```bash
make preprocess
# or
bash scripts/preprocess.sh
```

Applies quality filtering → MinHash LSH deduplication → PII scrubbing → tokenization → sequence packing → TFRecord sharding.

Synthetic layout and pattern examples are prepended and oversampled before the real data.

### Stage 4 — Train

```bash
# Default: medium model, bfloat16, all available GPUs
make train

# Choose size
make train-small
make train-large
MODEL_SIZE=xl make train

# Custom
MODEL_SIZE=medium BATCH_SIZE=64 MAX_STEPS=200000 bash scripts/train.sh
```

Training uses XLA JIT compilation, gradient accumulation (default 4 steps), and TensorFlow `MirroredStrategy` for multi-GPU.

Monitor with TensorBoard:

```bash
make tb
# opens http://localhost:6006
```

### Stage 5 — Evaluate

```bash
make evaluate
# or
bash scripts/evaluate.sh
```

Reports perplexity, pass@k (unbiased estimator), `gofmt` syntax pass rate, BLEU, and ROUGE-L.

### Stage 6 — Deploy to Hugging Face

```bash
export HF_TOKEN=hf_...
export HF_REPO_ID=your-org/llm-go-350m

make deploy
# or
bash scripts/deploy_huggingface.sh
```

Converts Keras weights to SafeTensors format, uploads the tokenizer as `PreTrainedTokenizerFast`, and generates a model card automatically.

---

## Go Layout Rule

One of the core conventions this model learns and enforces:

> **`cmd/` is always at the project root. Each binary lives in its own subdirectory with a `main.go`.**

```
my-project/              ← project root
├── cmd/
│   ├── server/
│   │   └── main.go      ← binary: server
│   ├── worker/
│   │   └── main.go      ← binary: background worker
│   └── cli/
│       └── main.go      ← binary: CLI tool
├── internal/
│   ├── config/
│   ├── handler/
│   └── service/
├── go.mod
└── go.sum
```

`main.go` only wires dependencies. All business logic lives in `internal/`. The `cmd/` directory is **never** nested inside `internal/`, `pkg/`, or any other subdirectory.

The `GoLayoutValidator` class enforces this during data collection: files from repositories with a nested or missing `cmd/` receive a lower training weight.

---

## Supported Frameworks

GoLLM is trained on idiomatic usage of the following libraries:

| Framework | Purpose |
|---|---|
| `github.com/gofiber/fiber/v2` | HTTP server (primary) |
| `github.com/spf13/cobra` | CLI applications |
| `github.com/spf13/viper` | Configuration |
| `gorm.io/gorm` | ORM + PostgreSQL |
| `github.com/gin-gonic/gin` | HTTP server (alternative) |
| `github.com/labstack/echo` | HTTP server (alternative) |
| `github.com/go-chi/chi` | Lightweight HTTP router |
| `google.golang.org/grpc` | gRPC services |
| `github.com/stretchr/testify` | Testing assertions |
| `go.uber.org/zap` | Structured logging |
| `github.com/golang-jwt/jwt` | JWT authentication |
| `golang.org/x/crypto/bcrypt` | Password hashing |
| `github.com/rabbitmq/amqp091-go` | RabbitMQ messaging |
| `github.com/redis/go-redis/v9` | Redis client |
| `github.com/prometheus/client_golang` | Metrics |
| `github.com/DATA-DOG/go-sqlmock` | SQL mocking in tests |

---

## Configuration

Training parameters can be set via environment variables, YAML configs, or Makefile overrides.

```bash
# Environment variables (all optional — defaults shown)
MODEL_SIZE=medium        # small | medium | large | xl
BATCH_SIZE=32
MAX_STEPS=100000
WARMUP_STEPS=2000
GRAD_ACCUM=4
PRECISION=bfloat16       # float32 | float16 | bfloat16
GPUS=-1                  # -1 = all GPUs, 0 = GPU 0 only
CKPT_DIR=checkpoints
LOG_DIR=logs
```

YAML configs for each size are in `configs/`:

```bash
# train from a YAML config
llm-go-train --config configs/large.yaml
```

---

## Evaluation

Metrics computed by `GoCodeEvaluator`:

| Metric | Description |
|---|---|
| Perplexity | Cross-entropy exponentiated on the validation split |
| pass@k | Unbiased estimator of functional correctness (k=1,10,100) |
| gofmt pass rate | % of generated files that parse and format without error |
| BLEU | n-gram overlap vs. reference completions |
| ROUGE-L | Longest-common-subsequence F1 vs. references |

---

## Deploying to Hugging Face

The uploader (`HuggingFaceUploader`) handles everything:

1. Converts Keras weights → SafeTensors
2. Writes `config.json` in GPT-2-compatible format
3. Uploads `PreTrainedTokenizerFast` (usable with `transformers`)
4. Generates a model card with usage examples
5. Optionally creates a Gradio demo space

```bash
export HF_TOKEN=hf_...
export HF_REPO_ID=your-org/llm-go-350m

llm-go-deploy \
  --ckpt-dir checkpoints/final \
  --tok-dir  data/tokenizer \
  --repo-id  "$HF_REPO_ID" \
  --token    "$HF_TOKEN" \
  --public
```

Once uploaded, use the model from any Python environment:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("your-org/llm-go-350m")
model     = AutoModelForCausalLM.from_pretrained("your-org/llm-go-350m")

inputs = tokenizer("package main\n\nfunc main() {", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0]))
```

---

## Development

### Run tests

```bash
make test
# or
pytest tests/ -v --cov=llm_go --cov-report=term-missing
```

### Lint and format

```bash
make lint    # ruff + mypy
make fmt     # black + ruff --fix
```

### Pre-commit hooks

```bash
pre-commit install
```

### GPU setup (NVIDIA)

```bash
pip install -r requirements-gpu.txt
# verify
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Patterns derived from real-world Go projects are used for educational and model-training purposes only. All generated code is original output of the model.
