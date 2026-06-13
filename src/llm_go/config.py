"""Model, training, and data configurations for llm-go."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ModelConfig:
    """Transformer architecture hyper-parameters."""

    # Architecture
    vocab_size: int = 32_000
    max_seq_len: int = 2048
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072           # feed-forward hidden dim (4 * d_model typical)
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    tie_embeddings: bool = True  # share input/output embeddings

    # RoPE positional encoding
    rope_theta: float = 10_000.0

    # Model identity
    model_type: Literal["small", "medium", "large", "xl"] = "medium"
    architecture: str = "gpt-decoder"

    # Padding / special tokens (aligned with tokenizer)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    @classmethod
    def small(cls) -> "ModelConfig":
        """~125 M parameters — fast iteration / CPU-feasible."""
        return cls(d_model=768, n_heads=12, n_layers=12, d_ff=3072, model_type="small")

    @classmethod
    def medium(cls) -> "ModelConfig":
        """~350 M parameters — good quality / single-GPU."""
        return cls(
            d_model=1024,
            n_heads=16,
            n_layers=24,
            d_ff=4096,
            max_seq_len=2048,
            model_type="medium",
        )

    @classmethod
    def large(cls) -> "ModelConfig":
        """~760 M parameters — high quality / multi-GPU."""
        return cls(
            d_model=1280,
            n_heads=20,
            n_layers=36,
            d_ff=5120,
            max_seq_len=4096,
            model_type="large",
        )

    @classmethod
    def xl(cls) -> "ModelConfig":
        """~1.5 B parameters — near state-of-the-art code quality."""
        return cls(
            d_model=1600,
            n_heads=25,
            n_layers=48,
            d_ff=6400,
            max_seq_len=4096,
            vocab_size=50_000,
            model_type="xl",
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        data = json.loads(Path(path).read_text())
        return cls(**data)


@dataclass
class TrainingConfig:
    """Training hyper-parameters and logistics."""

    # Optimisation
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    gradient_clip: float = 1.0
    warmup_steps: int = 2_000
    lr_schedule: Literal["cosine", "linear", "constant"] = "cosine"
    min_lr_ratio: float = 0.1       # final LR = min_lr_ratio * peak LR

    # Batch
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = field(init=False)

    # Training length
    max_steps: int = 100_000
    save_every_n_steps: int = 1_000
    eval_every_n_steps: int = 500
    log_every_n_steps: int = 50

    # Data
    train_split: float = 0.95
    val_split: float = 0.04
    test_split: float = 0.01
    num_workers: int = 4

    # Infrastructure
    mixed_precision: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    use_xla: bool = True
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    def __post_init__(self) -> None:
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        data = json.loads(Path(path).read_text())
        data.pop("effective_batch_size", None)
        return cls(**data)


@dataclass
class DataConfig:
    """Data collection and preprocessing settings."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    tokenizer_dir: str = "data/tokenizer"

    # GitHub scraping
    github_token: str = ""
    max_repos: int = 50_000
    min_stars: int = 10
    languages: list[str] = field(default_factory=lambda: ["Go"])

    # Go-version-specific data sources
    go_versions: list[str] = field(
        default_factory=lambda: [
            "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7",
            "1.8", "1.9", "1.10", "1.11", "1.12", "1.13", "1.14", "1.15",
            "1.16", "1.17", "1.18", "1.19", "1.20", "1.21", "1.22", "1.23",
            "1.24",
        ]
    )

    # Key framework/library targets (Go ecosystem)
    focus_topics: list[str] = field(
        default_factory=lambda: [
            "fiber",        # gofiber/fiber - fast HTTP framework
            "cobra",        # spf13/cobra - CLI framework
            "gin",          # gin-gonic/gin - HTTP web framework
            "echo",         # labstack/echo
            "chi",          # go-chi/chi - lightweight router
            "gorm",         # go-gorm/gorm - ORM
            "grpc",         # grpc/grpc-go
            "protobuf",     # google.golang.org/protobuf
            "urfave-cli",   # urfave/cli
            "viper",        # spf13/viper - config
            "zap",          # uber-go/zap - logging
            "testify",      # stretchr/testify - testing
            "sqlx",         # jmoiron/sqlx
            "redis",        # go-redis/redis
            "prometheus",   # prometheus/client_golang
            "kubernetes",   # kubernetes/kubernetes (Go internals)
            "docker",       # moby/moby (Go internals)
        ]
    )

    # Preprocessing
    min_file_chars: int = 100
    max_file_chars: int = 100_000
    dedup_threshold: float = 0.8    # MinHash similarity for deduplication

    # Tokenizer
    vocab_size: int = 32_000
    tokenizer_type: Literal["bpe", "unigram"] = "bpe"
    special_tokens: list[str] = field(
        default_factory=lambda: [
            "<pad>", "<bos>", "<eos>", "<unk>",
            "<go_file>", "</go_file>",
            "<go_func>", "</go_func>",
            "<go_type>", "</go_type>",
            "<go_pkg>", "</go_pkg>",
            "<go_comment>", "</go_comment>",
            "<go_test>", "</go_test>",
            "<go_version>",
            "<task:complete>", "<task:generate>", "<task:review>",
            "<task:explain>", "<task:fix>", "<task:optimize>",
        ]
    )
