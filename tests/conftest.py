"""Shared pytest fixtures used by all test modules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from llm_go.config import ModelConfig, DataConfig, TrainingConfig
from llm_go.tokenizer.go_tokenizer import GoTokenizer


# ---------------------------------------------------------------------------
# Tiny model config — avoids OOM in CI (CPU-only, tiny dimensions)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=1024,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ffn_multiplier=2,
        max_seq_len=128,
        dropout=0.0,
    )


@pytest.fixture(scope="session")
def small_data_config() -> DataConfig:
    return DataConfig(
        seq_len=128,
        batch_size=2,
        val_split=0.1,
        test_split=0.1,
    )


@pytest.fixture(scope="session")
def default_training_config() -> TrainingConfig:
    return TrainingConfig(
        max_steps=10,
        warmup_steps=2,
        lr=1e-4,
        grad_accum_steps=1,
        save_every=5,
        eval_every=5,
        precision="float32",
    )


# ---------------------------------------------------------------------------
# Temporary directories
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir() -> Path:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def tokenizer_dir(tmp_dir: Path) -> Path:
    return tmp_dir / "tokenizer"


# ---------------------------------------------------------------------------
# Trained tokenizer (session-scoped so we only train it once)
# ---------------------------------------------------------------------------

_GO_CORPUS = [
    "package main\n\nimport \"fmt\"\n\nfunc main() { fmt.Println(\"hello\") }",
    "package server\n\nimport \"github.com/gofiber/fiber/v2\"\n\nfunc New() *fiber.App { return fiber.New() }",
    "package service\n\ntype UserService interface { GetByID(id string) (*User, error) }",
    "package repository\n\nimport \"gorm.io/gorm\"\n\ntype UserRepo struct { db *gorm.DB }",
    "package main\n\nimport \"github.com/spf13/cobra\"\n\nvar rootCmd = &cobra.Command{Use: \"app\"}",
]


@pytest.fixture(scope="session")
def trained_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> GoTokenizer:
    tok_dir = tmp_path_factory.mktemp("tokenizer")
    tok = GoTokenizer(vocab_size=1024)
    tok.train(_GO_CORPUS * 10)
    tok.save(str(tok_dir))
    return tok


@pytest.fixture(scope="session")
def go_corpus() -> list[str]:
    return _GO_CORPUS


# ---------------------------------------------------------------------------
# Sample Go source snippets
# ---------------------------------------------------------------------------

@pytest.fixture
def fiber_controller_snippet() -> str:
    return """\
package handler

import "github.com/gofiber/fiber/v2"

type UserHandler struct{}

func (h *UserHandler) GetUser(c *fiber.Ctx) error {
    id := c.Params("id")
    return c.JSON(fiber.Map{"id": id})
}
"""


@pytest.fixture
def gorm_entity_snippet() -> str:
    return """\
package model

import (
    "github.com/google/uuid"
    "gorm.io/gorm"
)

type User struct {
    gorm.Model
    ID    uuid.UUID `gorm:"type:uuid;primaryKey"`
    Name  string    `gorm:"not null"`
    Email string    `gorm:"uniqueIndex;not null"`
}
"""
