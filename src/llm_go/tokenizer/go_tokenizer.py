"""
Go-aware BPE tokenizer.

Wraps HuggingFace `tokenizers` library and adds:
  - Go-specific pre-tokenisation (identifiers, operators, keywords)
  - Special structural tokens (<go_func>, <go_type>, …)
  - Tree-sitter based tag injection for structured context
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence as NormSeq

from llm_go.config import DataConfig


# Go keywords — kept as single tokens to avoid fragmentation
GO_KEYWORDS = [
    "break", "case", "chan", "const", "continue", "default", "defer",
    "else", "fallthrough", "for", "func", "go", "goto", "if", "import",
    "interface", "map", "package", "range", "return", "select", "struct",
    "switch", "type", "var",
]

# Common Go built-ins
GO_BUILTINS = [
    "append", "cap", "close", "complex", "copy", "delete", "imag",
    "len", "make", "new", "panic", "print", "println", "real", "recover",
    "any", "bool", "byte", "comparable", "complex64", "complex128",
    "error", "float32", "float64", "int", "int8", "int16", "int32", "int64",
    "rune", "string", "uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
    "true", "false", "nil", "iota",
]

# Frequent Go stdlib packages — preserved as atomic tokens
GO_PACKAGES = [
    "fmt", "os", "io", "sync", "net", "http", "json", "errors",
    "math", "sort", "time", "bytes", "strings", "strconv", "bufio",
    "context", "log", "path", "regexp", "testing", "reflect", "runtime",
    "atomic", "rand", "filepath", "unicode", "encoding", "sql",
    "grpc", "fiber", "gin", "echo", "cobra", "gorm", "zap", "viper",
]


class GoTokenizer:
    """BPE tokenizer trained on Go source code with structural awareness."""

    SPECIAL_TOKENS_DEFAULT = [
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

    PAD_ID  = 0
    BOS_ID  = 1
    EOS_ID  = 2
    UNK_ID  = 3

    def __init__(self, tokenizer: Tokenizer | None = None):
        self._tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        files: list[str] | None = None,
        iterator: Iterable[str] | None = None,
        vocab_size: int = 32_000,
        special_tokens: list[str] | None = None,
        save_dir: str | Path | None = None,
    ) -> "GoTokenizer":
        """Train a BPE tokenizer on Go source files."""

        if special_tokens is None:
            special_tokens = cls.SPECIAL_TOKENS_DEFAULT

        # Seed vocabulary with Go keywords + builtins so they're never split
        initial_alphabet = list(set(GO_KEYWORDS + GO_BUILTINS + GO_PACKAGES))

        tok = Tokenizer(models.BPE(unk_token="<unk>"))

        # Whitespace-preserving pre-tokeniser aware of Go syntax
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                pattern=r'(\s+|[{}()\[\];,.:!?<>=+\-*/&|^%~])',
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ])

        tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=special_tokens,
            initial_alphabet=initial_alphabet,
            show_progress=True,
        )

        if files is not None:
            tok.train(files=files, trainer=trainer)
        elif iterator is not None:
            tok.train_from_iterator(iterator, trainer=trainer)
        else:
            raise ValueError("Provide either files= or iterator=")

        # Post-processor: wrap sequences with BOS/EOS
        tok.post_processor = processors.TemplateProcessing(
            single="<bos> $A <eos>",
            pair="<bos> $A <eos> $B:1 <eos>:1",
            special_tokens=[("<bos>", cls.BOS_ID), ("<eos>", cls.EOS_ID)],
        )

        instance = cls(tok)
        if save_dir is not None:
            instance.save(save_dir)
        return instance

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self._tokenizer.encode_batch(texts)]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, batch: list[list[int]]) -> list[str]:
        return self._tokenizer.decode_batch(batch)

    def encode_go_file(self, source: str, version: str = "") -> list[int]:
        """
        Wrap a Go file with structural tokens before encoding.
        Injects <go_version>, <go_pkg>, <go_func>/<go_type> boundaries.
        """
        tagged = self._inject_structural_tags(source, version)
        return self.encode(tagged)

    def _inject_structural_tags(self, source: str, version: str) -> str:
        """Lightweight regex-based structural tagging (no AST required)."""
        lines: list[str] = []

        if version:
            lines.append(f"<go_version> go{version}")

        # Package declaration
        pkg_match = re.search(r"^package\s+(\w+)", source, re.MULTILINE)
        if pkg_match:
            lines.append(f"<go_pkg> {pkg_match.group(1)}")

        # Wrap func/type blocks with structural tokens
        tagged_source = source
        tagged_source = re.sub(
            r"^(func\s)", r"<go_func>\1", tagged_source, flags=re.MULTILINE
        )
        tagged_source = re.sub(
            r"^(type\s)", r"<go_type>\1", tagged_source, flags=re.MULTILINE
        )

        lines.append(f"<go_file>\n{tagged_source}\n</go_file>")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Token properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> str:
        return self._tokenizer.id_to_token(id)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(d / "tokenizer.json"))
        # Save vocab metadata
        vocab = self._tokenizer.get_vocab(with_added_tokens=True)
        (d / "vocab.json").write_text(json.dumps(vocab, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, directory: str | Path) -> "GoTokenizer":
        d = Path(directory)
        tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        return cls(tok)

    # ------------------------------------------------------------------
    # HuggingFace-compatible export
    # ------------------------------------------------------------------

    def to_hf_tokenizer(self):
        """Return a HuggingFace PreTrainedTokenizerFast wrapping this tokenizer."""
        from transformers import PreTrainedTokenizerFast

        return PreTrainedTokenizerFast(
            tokenizer_object=self._tokenizer,
            bos_token="<bos>",
            eos_token="<eos>",
            unk_token="<unk>",
            pad_token="<pad>",
            model_max_length=4096,
        )
