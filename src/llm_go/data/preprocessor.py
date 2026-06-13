"""
Data preprocessing pipeline.

Steps:
  1. Quality filtering (size, encoding, syntax check)
  2. Deduplication via MinHash LSH
  3. PII scrubbing (emails, tokens, secrets)
  4. Tokenisation and packing into fixed-length sequences
  5. Train / val / test split and TFRecord serialisation
"""

from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.progress import track

console = Console()

# ---- Patterns to scrub -------------------------------------------------------
_EMAIL_RE   = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_TOKEN_RE   = re.compile(
    r"(?i)(ghp_|gho_|github_pat_|sk-|xox[baprs]-)[A-Za-z0-9_\-]{10,}"
)
_SECRET_RE  = re.compile(
    r"(?i)(password|passwd|secret|api.?key|access.?token)\s*[:=]\s*['\"]?[^\s'\"]{8,}"
)
_IP_RE      = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# ---- MinHash config ----------------------------------------------------------
_MINHASH_SHINGLES = 5       # n-gram size for shingling
_MINHASH_PERMS    = 128     # number of hash permutations
_LARGE_PRIME      = (1 << 31) - 1
_MINHASH_A = np.array([
    1_543_763 + i * 2_654_435_761 for i in range(_MINHASH_PERMS)
], dtype=np.int64)
_MINHASH_B = np.array([
    3_267_000_013 + i * 2_246_822_519 for i in range(_MINHASH_PERMS)
], dtype=np.int64)


class GoPreprocessor:
    """Preprocess raw .go files into tokenised, packed TFRecord shards."""

    def __init__(
        self,
        tokenizer,                   # GoTokenizer instance
        raw_dir: str | Path = "data/raw",
        out_dir: str | Path = "data/processed",
        seq_len: int = 2048,
        shard_size: int = 10_000,    # sequences per TFRecord shard
        train_frac: float = 0.95,
        val_frac:   float = 0.04,
        min_chars:  int = 100,
        max_chars:  int = 100_000,
        dedup_threshold: float = 0.8,
    ):
        self.tokenizer = tokenizer
        self.raw_dir   = Path(raw_dir)
        self.out_dir   = Path(out_dir)
        self.seq_len   = seq_len
        self.shard_size     = shard_size
        self.train_frac     = train_frac
        self.val_frac       = val_frac
        self.min_chars      = min_chars
        self.max_chars      = max_chars
        self.dedup_threshold = dedup_threshold
        self._seen_hashes: set[tuple] = set()

        for split in ("train", "val", "test"):
            (self.out_dir / split).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> dict[str, int]:
        """Full pipeline. Returns sequence counts per split."""
        import itertools
        from llm_go.data.go_best_practices import GoProjectTemplates
        from llm_go.data.patterns.registry import PatternRegistry

        # ── Synthetic best-practice examples (cmd/ layout, idioms)
        # Repeated 5× so the model learns layout conventions with high weight
        templates     = GoProjectTemplates()
        layout_texts  = templates.all_examples() * 5

        # ── Real-world patterns from Medical-App-Core
        # (Fiber + GORM + JWT + RabbitMQ + Docker) — repeated 3×
        registry      = PatternRegistry()
        pattern_texts = registry.all_examples() * 3

        console.print(f"  [cyan]Layout examples:  {len(layout_texts):,}")
        console.print(f"  [cyan]Pattern examples: {len(pattern_texts):,}")
        console.print(registry.summary())

        synthetic_texts = layout_texts + pattern_texts
        synthetic_ids   = (self.tokenizer.encode(ex) for ex in synthetic_texts)

        real_ids = self._tokenise_files(self._iter_filtered_files())

        combined = itertools.chain(synthetic_ids, real_ids)
        packed   = list(self._pack_sequences(combined))

        np.random.shuffle(packed)
        n       = len(packed)
        n_train = int(n * self.train_frac)
        n_val   = int(n * self.val_frac)

        splits = {
            "train": packed[:n_train],
            "val":   packed[n_train : n_train + n_val],
            "test":  packed[n_train + n_val :],
        }

        counts: dict[str, int] = {}
        for split, seqs in splits.items():
            self._write_tfrecords(seqs, self.out_dir / split, split)
            counts[split] = len(seqs)

        console.print(f"[green]Done. Splits: {counts}")
        return counts

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _iter_filtered_files(self) -> Iterator[tuple[str, str]]:
        """Yield (path, cleaned_source) for files that pass all filters."""
        all_files = list(self.raw_dir.rglob("*.go"))
        console.print(f"Found {len(all_files)} raw .go files")

        for path in track(all_files, description="Filtering…"):
            try:
                raw = path.read_bytes()
                source = raw.decode("utf-8", errors="ignore")
            except OSError:
                continue

            if not self.min_chars <= len(source) <= self.max_chars:
                continue
            if not self._is_valid_go(source):
                continue

            source = self._scrub_pii(source)

            sig = self._minhash(source)
            if self._is_duplicate(sig):
                continue
            self._seen_hashes.add(sig)

            yield str(path), source

    def _is_valid_go(self, source: str) -> bool:
        """Heuristic syntax check — must have package declaration."""
        return bool(re.search(r"^package\s+\w+", source, re.MULTILINE))

    def _scrub_pii(self, source: str) -> str:
        source = _EMAIL_RE.sub("<EMAIL>", source)
        source = _TOKEN_RE.sub(r"\1<REDACTED>", source)
        source = _SECRET_RE.sub(r"\1: <REDACTED>", source)
        return source

    # ------------------------------------------------------------------
    # MinHash deduplication
    # ------------------------------------------------------------------

    def _shingles(self, text: str) -> set[int]:
        words = text.split()
        return {
            hash(" ".join(words[i : i + _MINHASH_SHINGLES]))
            for i in range(len(words) - _MINHASH_SHINGLES + 1)
        }

    def _minhash(self, text: str) -> tuple[int, ...]:
        shingles = np.array(list(self._shingles(text)), dtype=np.int64)
        if len(shingles) == 0:
            return tuple(np.zeros(_MINHASH_PERMS, dtype=np.int64).tolist())
        # MinHash: min over all shingles of (a*h + b) % p
        hvals = (
            np.outer(shingles, _MINHASH_A) + _MINHASH_B[np.newaxis, :]
        ) % _LARGE_PRIME                                          # [S, P]
        return tuple(hvals.min(axis=0).tolist())

    def _is_duplicate(self, sig: tuple[int, ...]) -> bool:
        if not self._seen_hashes:
            return False
        # Approximate Jaccard via banded LSH (b bands of r rows, b*r = P)
        b, r = 32, 4          # 32 bands × 4 rows = 128 = _MINHASH_PERMS
        for band in range(b):
            band_sig = sig[band * r : (band + 1) * r]
            band_hash = hashlib.md5(struct.pack(f"{r}q", *band_sig)).digest()
            # Check against all previously seen band hashes (simplified)
            # In production, use datasketch.MinHashLSH for scalability
            for seen in self._seen_hashes:
                seen_band = seen[band * r : (band + 1) * r]
                seen_hash = hashlib.md5(struct.pack(f"{r}q", *seen_band)).digest()
                if band_hash == seen_hash:
                    return True
        return False

    # ------------------------------------------------------------------
    # Tokenisation & packing
    # ------------------------------------------------------------------

    def _tokenise_files(
        self, files: Iterator[tuple[str, str]]
    ) -> Iterator[list[int]]:
        for path, source in files:
            # Extract Go version from directory name if available
            version = self._infer_version(path)
            ids = self.tokenizer.encode_go_file(source, version=version)
            yield ids

    def _infer_version(self, path: str) -> str:
        """Try to extract go version from go.mod in neighbouring files."""
        p = Path(path)
        gomod = p.parent / "go.mod"
        if gomod.exists():
            m = re.search(r"^go\s+(\d+\.\d+)", gomod.read_text(), re.MULTILINE)
            if m:
                return m.group(1)
        return ""

    def _pack_sequences(self, token_stream: Iterator[list[int]]) -> Iterator[list[int]]:
        """Concatenate token streams and yield fixed-length windows (no padding)."""
        buffer: list[int] = []
        for ids in token_stream:
            buffer.extend(ids)
            while len(buffer) >= self.seq_len + 1:
                yield buffer[: self.seq_len + 1]   # +1 for labels shift
                buffer = buffer[self.seq_len + 1:]
        # Drop remainder (avoids padding in packed pre-training)

    # ------------------------------------------------------------------
    # TFRecord I/O
    # ------------------------------------------------------------------

    def _write_tfrecords(
        self, sequences: list[list[int]], dest: Path, split: str
    ) -> None:
        n_shards = max(1, len(sequences) // self.shard_size)
        for shard_idx in range(n_shards):
            shard_path = dest / f"{split}-{shard_idx:05d}.tfrecord"
            with tf.io.TFRecordWriter(str(shard_path)) as writer:
                start = shard_idx * self.shard_size
                end   = start + self.shard_size
                for seq in sequences[start:end]:
                    feature = {
                        "input_ids": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=seq)
                        )
                    }
                    writer.write(
                        tf.train.Example(
                            features=tf.train.Features(feature=feature)
                        ).SerializeToString()
                    )
        console.print(f"  [{split}] wrote {len(sequences)} sequences → {n_shards} shards")

    @staticmethod
    def load_dataset(
        tfrecord_dir: str | Path,
        seq_len: int = 2048,
        batch_size: int = 32,
        shuffle_buffer: int = 10_000,
    ) -> tf.data.Dataset:
        """Load a preprocessed split as a batched tf.data.Dataset."""
        paths = sorted(Path(tfrecord_dir).glob("*.tfrecord"))
        raw = tf.data.TFRecordDataset(
            [str(p) for p in paths],
            num_parallel_reads=tf.data.AUTOTUNE,
        )

        feature_spec = {
            "input_ids": tf.io.FixedLenSequenceFeature(
                [], tf.int64, allow_missing=True
            )
        }

        def _parse(serialised: tf.Tensor) -> dict[str, tf.Tensor]:
            parsed = tf.io.parse_single_example(serialised, feature_spec)
            ids = tf.cast(parsed["input_ids"], tf.int32)
            return {"input_ids": ids[:-1], "labels": ids[1:]}

        return (
            raw.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(shuffle_buffer)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
