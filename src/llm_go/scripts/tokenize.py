"""CLI: train the Go BPE tokenizer."""

import click
from pathlib import Path
from llm_go.tokenizer.go_tokenizer import GoTokenizer


@click.command()
@click.option("--raw-dir",    default="data/raw",       show_default=True)
@click.option("--out-dir",    default="data/tokenizer", show_default=True)
@click.option("--vocab-size", default=32_000,           show_default=True)
def main(raw_dir, out_dir, vocab_size):
    """Train a BPE tokenizer on collected Go source files."""
    files = [str(p) for p in Path(raw_dir).rglob("*.go")]
    click.echo(f"Training tokenizer on {len(files):,} files…")
    GoTokenizer.train(files=files, vocab_size=vocab_size, save_dir=out_dir)
    click.echo(f"Tokenizer saved → {out_dir}")
