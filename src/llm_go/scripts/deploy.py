"""CLI: push a checkpoint to Hugging Face Hub."""

import os
import click
from llm_go.deployment.hf_uploader import HuggingFaceUploader


@click.command()
@click.option("--ckpt-dir",  required=True, help="Checkpoint directory to upload")
@click.option("--tok-dir",   required=True, help="Tokenizer directory")
@click.option("--repo-id",   required=True, help="HF repo, e.g. myorg/llm-go-350m")
@click.option("--token",     default=lambda: os.environ.get("HF_TOKEN", ""))
@click.option("--private/--public", default=False)
@click.option("--message",   default="Upload GoLLM checkpoint", show_default=True)
def main(ckpt_dir, tok_dir, repo_id, token, private, message):
    """Deploy GoLLM to Hugging Face Hub."""
    uploader = HuggingFaceUploader(repo_id=repo_id, hf_token=token, private=private)
    url = uploader.upload(
        checkpoint_dir=ckpt_dir,
        tokenizer_dir=tok_dir,
        commit_message=message,
    )
    click.echo(f"Model live at: {url}")
