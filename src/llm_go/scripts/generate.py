"""CLI: interactive Go code generation."""

import click
import tensorflow as tf
from llm_go.model.transformer import GoLLM
from llm_go.tokenizer.go_tokenizer import GoTokenizer


@click.command()
@click.option("--model-dir",  required=True, help="Checkpoint directory")
@click.option("--tok-dir",    required=True, help="Tokenizer directory")
@click.option("--prompt",     default="",    help="Prompt text (leave empty for interactive)")
@click.option("--max-tokens", default=256,   show_default=True)
@click.option("--temperature", default=0.8,  show_default=True)
@click.option("--top-p",      default=0.95,  show_default=True)
@click.option("--top-k",      default=50,    show_default=True)
def main(model_dir, tok_dir, prompt, max_tokens, temperature, top_p, top_k):
    """Generate Go code with a trained GoLLM checkpoint."""
    tok   = GoTokenizer.load(tok_dir)
    model = GoLLM.from_pretrained(model_dir)

    def generate(text: str) -> str:
        ids    = tok.encode(text)
        out    = model.generate(
            tf.constant([ids], dtype=tf.int32),
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return tok.decode(out[0].numpy().tolist())

    if prompt:
        click.echo(generate(prompt))
    else:
        click.echo("GoLLM interactive shell. Type 'exit' to quit.\n")
        while True:
            try:
                text = click.prompt(">>> ")
            except (EOFError, KeyboardInterrupt):
                break
            if text.strip().lower() in {"exit", "quit"}:
                break
            click.echo(generate(text))
