"""CLI: evaluate a trained checkpoint."""

import click
from llm_go.model.transformer import GoLLM
from llm_go.tokenizer.go_tokenizer import GoTokenizer
from llm_go.data.dataset import GoDataset
from llm_go.evaluation.metrics import CodeEvaluator


@click.command()
@click.option("--model-dir", required=True)
@click.option("--tok-dir",   required=True)
@click.option("--data-dir",  default="data/processed", show_default=True)
@click.option("--batch-size", default=16, show_default=True)
@click.option("--max-batches", default=200, show_default=True)
def main(model_dir, tok_dir, data_dir, batch_size, max_batches):
    """Evaluate perplexity and syntax pass rate."""
    tok   = GoTokenizer.load(tok_dir)
    model = GoLLM.from_pretrained(model_dir)
    ds    = GoDataset(data_dir, batch_size=batch_size)

    evaluator = CodeEvaluator(model, tok)
    report    = evaluator.full_report(ds.val())

    for k, v in report.items():
        click.echo(f"{k}: {v:.4f}")
