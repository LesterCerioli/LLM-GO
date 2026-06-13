"""CLI: collect Go source data from GitHub."""

import os
import click
from llm_go.data.collector import GoDataCollector


@click.command()
@click.option("--token",    default=lambda: os.environ.get("GITHUB_TOKEN", ""), help="GitHub token")
@click.option("--out-dir",  default="data/raw",   show_default=True)
@click.option("--min-stars", default=10,           show_default=True)
@click.option("--max-repos", default=50_000,       show_default=True)
@click.option("--stdlib/--no-stdlib", default=True, help="Also collect Go stdlib")
@click.option("--go-root",  default=None,          help="Local GOROOT (optional)")
def main(token, out_dir, min_stars, max_repos, stdlib, go_root):
    """Collect Go source code from GitHub and the Go standard library."""
    collector = GoDataCollector(
        token=token, output_dir=out_dir, min_stars=min_stars, max_repos=max_repos
    )
    collector.collect_all()
    if stdlib:
        collector.collect_stdlib(go_root)
