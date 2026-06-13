
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterator

import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()


SEARCH_QUERIES = [
    # Core language & ecosystem
    "language:Go stars:>10",
    # Frameworks
    "topic:fiber language:Go",
    "topic:gin language:Go",
    "topic:echo language:Go",
    "topic:chi language:Go",
    "topic:gorilla language:Go",
    # CLI
    "topic:cobra language:Go",
    "topic:cli language:Go",
    # Data & storage
    "topic:gorm language:Go",
    "topic:sql language:Go",
    "topic:redis language:Go",
    "topic:mongodb language:Go",
    # Distributed systems
    "topic:grpc language:Go",
    "topic:kubernetes language:Go",
    "topic:microservices language:Go",
    # Observability
    "topic:prometheus language:Go",
    "topic:opentelemetry language:Go",
    # Testing
    "topic:testing language:Go",
    "topic:benchmarks language:Go",
    # Go version-specific idioms (generics → Go 1.18+)
    "topic:generics language:Go",
]

# Official Go documentation and specification sources
OFFICIAL_SOURCES = [
    "https://raw.githubusercontent.com/golang/go/master/doc/go_spec.html",
    "https://raw.githubusercontent.com/golang/go/master/src/",
]


class GoDataCollector:
    """Collects Go source files from GitHub and official sources."""

    GITHUB_API = "https://api.github.com"

    def __init__(
        self,
        token: str,
        output_dir: str | Path = "data/raw",
        min_stars: int = 10,
        max_repos: int = 50_000,
    ):
        if not token:
            raise ValueError(
                "A GitHub personal access token is required. "
                "Set GITHUB_TOKEN env var or pass token= explicitly."
            )
        self.token = token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_stars = min_stars
        self.max_repos = max_repos
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_all(self, queries: list[str] | None = None) -> int:
        """Run the full collection pipeline. Returns total files downloaded."""
        queries = queries or SEARCH_QUERIES
        total_files = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Collecting repos…", total=self.max_repos)
            seen_repos: set[str] = set()

            for query in queries:
                for repo in self._search_repos(query):
                    full_name = repo["full_name"]
                    if full_name in seen_repos:
                        continue
                    seen_repos.add(full_name)

                    files = self._collect_repo(repo)
                    total_files += files
                    progress.advance(task)

                    if len(seen_repos) >= self.max_repos:
                        break
                if len(seen_repos) >= self.max_repos:
                    break

        console.print(f"[green]Collected {total_files} .go files from {len(seen_repos)} repos")
        return total_files

    def collect_stdlib(self, go_root: str | None = None) -> int:
        """Collect Go standard library source from local GOROOT or GitHub."""
        stdlib_dir = self.output_dir / "stdlib"
        stdlib_dir.mkdir(exist_ok=True)

        if go_root and Path(go_root).exists():
            return self._collect_local_stdlib(Path(go_root), stdlib_dir)

        return self._collect_github_stdlib(stdlib_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search_repos(self, query: str) -> Iterator[dict]:
        """Paginate GitHub search API for repositories."""
        page = 1
        while True:
            resp = self._get(
                f"{self.GITHUB_API}/search/repositories",
                params={
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 100,
                    "page": page,
                },
            )
            items = resp.get("items", [])
            if not items:
                break
            for item in items:
                if item.get("stargazers_count", 0) >= self.min_stars:
                    yield item
            if len(items) < 100:
                break
            page += 1

    def _collect_repo(self, repo: dict) -> int:
        """Download .go files from a single repo. Returns file count."""
        owner, name = repo["owner"]["login"], repo["name"]
        repo_dir = self.output_dir / "repos" / owner / name
        repo_dir.mkdir(parents=True, exist_ok=True)

        # Skip already-collected repos
        meta_file = repo_dir / "meta.json"
        if meta_file.exists():
            return 0

        try:
            files_collected = 0
            for go_file in self._iter_go_files(owner, name):
                dest = repo_dir / go_file["path"].replace("/", "__")
                dest.write_bytes(go_file["content"])
                files_collected += 1

            meta = {
                "full_name": repo["full_name"],
                "stars": repo.get("stargazers_count", 0),
                "license": repo.get("license", {}).get("spdx_id", "unknown"),
                "topics": repo.get("topics", []),
                "default_branch": repo.get("default_branch", "main"),
                "files": files_collected,
            }
            meta_file.write_text(json.dumps(meta, indent=2))
            return files_collected

        except Exception as exc:
            console.print(f"[yellow]Warning: skipping {owner}/{name}: {exc}")
            return 0

    def _iter_go_files(self, owner: str, name: str) -> Iterator[dict]:
        """Yield dicts with path and content for every .go file in a repo."""
        try:
            tree = self._get(
                f"{self.GITHUB_API}/repos/{owner}/{name}/git/trees/HEAD",
                params={"recursive": "1"},
            )
        except requests.HTTPError:
            return

        for item in tree.get("tree", []):
            if item["type"] == "blob" and item["path"].endswith(".go"):
                try:
                    raw_url = (
                        f"https://raw.githubusercontent.com/{owner}/{name}/HEAD/{item['path']}"
                    )
                    resp = self.session.get(raw_url, timeout=30)
                    resp.raise_for_status()
                    yield {"path": item["path"], "content": resp.content}
                except Exception:
                    continue

    def _collect_local_stdlib(self, go_root: Path, dest: Path) -> int:
        src = go_root / "src"
        count = 0
        for go_file in src.rglob("*.go"):
            rel = go_file.relative_to(src)
            target = dest / str(rel).replace("/", "__")
            target.write_bytes(go_file.read_bytes())
            count += 1
        return count

    def _collect_github_stdlib(self, dest: Path) -> int:
        """Collect stdlib directly from golang/go on GitHub."""
        count = 0
        for item in self._iter_go_files("golang", "go"):
            if item["path"].startswith("src/"):
                target = dest / item["path"].replace("/", "__")
                target.write_bytes(item["content"])
                count += 1
        return count

    def _get(self, url: str, params: dict | None = None) -> dict:
        """GET with automatic rate-limit backoff."""
        while True:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 429 or (
                resp.status_code == 403
                and "rate limit" in resp.text.lower()
            ):
                reset_at = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(0, reset_at - int(time.time())) + 5
                console.print(f"[yellow]Rate limited. Sleeping {wait}s…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
