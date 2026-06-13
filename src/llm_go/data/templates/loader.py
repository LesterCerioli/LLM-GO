"""Loads canonical Go template files from disk and exposes them as training texts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

_TEMPLATES_ROOT = Path(__file__).parent


class TemplateLoader:
    """Walks the go_project template tree and yields each file as a training string.

    Each yielded string is wrapped with structural tags so the tokenizer
    can inject <go_file>, <go_func>, etc. just like real corpus files.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root) if root else _TEMPLATES_ROOT / "go_project"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def all_texts(self) -> list[str]:
        return list(self._iter_texts())

    def _iter_texts(self) -> Iterator[str]:
        for path in sorted(self._root.rglob("*")):
            if path.is_file():
                yield self._format(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format(self, path: Path) -> str:
        rel = path.relative_to(self._root)
        content = path.read_text(encoding="utf-8")
        pkg = _infer_package(path)
        return (
            f"<go_file>{rel}</go_file>\n"
            f"<go_pkg>{pkg}</go_pkg>\n"
            f"<go_version>1.24</go_version>\n"
            f"{content}"
        )


def _infer_package(path: Path) -> str:
    """Derive a Go package name from the file path (last non-extension component)."""
    if path.suffix == ".go":
        return path.parent.name
    if path.name == "go.mod":
        try:
            for line in path.read_text().splitlines():
                if line.startswith("module "):
                    return line.split()[1]
        except OSError:
            pass
    return os.path.splitext(path.name)[0]
