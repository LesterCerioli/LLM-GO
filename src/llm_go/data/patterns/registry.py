
from __future__ import annotations

import json
from pathlib import Path

from llm_go.data.patterns.fiber_patterns import FiberPatternGenerator
from llm_go.data.patterns.gorm_patterns import GormPatternGenerator
from llm_go.data.patterns.service_patterns import ServicePatternGenerator
from llm_go.data.patterns.auth_patterns import AuthPatternGenerator
from llm_go.data.patterns.test_patterns import TestPatternGenerator
from llm_go.data.patterns.docker_patterns import DockerPatternGenerator


class PatternRegistry:
    """
    Central registry for all real-world Go pattern generators.

    All patterns are extracted from Medical-App-Core (Fiber + GORM + JWT + RabbitMQ)
    and rendered as structured training examples with <go_file> / <go_version> tags.
    """

    GENERATORS = [
        ("fiber",   FiberPatternGenerator),
        ("gorm",    GormPatternGenerator),
        ("service", ServicePatternGenerator),
        ("auth",    AuthPatternGenerator),
        ("test",    TestPatternGenerator),
        ("docker",  DockerPatternGenerator),
    ]

    def __init__(self):
        self._generators = [(name, cls()) for name, cls in self.GENERATORS]

    def all_examples(self) -> list[str]:
        """Return every training example from every registered generator."""
        examples: list[str] = []
        for _, gen in self._generators:
            examples.extend(gen.all_examples())
        return examples

    def examples_by_category(self) -> dict[str, list[str]]:
        """Return examples grouped by category name."""
        return {name: gen.all_examples() for name, gen in self._generators}

    def count(self) -> dict[str, int]:
        """Return example count per category plus total."""
        counts = {name: len(gen.all_examples()) for name, gen in self._generators}
        counts["total"] = sum(counts.values())
        return counts

    def save_to_file(self, path: str | Path, pretty: bool = False) -> None:
        """
        Save all examples to a JSONL file for inspection or offline analysis.
        Each line is: {"category": "...", "text": "..."}
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for name, gen in self._generators:
                for ex in gen.all_examples():
                    record = {"category": name, "text": ex}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {self.count()['total']} examples → {path}")

    def summary(self) -> str:
        counts = self.count()
        lines = ["Pattern Registry Summary", "=" * 40]
        for name, n in counts.items():
            if name != "total":
                lines.append(f"  {name:<12} {n:>4} examples")
        lines.append("-" * 40)
        lines.append(f"  {'TOTAL':<12} {counts['total']:>4} examples")
        return "\n".join(lines)
