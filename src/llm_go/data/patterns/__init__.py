"""
Real-world Go pattern training data extracted from the Medical-App-Core project.
Each submodule generates annotated training examples for a specific pattern category.
"""

from llm_go.data.patterns.fiber_patterns import FiberPatternGenerator
from llm_go.data.patterns.gorm_patterns import GormPatternGenerator
from llm_go.data.patterns.service_patterns import ServicePatternGenerator
from llm_go.data.patterns.auth_patterns import AuthPatternGenerator
from llm_go.data.patterns.test_patterns import TestPatternGenerator
from llm_go.data.patterns.docker_patterns import DockerPatternGenerator
from llm_go.data.patterns.registry import PatternRegistry

__all__ = [
    "FiberPatternGenerator",
    "GormPatternGenerator",
    "ServicePatternGenerator",
    "AuthPatternGenerator",
    "TestPatternGenerator",
    "DockerPatternGenerator",
    "PatternRegistry",
]
