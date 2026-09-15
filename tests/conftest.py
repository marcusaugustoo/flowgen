"""
Shared test fixtures for FlowGen tests.
"""

import sys
from pathlib import Path

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarks.base import Problem
from src.config import ExperimentConfig
from src.llm.mock_provider import MockLLMProvider
from src.orchestration.context import SharedContext


@pytest.fixture
def mock_llm():
    """Create a MockLLMProvider with default response."""
    return MockLLMProvider(
        default_response="Mock response content.",
        latency_ms=5.0,
    )


@pytest.fixture
def mock_llm_with_code():
    """Create a MockLLMProvider that returns Python code."""
    return MockLLMProvider(
        default_response=(
            "```python\n"
            "def has_close_elements(numbers, threshold):\n"
            "    for i in range(len(numbers)):\n"
            "        for j in range(i + 1, len(numbers)):\n"
            "            if abs(numbers[i] - numbers[j]) < threshold:\n"
            "                return True\n"
            "    return False\n"
            "```"
        ),
        latency_ms=5.0,
    )


@pytest.fixture
def sample_config():
    """Create a default ExperimentConfig for testing."""
    return ExperimentConfig.load(
        overrides={
            "llm": {"provider": "mock"},
            "experiment": {"name": "test_experiment", "runs": 1},
            "benchmark": {"name": "humaneval"},
            "pipeline": {"type": "waterfall"},
        }
    )


@pytest.fixture
def raw_config():
    """Create a raw process ExperimentConfig for testing."""
    return ExperimentConfig.load(
        overrides={
            "llm": {"provider": "mock"},
            "experiment": {"name": "test_raw", "runs": 1},
            "pipeline": {"type": "raw"},
            "self_refinement": {"enabled": False, "iterations": 0},
            "agents": {
                "requirement_engineer": False,
                "architect": False,
                "developer": False,
                "tester": False,
                "scrum_master": False,
            },
        }
    )


@pytest.fixture
def sample_problem():
    """Create a sample HumanEval problem."""
    return Problem(
        task_id="HumanEval/0",
        prompt=(
            "from typing import List\n\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            '    """ Check if in given list of numbers, are any two numbers closer to each other than\n'
            "    given threshold.\n"
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
            "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
            "    True\n"
            '    """\n'
        ),
        entry_point="has_close_elements",
        test=(
            "\ndef check(candidate):\n"
            "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
            "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n"
            "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n"
            "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n"
            "    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0], 2.0) == True\n\n"
            "check(has_close_elements)\n"
        ),
        canonical_solution=(
            "    for idx, elem in enumerate(numbers):\n"
            "        for idx2, elem2 in enumerate(numbers):\n"
            "            if idx != idx2:\n"
            "                distance = abs(elem - elem2)\n"
            "                if distance < threshold:\n"
            "                    return True\n"
            "    return False\n"
        ),
    )


@pytest.fixture
def sample_context(sample_problem):
    """Create a SharedContext initialized with a sample problem."""
    return SharedContext(
        problem=sample_problem.prompt,
        problem_id=sample_problem.task_id,
        entry_point=sample_problem.entry_point,
        canonical_tests=sample_problem.test,
    )


@pytest.fixture
def sample_custom_problem():
    """Create a sample custom task Problem for testing."""
    return Problem(
        task_id="Custom/Test",
        prompt="Implemente uma função chamada add(a, b) que retorne a soma.",
        entry_point="add",
        test=(
            "def check(candidate):\n"
            "    assert candidate(1, 2) == 3\n"
            "    assert candidate(0, 0) == 0\n"
            "    assert candidate(-1, 1) == 0\n"
            "    assert candidate(100, 200) == 300\n"
            "\ncheck(add)\n"
        ),
        canonical_solution="",
    )
