"""
Tester agent.

Generates tests, executes them against the generated code, and produces failure reports.
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.code_extractor import extract_code
from src.llm.base import LLMResponse
from src.orchestration.artifacts import Artifact
from src.orchestration.context import SharedContext


class Tester(BaseAgent):
    """
    Tests the generated code.

    Responsibilities:
    1. Analyze requirements and design.
    2. Create test cases.
    3. Generate a test script.
    4. Return test artifact for execution by the TestExecutor.
    5. When test results are available, produce a failure report.
    """

    agent_name = "tester"
    agent_role = "Software Tester"
    artifact_type = "tests"
    prompt_file = "system.txt"

    def execute(self, context: SharedContext) -> Artifact:
        """
        Execute the tester agent.

        Uses the review prompt if we're reviewing existing test results.
        """
        if context.test_results:
            self.prompt_file = "review.txt"
        else:
            self.prompt_file = "system.txt"

        return super().execute(context)

    def parse_response(self, response: LLMResponse, context: SharedContext) -> Artifact:
        """Parse the LLM response into a test artifact."""
        test_code = extract_code(response.content)

        # Determine artifact type based on context
        if context.test_results:
            artifact_type = "test_failure_report"
        else:
            artifact_type = "tests"

        return Artifact(
            type=artifact_type,
            content=test_code if artifact_type == "tests" else response.content,
            language="python",
            created_by=self.agent_name,
            metadata={
                "model": response.model,
                "latency_ms": response.latency_ms,
                "tokens": response.total_tokens,
            },
        )

    def update_context(self, context: SharedContext, artifact: Artifact) -> None:
        """Store test output in the shared context."""
        if artifact.type == "tests":
            context.tests = artifact.content
        elif artifact.type == "test_failure_report":
            context.failure_reports.append(artifact.content)

