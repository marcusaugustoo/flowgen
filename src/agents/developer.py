"""
Developer agent.

Generates Python code based on requirements and design.
Also handles refinement when receiving failure reports.
"""

from __future__ import annotations

from typing import Any, Optional

from src.agents.base_agent import BaseAgent
from src.agents.code_extractor import extract_code
from src.llm.base import LLMResponse
from src.orchestration.artifacts import Artifact
from src.orchestration.context import SharedContext


class Developer(BaseAgent):
    """
    Generates Python code based on requirements and design.

    In refinement mode, receives failure reports and produces corrected code.
    """

    agent_name = "developer"
    agent_role = "Software Developer"
    artifact_type = "code"
    prompt_file = "system.txt"

    def execute(self, context: SharedContext) -> Artifact:
        """
        Execute the developer agent.

        Uses the refinement prompt if failure reports are present.
        """
        # Determine which prompt to use
        if context.failure_reports:
            self.prompt_file = "refinement.txt"
        else:
            self.prompt_file = "system.txt"

        return super().execute(context)

    def parse_response(self, response: LLMResponse, context: SharedContext) -> Artifact:
        """Parse the LLM response, extracting code from markdown blocks if present."""
        code = extract_code(response.content)

        return Artifact(
            type=self.artifact_type,
            content=code,
            language="python",
            created_by=self.agent_name,
            metadata={
                "model": response.model,
                "latency_ms": response.latency_ms,
                "tokens": response.total_tokens,
                "is_refinement": bool(context.failure_reports),
            },
        )

    def update_context(self, context: SharedContext, artifact: Artifact) -> None:
        """Store generated code in the shared context."""
        context.code = artifact.content

