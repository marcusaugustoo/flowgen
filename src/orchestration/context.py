"""
Shared context for inter-agent communication.

Stores all artifacts produced during a pipeline run.
Each agent receives only the context relevant to its task.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SharedContext:
    """
    Central context that accumulates artifacts as agents execute.

    Each field stores the output of a specific agent/phase.
    The orchestrator manages what context each agent receives.
    """

    # The original programming problem
    problem: str = ""
    problem_id: str = ""
    entry_point: str = ""
    canonical_tests: str = ""

    # Agent outputs
    requirements: str = ""
    design: str = ""
    code: str = ""
    tests: str = ""
    test_results: str = ""

    # Refinement data
    failure_reports: list[str] = field(default_factory=list)
    reviews: list[str] = field(default_factory=list)

    # Iteration evaluation tracking
    iteration_results: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_context_for_agent(self, agent_name: str) -> dict[str, Any]:
        """
        Return only the context fields relevant to a specific agent.

        This prevents sending unnecessary information to the LLM,
        reducing token usage and potential confusion.

        Args:
            agent_name: The name/type of the agent.

        Returns:
            Dict with relevant context fields.
        """
        # Base context always includes the problem
        context: dict[str, Any] = {
            "problem": self.problem,
            "problem_id": self.problem_id,
            "entry_point": self.entry_point,
        }

        if agent_name == "requirement_engineer":
            # RE only needs the problem
            pass

        elif agent_name == "architect":
            context["requirements"] = self.requirements

        elif agent_name == "developer":
            context["requirements"] = self.requirements
            context["design"] = self.design
            if self.failure_reports:
                context["failure_reports"] = [
                    _truncate_error_report(r) for r in self.failure_reports
                ]
            if self.reviews:
                context["reviews"] = self.reviews

        elif agent_name == "tester":
            context["requirements"] = self.requirements
            context["design"] = self.design
            context["code"] = self.code

        elif agent_name == "scrum_master":
            # Scrum master gets everything for coordination
            context["requirements"] = self.requirements
            context["design"] = self.design
            context["code"] = self.code
            context["tests"] = self.tests
            context["test_results"] = self.test_results

        elif agent_name == "raw":
            # Raw mode: only the problem
            pass

        else:
            # Unknown agent: give full context
            context["requirements"] = self.requirements
            context["design"] = self.design
            context["code"] = self.code
            context["tests"] = self.tests

        return {k: v for k, v in context.items() if v}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for persistence."""
        return {
            "problem": self.problem,
            "problem_id": self.problem_id,
            "entry_point": self.entry_point,
            "canonical_tests": self.canonical_tests,
            "requirements": self.requirements,
            "design": self.design,
            "code": self.code,
            "tests": self.tests,
            "test_results": self.test_results,
            "failure_reports": self.failure_reports,
            "reviews": self.reviews,
            "iteration_results": self.iteration_results,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SharedContext:
        """Deserialize from dict."""
        return cls(
            problem=data.get("problem", ""),
            problem_id=data.get("problem_id", ""),
            entry_point=data.get("entry_point", ""),
            canonical_tests=data.get("canonical_tests", ""),
            requirements=data.get("requirements", ""),
            design=data.get("design", ""),
            code=data.get("code", ""),
            tests=data.get("tests", ""),
            test_results=data.get("test_results", ""),
            failure_reports=data.get("failure_reports", []),
            reviews=data.get("reviews", []),
            iteration_results=data.get("iteration_results", []),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str | Path) -> None:
        """Save context to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> SharedContext:
        """Load context from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# ── Helpers ─────────────────────────────────────────────────────────

# Limite máximo de caracteres por relatório de falha enviado ao LLM.
# Tracebacks enormes consomem tokens preciosos da janela de contexto
# de SLMs (que costumam ter 4k-8k tokens).
_MAX_REPORT_CHARS = 1500


def _truncate_error_report(report: str) -> str:
    """
    Trunca um relatório de erro mantendo as partes mais informativas.

    Estratégia:
    - Se o relatório cabe dentro do limite, retorna como está.
    - Caso contrário, mantém as primeiras linhas (contexto) e as
      últimas linhas (a exceção final e o traceback mais relevante),
      cortando o miolo com um indicador de truncamento.
    """
    if len(report) <= _MAX_REPORT_CHARS:
        return report

    lines = report.splitlines()

    # Reserva ~40% para o início e ~60% para o final (o erro é mais importante)
    head_budget = _MAX_REPORT_CHARS * 2 // 5
    tail_budget = _MAX_REPORT_CHARS - head_budget

    head_lines: list[str] = []
    head_len = 0
    for line in lines:
        if head_len + len(line) + 1 > head_budget:
            break
        head_lines.append(line)
        head_len += len(line) + 1

    tail_lines: list[str] = []
    tail_len = 0
    for line in reversed(lines):
        if tail_len + len(line) + 1 > tail_budget:
            break
        tail_lines.insert(0, line)
        tail_len += len(line) + 1

    truncation_marker = "\n... [TRUNCADO — relatório de erro reduzido para caber no contexto] ...\n"

    return "\n".join(head_lines) + truncation_marker + "\n".join(tail_lines)
