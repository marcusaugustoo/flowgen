"""Tests for agents."""

import pytest

from src.agents.factory import create_agent, available_agents
from src.agents.requirement_engineer import RequirementEngineer
from src.agents.architect import Architect
from src.agents.developer import Developer
from src.agents.tester import Tester
from src.agents.scrum_master import ScrumMaster


class TestAgentFactory:
    """Tests for agent factory."""

    def test_available_agents(self):
        agents = available_agents()
        assert "requirement_engineer" in agents
        assert "architect" in agents
        assert "developer" in agents
        assert "tester" in agents
        assert "scrum_master" in agents

    def test_create_requirement_engineer(self, mock_llm):
        agent = create_agent("requirement_engineer", mock_llm)
        assert isinstance(agent, RequirementEngineer)
        assert agent.name == "requirement_engineer"

    def test_create_developer(self, mock_llm):
        agent = create_agent("developer", mock_llm)
        assert isinstance(agent, Developer)

    def test_unknown_agent_raises(self, mock_llm):
        with pytest.raises(ValueError, match="Unknown agent"):
            create_agent("nonexistent", mock_llm)


class TestRequirementEngineer:
    """Tests for RequirementEngineer agent."""

    def test_execute(self, mock_llm, sample_context):
        agent = RequirementEngineer(llm=mock_llm)
        artifact = agent.execute(sample_context)
        assert artifact.type == "requirements"
        assert artifact.created_by == "requirement_engineer"
        assert artifact.content  # Not empty
        assert sample_context.requirements  # Context was updated

    def test_uses_llm(self, mock_llm, sample_context):
        agent = RequirementEngineer(llm=mock_llm)
        agent.execute(sample_context)
        assert mock_llm.call_count == 1
        # Check that the prompt contained the problem
        prompt = mock_llm.get_last_prompt()
        assert "has_close_elements" in prompt or "close" in prompt.lower()


class TestArchitect:
    """Tests for Architect agent."""

    def test_execute(self, mock_llm, sample_context):
        sample_context.requirements = "Some requirements"
        agent = Architect(llm=mock_llm)
        artifact = agent.execute(sample_context)
        assert artifact.type == "design"
        assert artifact.created_by == "architect"
        assert sample_context.design  # Context was updated


class TestDeveloper:
    """Tests for Developer agent."""

    def test_execute(self, mock_llm_with_code, sample_context):
        sample_context.requirements = "Some requirements"
        sample_context.design = "Some design"
        agent = Developer(llm=mock_llm_with_code)
        artifact = agent.execute(sample_context)
        assert artifact.type == "code"
        assert artifact.created_by == "developer"
        assert "def has_close_elements" in artifact.content

    def test_code_extraction(self):
        """Test that code is extracted from markdown blocks."""
        from src.agents.code_extractor import extract_code
        text = '```python\ndef foo():\n    return 42\n```\nSome explanation.'
        code = extract_code(text)
        assert code == "def foo():\n    return 42"

    def test_refinement_mode(self, mock_llm_with_code, sample_context):
        sample_context.requirements = "req"
        sample_context.design = "design"
        sample_context.failure_reports = ["Test failed: assertion error"]
        agent = Developer(llm=mock_llm_with_code)
        artifact = agent.execute(sample_context)
        assert artifact.metadata.get("is_refinement") is True


class TestTester:
    """Tests for Tester agent."""

    def test_execute(self, mock_llm, sample_context):
        sample_context.requirements = "req"
        sample_context.design = "design"
        sample_context.code = "def foo(): pass"
        agent = Tester(llm=mock_llm)
        artifact = agent.execute(sample_context)
        assert artifact.type == "tests"
        assert artifact.created_by == "tester"


class TestScrumMaster:
    """Tests for ScrumMaster agent."""

    def test_execute(self, mock_llm, sample_context):
        sample_context.requirements = "req"
        sample_context.design = "design"
        agent = ScrumMaster(llm=mock_llm)
        artifact = agent.execute(sample_context)
        assert artifact.type == "sprint_plan"
        assert artifact.created_by == "scrum_master"
