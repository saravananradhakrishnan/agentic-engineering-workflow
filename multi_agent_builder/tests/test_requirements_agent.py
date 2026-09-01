"""Unit tests for RequirementsAgent and RequirementSpec schema validation."""

import pytest
from unittest.mock import MagicMock
from multi_agent_builder.models.schemas import RequirementSpec
from multi_agent_builder.agents.requirements import RequirementsAgent


def test_requirement_spec_validation():
    """Verify RequirementSpec schema requires all 8 specified fields."""
    spec = RequirementSpec(
        application_name="TaskManagerCLI",
        problem_statement="Track tasks from terminal efficiently.",
        functional_requirements=["Add task", "List tasks", "Complete task"],
        non_functional_requirements=["Response time < 100ms", "Zero external DB dependency"],
        api_requirements=["cli.add(task_name: str)", "cli.list()"],
        data_requirements=["Task dataclass with id, title, completed bool"],
        assumptions=["Runs in standard terminal with Python 3.12"],
        acceptance_criteria=["Tasks persist in JSON file"],
    )

    assert spec.application_name == "TaskManagerCLI"
    assert spec.problem_statement == "Track tasks from terminal efficiently."
    assert len(spec.functional_requirements) == 3
    assert len(spec.non_functional_requirements) == 2
    assert len(spec.api_requirements) == 2
    assert len(spec.data_requirements) == 1
    assert len(spec.assumptions) == 1
    assert len(spec.acceptance_criteria) == 1


def test_requirements_agent_fallback_mode():
    """Test RequirementsAgent behavior when no LLM is configured."""
    agent = RequirementsAgent()
    agent.llm = None  # Force offline fallback mode

    state = {"user_requirement": "Build a simple calculator"}
    res = agent.run(state)

    assert "structured_requirements" in res
    req = res["structured_requirements"]

    assert "application_name" in req
    assert "problem_statement" in req
    assert "functional_requirements" in req
    assert "non_functional_requirements" in req
    assert "api_requirements" in req
    assert "data_requirements" in req
    assert "assumptions" in req
    assert "acceptance_criteria" in req


def test_requirements_agent_mock_llm(mocker=None):
    """Test RequirementsAgent structured output invocation using a mock LLM."""
    mock_spec = RequirementSpec(
        application_name="FibonacciCalc",
        problem_statement="Calculate Fibonacci numbers efficiently",
        functional_requirements=["Compute nth Fibonacci"],
        non_functional_requirements=["O(n) time complexity"],
        api_requirements=["fibonacci(n: int) -> int"],
        data_requirements=["Integer state"],
        assumptions=["Input n >= 0"],
        acceptance_criteria=["fibonacci(10) == 55"],
    )

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_spec
    mock_llm.with_structured_output.return_value = mock_structured_llm

    agent = RequirementsAgent(llm=mock_llm)
    state = {"user_requirement": "Build a Fibonacci generator"}

    # Mock prompt | structured_llm chain
    with pytest.MonkeyPatch.context() as mp:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_spec
        mp.setattr("langchain_core.prompts.ChatPromptTemplate.__or__", lambda self, other: mock_chain)

        result = agent.run(state)

    assert "structured_requirements" in result
    output_spec = result["structured_requirements"]
    assert output_spec["application_name"] == "FibonacciCalc"
    assert output_spec["functional_requirements"] == ["Compute nth Fibonacci"]
    assert output_spec["acceptance_criteria"] == ["fibonacci(10) == 55"]
