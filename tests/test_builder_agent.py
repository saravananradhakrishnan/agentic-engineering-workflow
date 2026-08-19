"""Unit tests for BuilderAgent and BuildResult Pydantic schema validation."""

import pytest
from unittest.mock import MagicMock
from multi_agent_builder.models.schemas import (
    RequirementSpec,
    BuildResult,
    GeneratedFile,
    ImplementationSummary,
)
from multi_agent_builder.agents.builder_agent import BuilderAgent


def test_builder_result_schema_validation():
    """Verify BuildResult, GeneratedFile, and ImplementationSummary schema instantiations."""
    file1 = GeneratedFile(
        path="calculator/core.py",
        content="def add(a, b):\n    return a + b\n",
        purpose="Core calculator addition module",
    )
    file2 = GeneratedFile(
        path="tests/test_core.py",
        content="from calculator.core import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        purpose="Unit tests for addition",
    )
    summary = ImplementationSummary(
        overview="Simple calculator application with basic addition feature.",
        components=["calculator.core", "tests.test_core"],
        key_decisions=["Decoupled math operations from interface"],
    )
    result = BuildResult(
        status="SUCCESS",
        files=[file1, file2],
        implementation_summary=summary,
        assumptions=["Python 3.12 compatibility"],
        potential_risks=["No support for negative numbers yet"],
    )

    assert result.status == "SUCCESS"
    assert len(result.files) == 2
    assert result.files[0].path == "calculator/core.py"
    assert result.files[1].path == "tests/test_core.py"
    assert result.implementation_summary.overview.startswith("Simple calculator")
    assert len(result.assumptions) == 1
    assert len(result.potential_risks) == 1


def test_builder_agent_offline_fallback():
    """Test BuilderAgent when LLM is None (offline fallback mode)."""
    agent = BuilderAgent()
    agent.llm = None

    state = {
        "structured_requirements": {
            "application_name": "MathUtils",
            "problem_statement": "Perform basic arithmetic calculations",
            "functional_requirements": ["Add numbers"],
            "non_functional_requirements": ["High performance"],
            "api_requirements": ["add(a: float, b: float) -> float"],
            "data_requirements": ["Float numbers"],
            "assumptions": ["Valid numerical input"],
            "acceptance_criteria": ["add(1, 2) == 3"],
        }
    }

    output = agent.run(state)

    assert "build_result" in output
    assert "code_artifacts" in output

    build_res = output["build_result"]
    assert build_res["status"] == "SUCCESS"
    assert len(build_res["files"]) == 2
    assert "mathutils/main.py" in build_res["files"][0]["path"]


def test_builder_agent_mock_llm():
    """Test BuilderAgent execution using a mocked LLM structured output (no real API call)."""
    expected_files = [
        GeneratedFile(
            path="fib/calc.py",
            content="def fib(n: int) -> int:\n    return n if n <= 1 else fib(n-1) + fib(n-2)\n",
            purpose="Fibonacci implementation",
        ),
        GeneratedFile(
            path="tests/test_calc.py",
            content="from fib.calc import fib\n\ndef test_fib():\n    assert fib(10) == 55\n",
            purpose="Unit test for fib",
        ),
    ]
    expected_summary = ImplementationSummary(
        overview="Recursive Fibonacci implementation with unit tests.",
        components=["fib.calc", "tests.test_calc"],
        key_decisions=["Recursive approach chosen for clarity"],
    )
    expected_build = BuildResult(
        status="SUCCESS",
        files=expected_files,
        implementation_summary=expected_summary,
        assumptions=["Input n is non-negative"],
        potential_risks=["Recursion depth limit for large n"],
    )

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = expected_build
    mock_llm.with_structured_output.return_value = mock_structured_llm

    agent = BuilderAgent(llm=mock_llm)

    state = {
        "structured_requirements": {
            "application_name": "FibonacciCalc",
            "problem_statement": "Calculate nth Fibonacci number",
            "functional_requirements": ["Compute fibonacci"],
            "non_functional_requirements": ["Fast execution"],
            "api_requirements": ["fib(n: int) -> int"],
            "data_requirements": ["Integer input"],
            "assumptions": ["n >= 0"],
            "acceptance_criteria": ["fib(10) == 55"],
        }
    }

    with pytest.MonkeyPatch.context() as mp:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected_build
        mp.setattr("langchain_core.prompts.ChatPromptTemplate.__or__", lambda self, other: mock_chain)

        result = agent.run(state)

    assert "build_result" in result
    res_data = result["build_result"]
    assert res_data["status"] == "SUCCESS"
    assert len(res_data["files"]) == 2
    assert res_data["files"][0]["path"] == "fib/calc.py"
    assert res_data["files"][1]["path"] == "tests/test_calc.py"
    assert res_data["implementation_summary"]["overview"] == "Recursive Fibonacci implementation with unit tests."
