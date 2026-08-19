"""Unit tests for TestAgent and TestResult evaluation logic."""

import pytest
from unittest.mock import MagicMock
from multi_agent_builder.models.schemas import (
    RequirementSpec,
    BuildResult,
    GeneratedFile,
    ImplementationSummary,
    TestResult,
    TestCase,
    TestExecution,
)
from multi_agent_builder.agents.test_agent import TestAgent


@pytest.fixture
def sample_requirement_spec():
    """Sample RequirementSpec dictionary for test cases."""
    return {
        "application_name": "CalculatorApp",
        "problem_statement": "Perform division and addition",
        "functional_requirements": ["Add two numbers", "Divide two numbers"],
        "non_functional_requirements": ["Raise ValueError on division by zero"],
        "api_requirements": ["add(a: float, b: float) -> float", "divide(a: float, b: float) -> float"],
        "data_requirements": ["Float numerical inputs"],
        "assumptions": ["Standard Python environment"],
        "acceptance_criteria": ["divide(10, 2) == 5.0", "divide(5, 0) raises ValueError"],
    }


def test_test_agent_all_requirements_satisfied(sample_requirement_spec):
    """Test 1: All requirements satisfied scenario."""
    files = [
        GeneratedFile(
            path="calc.py",
            content="def add(a, b):\n    return a + b\n\ndef divide(a, b):\n    if b == 0:\n        raise ValueError('Division by zero')\n    return a / b\n",
            purpose="Calculator implementation",
        )
    ]
    build = BuildResult(
        status="SUCCESS",
        files=files,
        implementation_summary=ImplementationSummary(
            overview="Complete calculator", components=["calc"], key_decisions=[]
        ),
        assumptions=[],
        potential_risks=[],
    )

    expected_tc = [
        TestCase(id="TC-001", requirement="Add two numbers", description="Verify addition", test_type="functional", expected_result="Returns sum"),
        TestCase(id="TC-002", requirement="Divide two numbers", description="Verify division", test_type="functional", expected_result="Returns quotient"),
        TestCase(id="TC-003", requirement="Raise ValueError on division by zero", description="Verify zero division error", test_type="invalid_input", expected_result="Raises ValueError"),
    ]
    expected_ex = [
        TestExecution(test_case_id="TC-001", status="PASSED", actual_result="add function present"),
        TestExecution(test_case_id="TC-002", status="PASSED", actual_result="divide function present"),
        TestExecution(test_case_id="TC-003", status="PASSED", actual_result="Zero division check implemented"),
    ]
    mock_test_result = TestResult(
        status="PASSED",
        test_cases=expected_tc,
        executions=expected_ex,
        passed_count=3,
        failed_count=0,
        coverage_summary="100% requirements satisfied",
        issues=[],
        recommendations=[],
    )

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_test_result
    mock_llm.with_structured_output.return_value = mock_structured_llm

    agent = TestAgent(llm=mock_llm)
    state = {
        "structured_requirements": sample_requirement_spec,
        "build_result": build.model_dump(),
    }

    with pytest.MonkeyPatch.context() as mp:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_test_result
        mp.setattr("langchain_core.prompts.ChatPromptTemplate.__or__", lambda self, other: mock_chain)

        res = agent.run(state)

    assert res["test_result"]["status"] == "PASSED"
    assert res["test_result"]["passed_count"] == 3
    assert res["test_result"]["failed_count"] == 0


def test_test_agent_missing_functionality(sample_requirement_spec):
    """Test 2: Missing functionality scenario (e.g. divide function missing)."""
    files = [
        GeneratedFile(
            path="calc.py",
            content="def add(a, b):\n    return a + b\n",
            purpose="Incomplete implementation",
        )
    ]
    build = BuildResult(
        status="SUCCESS",
        files=files,
        implementation_summary=ImplementationSummary(
            overview="Incomplete calculator", components=["calc"], key_decisions=[]
        ),
        assumptions=[],
        potential_risks=[],
    )

    mock_test_result = TestResult(
        status="FAILED",
        test_cases=[
            TestCase(id="TC-001", requirement="Add two numbers", description="Verify add", test_type="functional", expected_result="Returns sum"),
            TestCase(id="TC-002", requirement="Divide two numbers", description="Verify divide", test_type="functional", expected_result="Returns quotient"),
        ],
        executions=[
            TestExecution(test_case_id="TC-001", status="PASSED", actual_result="add present"),
            TestExecution(test_case_id="TC-002", status="FAILED", actual_result="divide function missing", error="Function divide is not defined"),
        ],
        passed_count=1,
        failed_count=1,
        coverage_summary="50% requirements satisfied",
        issues=["divide function missing from calc.py"],
        recommendations=["Implement divide function in calc.py"],
    )

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_test_result
    mock_llm.with_structured_output.return_value = mock_structured_llm

    agent = TestAgent(llm=mock_llm)
    state = {
        "structured_requirements": sample_requirement_spec,
        "build_result": build.model_dump(),
    }

    with pytest.MonkeyPatch.context() as mp:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_test_result
        mp.setattr("langchain_core.prompts.ChatPromptTemplate.__or__", lambda self, other: mock_chain)

        res = agent.run(state)

    assert res["test_result"]["status"] == "FAILED"
    assert res["test_result"]["failed_count"] == 1
    assert "divide function missing" in res["test_result"]["issues"][0]


def test_test_agent_invalid_input_handling(sample_requirement_spec):
    """Test 3: Invalid input handling scenario (missing error check for zero division)."""
    files = [
        GeneratedFile(
            path="calc.py",
            content="def divide(a, b):\n    return a / b\n",
            purpose="Calculator divide without error handling",
        )
    ]
    build = BuildResult(
        status="SUCCESS",
        files=files,
        implementation_summary=ImplementationSummary(
            overview="Calculator without error validation", components=["calc"], key_decisions=[]
        ),
        assumptions=[],
        potential_risks=[],
    )

    mock_test_result = TestResult(
        status="FAILED",
        test_cases=[
            TestCase(id="TC-001", requirement="Raise ValueError on division by zero", description="Invalid input check", test_type="invalid_input", expected_result="ValueError"),
        ],
        executions=[
            TestExecution(test_case_id="TC-001", status="FAILED", actual_result="Raises ZeroDivisionError instead of ValueError", error="Wrong exception type"),
        ],
        passed_count=0,
        failed_count=1,
        coverage_summary="Failed invalid input handling requirement",
        issues=["Zero division check does not raise ValueError"],
        recommendations=["Explicitly check if b == 0 and raise ValueError"],
    )

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_test_result
    mock_llm.with_structured_output.return_value = mock_structured_llm

    agent = TestAgent(llm=mock_llm)
    state = {
        "structured_requirements": sample_requirement_spec,
        "build_result": build.model_dump(),
    }

    with pytest.MonkeyPatch.context() as mp:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_test_result
        mp.setattr("langchain_core.prompts.ChatPromptTemplate.__or__", lambda self, other: mock_chain)

        res = agent.run(state)

    assert res["test_result"]["status"] == "FAILED"
    assert res["test_result"]["executions"][0]["status"] == "FAILED"


def test_test_agent_malformed_build_result(sample_requirement_spec):
    """Test 4: Malformed BuildResult scenario."""
    agent = TestAgent()
    agent.llm = None

    state = {
        "structured_requirements": sample_requirement_spec,
        "build_result": {},  # Empty / malformed build result
    }

    res = agent.run(state)

    assert "test_result" in res
    assert res["test_result"]["status"] == "FAILED"
    assert res["test_result"]["failed_count"] == 1
    assert "malformed" in res["test_result"]["issues"][0].lower()


def test_test_agent_implementation_does_not_satisfy_requirement(sample_requirement_spec):
    """Test 5: Implementation does not satisfy a requirement scenario."""
    files = [
        GeneratedFile(
            path="calc.py",
            content="def add(a, b):\n    return a - b  # Bug: returns subtraction instead of addition!\n",
            purpose="Faulty addition implementation",
        )
    ]
    build = BuildResult(
        status="SUCCESS",
        files=files,
        implementation_summary=ImplementationSummary(
            overview="Buggy calculator", components=["calc"], key_decisions=[]
        ),
        assumptions=[],
        potential_risks=[],
    )

    mock_test_result = TestResult(
        status="FAILED",
        test_cases=[
            TestCase(id="TC-001", requirement="Add two numbers", description="Verify addition calculation", test_type="functional", expected_result="a + b"),
        ],
        executions=[
            TestExecution(test_case_id="TC-001", status="FAILED", actual_result="Performs subtraction (a - b) instead of addition", error="Calculation logic error"),
        ],
        passed_count=0,
        failed_count=1,
        coverage_summary="0% requirement compliance",
        issues=["Addition function logic is incorrect"],
        recommendations=["Change subtraction operator (-) to addition operator (+)"],
    )

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_test_result
    mock_llm.with_structured_output.return_value = mock_structured_llm

    agent = TestAgent(llm=mock_llm)
    state = {
        "structured_requirements": sample_requirement_spec,
        "build_result": build.model_dump(),
    }

    with pytest.MonkeyPatch.context() as mp:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_test_result
        mp.setattr("langchain_core.prompts.ChatPromptTemplate.__or__", lambda self, other: mock_chain)

        res = agent.run(state)

    assert res["test_result"]["status"] == "FAILED"
    assert res["test_result"]["executions"][0]["error"] == "Calculation logic error"
