"""Initial verification tests for module imports and graph structure."""

from multi_agent_builder.graph.workflow import build_graph, route_after_validation
from multi_agent_builder.models.schemas import (
    UserRequirement,
    StructuredRequirements,
    CodeArtifact,
    TestResult,
    ValidationReport,
)
from multi_agent_builder.agents import (
    RequirementsAgent,
    BuilderAgent,
    TestAgent,
    ValidationAgent,
)


def test_schemas_instantiation():
    """Verify Pydantic models can be instantiated."""
    req = UserRequirement(raw_text="Build a CLI tool")
    assert req.raw_text == "Build a CLI tool"

    struct_req = StructuredRequirements(
        title="CLI Tool",
        features=["CLI interface"],
        constraints=["Python 3.12"],
        acceptance_criteria=["Runs without error"],
    )
    assert struct_req.title == "CLI Tool"

    code = CodeArtifact(filename="main.py", content="print('hello')")
    assert code.filename == "main.py"

    test_res = TestResult(
        status="PASSED",
        test_cases=[],
        executions=[],
        passed_count=1,
        failed_count=0,
        coverage_summary="Full coverage",
        issues=[],
        recommendations=[],
    )
    assert test_res.status == "PASSED"

    val_rep = ValidationReport(verdict="PASS", feedback="Good job")
    assert val_rep.verdict == "PASS"


def test_graph_compilation():
    """Verify LangGraph StateGraph builds successfully."""
    graph = build_graph()
    assert graph is not None


def test_route_after_validation_pass():
    """Verify graph termination when validation passes."""
    state = {
        "validation_result": {"status": "PASS"},
        "iteration_count": 1,
        "max_iterations": 3,
    }
    decision = route_after_validation(state)
    assert decision == "end"


def test_route_after_validation_fail_under_max():
    """Verify retry loopback when validation fails under max iterations."""
    state = {
        "validation_result": {"status": "FAIL"},
        "iteration_count": 1,
        "max_iterations": 3,
    }
    decision = route_after_validation(state)
    assert decision == "retry"


def test_route_after_validation_fail_exceeded_max():
    """Verify termination when retry limit is exceeded."""
    state = {
        "validation_result": {"status": "FAIL"},
        "iteration_count": 3,
        "max_iterations": 3,
    }
    decision = route_after_validation(state)
    assert decision == "max_iterations"


def test_agents_instantiation_without_api_key():
    """Verify agent stub initialization."""
    req_agent = RequirementsAgent()
    builder_agent = BuilderAgent()
    test_agent = TestAgent()
    val_agent = ValidationAgent()

    assert req_agent.name == "RequirementsAgent"
    assert builder_agent.name == "BuilderAgent"
    assert test_agent.name == "TestAgent"
    assert val_agent.name == "ValidationAgent"
