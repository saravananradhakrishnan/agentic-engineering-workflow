"""Unit tests for LangGraph conditional feedback loop routing."""

import pytest
from unittest.mock import MagicMock
from multi_agent_builder.graph.workflow import (
    build_graph,
    route_after_validation,
    MAX_ITERATIONS,
)
from multi_agent_builder.models.schemas import (
    RequirementSpec,
    BuildResult,
    GeneratedFile,
    ImplementationSummary,
    TestResult,
    TestCase,
    TestExecution,
    ValidationResult,
)


@pytest.fixture
def mock_req_spec():
    return {
        "application_name": "FeedbackApp",
        "problem_statement": "Iterative feedback loop application",
        "functional_requirements": ["Feature A"],
        "non_functional_requirements": ["High reliability"],
        "api_requirements": ["run()"],
        "data_requirements": ["Dict"],
        "assumptions": ["Python 3.12"],
        "acceptance_criteria": ["Feature A works"],
    }


@pytest.fixture
def mock_builder_result():
    return {
        "build_result": {
            "status": "SUCCESS",
            "files": [
                {
                    "path": "app.py",
                    "content": "# App code",
                    "purpose": "Main app",
                }
            ],
            "implementation_summary": {
                "overview": "App code",
                "components": ["app"],
                "key_decisions": [],
            },
            "assumptions": [],
            "potential_risks": [],
        }
    }


@pytest.fixture
def setup_graph_mocks(mp, mock_req_spec, mock_builder_result):
    """Utility helper to mock RequirementsAgent, BuilderAgent, and TestAgent."""
    mock_req_agent = MagicMock()
    mock_req_agent.run.return_value = {"structured_requirements": mock_req_spec}
    mp.setattr("multi_agent_builder.graph.workflow.RequirementsAgent", lambda **kwargs: mock_req_agent)

    mock_builder_agent = MagicMock()
    mock_builder_agent.run.return_value = mock_builder_result
    mp.setattr("multi_agent_builder.graph.workflow.BuilderAgent", lambda **kwargs: mock_builder_agent)

    mock_test_agent = MagicMock()
    mock_test_agent.run.return_value = {
        "test_result": {"status": "SUCCESS", "passed_count": 1, "failed_count": 0}
    }
    mp.setattr("multi_agent_builder.graph.workflow.TestAgent", lambda **kwargs: mock_test_agent)


def test_route_after_validation_pass():
    """Verify route_after_validation returns 'end' when validation passes."""
    state = {
        "validation_result": {"status": "PASS"},
        "iteration_count": 1,
        "max_iterations": 3,
    }
    route = route_after_validation(state)
    assert route == "end"


def test_route_after_validation_fail_retry():
    """Verify route_after_validation returns 'retry' when status is FAIL and iteration < MAX."""
    state = {
        "validation_result": {"status": "FAIL"},
        "iteration_count": 1,
        "max_iterations": 3,
    }
    route = route_after_validation(state)
    assert route == "retry"


def test_route_after_validation_fail_max_iterations():
    """Verify route_after_validation returns 'max_iterations' when status is FAIL and iteration >= MAX."""
    state = {
        "validation_result": {"status": "FAIL"},
        "iteration_count": 3,
        "max_iterations": 3,
    }
    route = route_after_validation(state)
    assert route == "max_iterations"


def test_graph_scenario_1_pass_on_first_attempt(mock_req_spec, mock_builder_result):
    """Test 1: Validation PASS on first attempt -> workflow ends after iteration 1."""
    pass_val = ValidationResult(
        status="PASS",
        overall_score=1.0,
        requirements_coverage=1.0,
        functional_assessment="All met",
        test_assessment="All passed",
        architecture_assessment="Good",
        security_assessment="Safe",
        issues=[],
        recommendations=[],
        failed_requirements=[],
        approval_reason="Full compliance",
    )

    with pytest.MonkeyPatch.context() as mp:
        mock_req_agent = MagicMock()
        mock_req_agent.run.return_value = {"structured_requirements": mock_req_spec}
        mp.setattr("multi_agent_builder.graph.workflow.RequirementsAgent", lambda **kwargs: mock_req_agent)

        mock_builder_agent = MagicMock()
        mock_builder_agent.run.return_value = mock_builder_result
        mp.setattr("multi_agent_builder.graph.workflow.BuilderAgent", lambda **kwargs: mock_builder_agent)

        mock_test_agent = MagicMock()
        mock_test_agent.run.return_value = {"test_result": {"status": "SUCCESS"}}
        mp.setattr("multi_agent_builder.graph.workflow.TestAgent", lambda **kwargs: mock_test_agent)

        mock_val_agent = MagicMock()
        mock_val_agent.run.return_value = {"validation_result": pass_val.model_dump()}
        mp.setattr("multi_agent_builder.graph.workflow.ValidationAgent", lambda **kwargs: mock_val_agent)

        graph = build_graph()
        initial_state = {
            "user_requirement": "Build a module",
            "iteration_count": 0,
            "max_iterations": 3,
            "human_approval": "APPROVED",
        }

        final_state = graph.invoke(initial_state)

        assert final_state.get("iteration_count") == 1
        assert final_state.get("validation_result", {}).get("status") == "PASS"


def test_graph_scenario_2_fail_then_pass(mock_req_spec, mock_builder_result):
    """Test 2: Validation FAIL then PASS -> Builder executes twice."""
    fail_val = ValidationResult(
        status="FAIL",
        overall_score=0.4,
        requirements_coverage=0.5,
        functional_assessment="Incomplete",
        test_assessment="Failed",
        architecture_assessment="Needs work",
        security_assessment="Unknown",
        issues=["Missing Feature A"],
        recommendations=["Add Feature A"],
        failed_requirements=["Feature A"],
        approval_reason="Missing feature",
    )
    pass_val = ValidationResult(
        status="PASS",
        overall_score=1.0,
        requirements_coverage=1.0,
        functional_assessment="Complete",
        test_assessment="Passed",
        architecture_assessment="Good",
        security_assessment="Safe",
        issues=[],
        recommendations=[],
        failed_requirements=[],
        approval_reason="Approved on retry",
    )

    val_calls = 0

    def mock_val_run(state):
        nonlocal val_calls
        val_calls += 1
        if val_calls == 1:
            return {"validation_result": fail_val.model_dump()}
        return {"validation_result": pass_val.model_dump()}

    with pytest.MonkeyPatch.context() as mp:
        mock_req_agent = MagicMock()
        mock_req_agent.run.return_value = {"structured_requirements": mock_req_spec}
        mp.setattr("multi_agent_builder.graph.workflow.RequirementsAgent", lambda **kwargs: mock_req_agent)

        mock_builder_agent = MagicMock()
        mock_builder_agent.run.return_value = mock_builder_result
        mp.setattr("multi_agent_builder.graph.workflow.BuilderAgent", lambda **kwargs: mock_builder_agent)

        mock_test_agent = MagicMock()
        mock_test_agent.run.return_value = {"test_result": {"status": "SUCCESS"}}
        mp.setattr("multi_agent_builder.graph.workflow.TestAgent", lambda **kwargs: mock_test_agent)

        mock_val_agent = MagicMock()
        mock_val_agent.run.side_effect = mock_val_run
        mp.setattr("multi_agent_builder.graph.workflow.ValidationAgent", lambda **kwargs: mock_val_agent)

        graph = build_graph()
        initial_state = {
            "user_requirement": "Build a module",
            "iteration_count": 0,
            "max_iterations": 3,
            "human_approval": "APPROVED",
        }

        final_state = graph.invoke(initial_state)

        assert val_calls == 2
        assert final_state.get("iteration_count") == 2
        assert final_state.get("validation_result", {}).get("status") == "PASS"


def test_graph_scenario_3_fail_three_times(mock_req_spec, mock_builder_result):
    """Test 3: Validation FAIL three times -> workflow terminates after MAX_ITERATIONS (3)."""
    fail_val = ValidationResult(
        status="FAIL",
        overall_score=0.3,
        requirements_coverage=0.3,
        functional_assessment="Failing",
        test_assessment="Failing",
        architecture_assessment="Poor",
        security_assessment="Unverified",
        issues=["Persistent defect"],
        recommendations=["Refactor completely"],
        failed_requirements=["Feature A"],
        approval_reason="Failed validation repeatedly",
    )

    val_calls = 0

    def mock_val_run(state):
        nonlocal val_calls
        val_calls += 1
        return {"validation_result": fail_val.model_dump()}

    with pytest.MonkeyPatch.context() as mp:
        mock_req_agent = MagicMock()
        mock_req_agent.run.return_value = {"structured_requirements": mock_req_spec}
        mp.setattr("multi_agent_builder.graph.workflow.RequirementsAgent", lambda **kwargs: mock_req_agent)

        mock_builder_agent = MagicMock()
        mock_builder_agent.run.return_value = mock_builder_result
        mp.setattr("multi_agent_builder.graph.workflow.BuilderAgent", lambda **kwargs: mock_builder_agent)

        mock_test_agent = MagicMock()
        mock_test_agent.run.return_value = {"test_result": {"status": "SUCCESS"}}
        mp.setattr("multi_agent_builder.graph.workflow.TestAgent", lambda **kwargs: mock_test_agent)

        mock_val_agent = MagicMock()
        mock_val_agent.run.side_effect = mock_val_run
        mp.setattr("multi_agent_builder.graph.workflow.ValidationAgent", lambda **kwargs: mock_val_agent)

        graph = build_graph()
        initial_state = {
            "user_requirement": "Build a module",
            "iteration_count": 0,
            "max_iterations": 3,
            "human_approval": "APPROVED",
        }

        final_state = graph.invoke(initial_state)

        assert val_calls == 3
        assert final_state.get("iteration_count") == 3
        assert final_state.get("validation_result", {}).get("status") == "FAIL"


def test_graph_scenario_4_validation_result_passed_to_next_builder(mock_req_spec, mock_builder_result):
    """Test 4: Validation FAIL -> ValidationResult is passed into the next Builder iteration."""
    fail_val = ValidationResult(
        status="FAIL",
        overall_score=0.4,
        requirements_coverage=0.5,
        functional_assessment="Missing Feature A",
        test_assessment="Failing tests",
        architecture_assessment="Basic",
        security_assessment="Safe",
        issues=["Missing Feature A"],
        recommendations=["Add Feature A"],
        failed_requirements=["Feature A"],
        approval_reason="Missing feature A",
    )

    builder_states = []

    def mock_builder_run(state):
        builder_states.append(dict(state))
        return mock_builder_result

    val_calls = 0

    def mock_val_run(state):
        nonlocal val_calls
        val_calls += 1
        if val_calls == 1:
            return {"validation_result": fail_val.model_dump()}
        return {
            "validation_result": {
                "status": "PASS",
                "overall_score": 1.0,
                "requirements_coverage": 1.0,
                "functional_assessment": "All met",
                "test_assessment": "Passed",
                "architecture_assessment": "Good",
                "security_assessment": "Safe",
                "issues": [],
                "recommendations": [],
                "failed_requirements": [],
                "approval_reason": "Approved",
            }
        }

    with pytest.MonkeyPatch.context() as mp:
        mock_req_agent = MagicMock()
        mock_req_agent.run.return_value = {"structured_requirements": mock_req_spec}
        mp.setattr("multi_agent_builder.graph.workflow.RequirementsAgent", lambda **kwargs: mock_req_agent)

        mock_builder = MagicMock()
        mock_builder.run.side_effect = mock_builder_run
        mp.setattr("multi_agent_builder.graph.workflow.BuilderAgent", lambda **kwargs: mock_builder)

        mock_test_agent = MagicMock()
        mock_test_agent.run.return_value = {"test_result": {"status": "SUCCESS"}}
        mp.setattr("multi_agent_builder.graph.workflow.TestAgent", lambda **kwargs: mock_test_agent)

        mock_val = MagicMock()
        mock_val.run.side_effect = mock_val_run
        mp.setattr("multi_agent_builder.graph.workflow.ValidationAgent", lambda **kwargs: mock_val)

        graph = build_graph()
        initial_state = {
            "user_requirement": "Build a module",
            "iteration_count": 0,
            "max_iterations": 3,
            "human_approval": "APPROVED",
        }

        graph.invoke(initial_state)

        assert len(builder_states) == 2
        # First iteration: no previous validation result
        assert "validation_result" not in builder_states[0] or builder_states[0].get("validation_result") is None
        # Second iteration: previous validation result IS present
        assert "validation_result" in builder_states[1]
        assert builder_states[1]["validation_result"]["status"] == "FAIL"
        assert "Feature A" in builder_states[1]["validation_result"]["failed_requirements"]


def test_graph_scenario_5_iteration_count_never_exceeds_max(mock_req_spec, mock_builder_result):
    """Test 5: Iteration count never exceeds MAX_ITERATIONS (3)."""
    always_fail_val = ValidationResult(
        status="FAIL",
        overall_score=0.1,
        requirements_coverage=0.1,
        functional_assessment="Failing continuously",
        test_assessment="Failed",
        architecture_assessment="Poor",
        security_assessment="Safe",
        issues=["Always fails"],
        recommendations=["Fix all"],
        failed_requirements=["All"],
        approval_reason="Continuous failure",
    )

    with pytest.MonkeyPatch.context() as mp:
        mock_req_agent = MagicMock()
        mock_req_agent.run.return_value = {"structured_requirements": mock_req_spec}
        mp.setattr("multi_agent_builder.graph.workflow.RequirementsAgent", lambda **kwargs: mock_req_agent)

        mock_builder_agent = MagicMock()
        mock_builder_agent.run.return_value = mock_builder_result
        mp.setattr("multi_agent_builder.graph.workflow.BuilderAgent", lambda **kwargs: mock_builder_agent)

        mock_test_agent = MagicMock()
        mock_test_agent.run.return_value = {"test_result": {"status": "SUCCESS"}}
        mp.setattr("multi_agent_builder.graph.workflow.TestAgent", lambda **kwargs: mock_test_agent)

        mock_val_agent = MagicMock()
        mock_val_agent.run.return_value = {"validation_result": always_fail_val.model_dump()}
        mp.setattr("multi_agent_builder.graph.workflow.ValidationAgent", lambda **kwargs: mock_val_agent)

        graph = build_graph()
        initial_state = {
            "user_requirement": "Build a module",
            "iteration_count": 0,
            "max_iterations": 3,
            "human_approval": "APPROVED",
        }

        final_state = graph.invoke(initial_state)

        assert final_state.get("iteration_count") <= 3
        assert final_state.get("iteration_count") == 3

