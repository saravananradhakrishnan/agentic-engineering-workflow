"""Unit tests for Phase 3: Parallel Planning Agents and Human-in-the-Loop Approval."""

import pytest
from unittest.mock import MagicMock, patch
from multi_agent_builder.models.schemas import (
    ArchitecturePlan,
    SecurityAssessment,
    TestStrategy,
    RequirementSpec,
    BuildResult,
    ValidationResult,
)
from multi_agent_builder.agents.architecture_agent import ArchitectureAgent
from multi_agent_builder.agents.security_agent import SecurityAgent
from multi_agent_builder.agents.test_strategy_agent import TestStrategyAgent
from multi_agent_builder.agents.builder_agent import BuilderAgent
from multi_agent_builder.graph.workflow import (
    build_graph,
    create_human_approval_node,
    route_after_human_approval,
)


@pytest.fixture
def mock_req_spec():
    return {
        "application_name": "TestPhase3App",
        "problem_statement": "Validation app for Phase 3 capabilities",
        "functional_requirements": ["Perform secure calculation"],
        "non_functional_requirements": ["High security", "Comprehensive testing"],
        "api_requirements": ["calculate(a: int, b: int) -> int"],
        "data_requirements": ["Input JSON schema"],
        "assumptions": ["Python 3.12+"],
        "acceptance_criteria": ["All tests pass"],
    }


def test_1_architecture_agent_independently_produces_architecture_plan(mock_req_spec):
    """1. ArchitectureAgent independently produces ArchitecturePlan."""
    agent = ArchitectureAgent(llm=None)
    state = {"structured_requirements": mock_req_spec}
    result = agent.run(state)

    assert "architecture_plan" in result
    plan_dict = result["architecture_plan"]
    plan = ArchitecturePlan(**plan_dict)
    assert plan.architecture_style is not None
    assert isinstance(plan.components, list)
    assert isinstance(plan.recommendations, list)


def test_2_security_agent_independently_produces_security_assessment(mock_req_spec):
    """2. SecurityAgent independently produces SecurityAssessment."""
    agent = SecurityAgent(llm=None)
    state = {"structured_requirements": mock_req_spec}
    result = agent.run(state)

    assert "security_assessment" in result
    sec_dict = result["security_assessment"]
    sec = SecurityAssessment(**sec_dict)
    assert sec.security_status in ["SAFE", "NEEDS_REVIEW", "CRITICAL"]
    assert isinstance(sec.threats, list)
    assert isinstance(sec.recommendations, list)


def test_3_test_strategy_agent_independently_produces_test_strategy(mock_req_spec):
    """3. TestStrategyAgent independently produces TestStrategy."""
    agent = TestStrategyAgent(llm=None)
    state = {"structured_requirements": mock_req_spec}
    result = agent.run(state)

    assert "test_strategy" in result
    strat_dict = result["test_strategy"]
    strat = TestStrategy(**strat_dict)
    assert strat.test_strategy is not None
    assert isinstance(strat.functional_tests, list)
    assert isinstance(strat.edge_cases, list)


def test_4_all_three_agents_can_execute_from_same_requirement_spec(mock_req_spec):
    """4. All three agents can execute from the same RequirementSpec."""
    state = {"structured_requirements": mock_req_spec}

    arch_res = ArchitectureAgent(llm=None).run(state)
    sec_res = SecurityAgent(llm=None).run(state)
    strat_res = TestStrategyAgent(llm=None).run(state)

    assert "architecture_plan" in arch_res
    assert "security_assessment" in sec_res
    assert "test_strategy" in strat_res


def test_5_builder_receives_all_three_planning_outputs(mock_req_spec):
    """5. Builder receives all three planning outputs."""
    arch_plan = ArchitectureAgent(llm=None).run({"structured_requirements": mock_req_spec})["architecture_plan"]
    sec_assess = SecurityAgent(llm=None).run({"structured_requirements": mock_req_spec})["security_assessment"]
    test_strat = TestStrategyAgent(llm=None).run({"structured_requirements": mock_req_spec})["test_strategy"]

    state = {
        "structured_requirements": mock_req_spec,
        "architecture_plan": arch_plan,
        "security_assessment": sec_assess,
        "test_strategy": test_strat,
    }

    mock_llm = MagicMock()

    mock_build_result = BuildResult(
        status="SUCCESS",
        files=[],
        implementation_summary={"overview": "Built", "components": [], "key_decisions": []},
        assumptions=[],
        potential_risks=[],
    )

    with patch("langchain_core.runnables.base.RunnableSequence.invoke", return_value=mock_build_result) as mock_invoke:
        builder = BuilderAgent(llm=mock_llm)
        builder.run(state)

        assert mock_invoke.called
        call_kwargs = mock_invoke.call_args[0][0]
        context_prompt = call_kwargs.get("context_prompt", "")

        assert "Architecture Plan:" in context_prompt
        assert "Security Assessment:" in context_prompt
        assert "Test Strategy:" in context_prompt


def test_6_human_approval_y_continues_workflow():
    """6. Human approval 'y' continues workflow."""
    with patch("builtins.input", side_effect=["y"]):
        state = {
            "architecture_plan": {"architecture_style": "Modular"},
            "security_assessment": {"security_status": "SAFE"},
            "test_strategy": {"test_strategy": "Pytest"},
        }
        res = create_human_approval_node(state)
        assert res["human_approval"] == "APPROVED"
        assert route_after_human_approval(res) == "approved"


def test_7_human_approval_n_terminates_workflow(mock_req_spec):
    """7. Human approval 'n' terminates workflow."""
    with patch("builtins.input", side_effect=["n", "Design issue"]):
        state = {
            "architecture_plan": {"architecture_style": "Modular"},
            "security_assessment": {"security_status": "SAFE"},
            "test_strategy": {"test_strategy": "Pytest"},
        }
        res = create_human_approval_node(state)
        assert res["human_approval"] == "REJECTED"
        assert route_after_human_approval(res) == "rejected"


def test_8_human_rejection_feedback_stored_in_agent_state():
    """8. Human rejection feedback is stored in AgentState."""
    rejection_reason = "Architecture violates performance requirements"
    with patch("builtins.input", side_effect=["n", rejection_reason]):
        state = {}
        res = create_human_approval_node(state)
        assert res["human_approval"] == "REJECTED"
        assert res["human_feedback"] == rejection_reason


def test_9_builder_cannot_execute_when_human_approval_is_rejected(mock_req_spec):
    """9. Builder cannot execute when human approval is rejected."""
    mock_builder_run = MagicMock()

    with patch.object(BuilderAgent, "run", mock_builder_run):
        graph = build_graph()
        initial_state = {
            "user_requirement": "Build a module",
            "human_approval": "REJECTED",
            "human_feedback": "Not approved",
        }
        final_state = graph.invoke(initial_state)

        assert mock_builder_run.call_count == 0
        assert final_state.get("human_approval") == "REJECTED"


def test_10_parallel_planning_completes_before_builder_starts(mock_req_spec):
    """10. Parallel planning completes before Builder starts."""
    execution_order = []

    def mock_arch_run(self, state=None):
        execution_order.append("architecture")
        return {"architecture_plan": {"architecture_style": "Layered"}}

    def mock_sec_run(self, state=None):
        execution_order.append("security")
        return {"security_assessment": {"security_status": "SAFE"}}

    def mock_strat_run(self, state=None):
        execution_order.append("test_strategy")
        return {"test_strategy": {"test_strategy": "Pytest"}}

    def mock_builder_run(self, state=None):
        execution_order.append("builder")
        return {
            "build_result": {
                "status": "SUCCESS",
                "files": [],
                "implementation_summary": {"overview": "ok", "components": [], "key_decisions": []},
                "assumptions": [],
                "potential_risks": [],
            }
        }

    with patch.object(ArchitectureAgent, "run", mock_arch_run), \
         patch.object(SecurityAgent, "run", mock_sec_run), \
         patch.object(TestStrategyAgent, "run", mock_strat_run), \
         patch.object(BuilderAgent, "run", mock_builder_run):

        graph = build_graph()
        initial_state = {
            "user_requirement": "Build a module",
            "human_approval": "APPROVED",
        }
        graph.invoke(initial_state)

        # Ensure all three planning agents ran before builder
        assert "architecture" in execution_order
        assert "security" in execution_order
        assert "test_strategy" in execution_order
        assert "builder" in execution_order

        builder_idx = execution_order.index("builder")
        arch_idx = execution_order.index("architecture")
        sec_idx = execution_order.index("security")
        strat_idx = execution_order.index("test_strategy")

        assert arch_idx < builder_idx
        assert sec_idx < builder_idx
        assert strat_idx < builder_idx


def test_11_existing_validation_retry_loop_still_works_after_builder(mock_req_spec):
    """11. Existing validation retry loop still works after Builder."""
    fail_val = ValidationResult(
        status="FAIL",
        overall_score=0.5,
        requirements_coverage=0.5,
        functional_assessment="Missing feature",
        test_assessment="Failed",
        architecture_assessment="Good",
        security_assessment="Safe",
        issues=["Feature incomplete"],
        recommendations=["Fix feature"],
        failed_requirements=["Feature"],
        approval_reason="Fail 1",
    )
    pass_val = ValidationResult(
        status="PASS",
        overall_score=1.0,
        requirements_coverage=1.0,
        functional_assessment="Passed",
        test_assessment="Passed",
        architecture_assessment="Good",
        security_assessment="Safe",
        issues=[],
        recommendations=[],
        failed_requirements=[],
        approval_reason="Approved",
    )

    builder_calls = 0

    def mock_builder_run(state):
        nonlocal builder_calls
        builder_calls += 1
        return {
            "build_result": {
                "status": "SUCCESS",
                "files": [],
                "implementation_summary": {"overview": "ok", "components": [], "key_decisions": []},
                "assumptions": [],
                "potential_risks": [],
            }
        }

    val_calls = 0

    def mock_val_run(state):
        nonlocal val_calls
        val_calls += 1
        if val_calls == 1:
            return {"validation_result": fail_val.model_dump()}
        return {"validation_result": pass_val.model_dump()}

    with patch.object(BuilderAgent, "run", side_effect=mock_builder_run), \
         patch("multi_agent_builder.graph.workflow.ValidationAgent.run", side_effect=mock_val_run):

        graph = build_graph()
        initial_state = {
            "user_requirement": "Build a module",
            "human_approval": "APPROVED",
            "iteration_count": 0,
            "max_iterations": 3,
        }
        final_state = graph.invoke(initial_state)

        assert builder_calls == 2
        assert final_state.get("validation_result", {}).get("status") == "PASS"


def test_12_existing_max_iterations_behavior_still_works(mock_req_spec):
    """12. Existing MAX_ITERATIONS behavior still works."""
    always_fail_val = ValidationResult(
        status="FAIL",
        overall_score=0.1,
        requirements_coverage=0.1,
        functional_assessment="Fail",
        test_assessment="Fail",
        architecture_assessment="Fail",
        security_assessment="Fail",
        issues=["Fail"],
        recommendations=["Fix"],
        failed_requirements=["All"],
        approval_reason="Continuous failure",
    )

    builder_calls = 0

    def mock_builder_run(state):
        nonlocal builder_calls
        builder_calls += 1
        return {
            "build_result": {
                "status": "SUCCESS",
                "files": [],
                "implementation_summary": {"overview": "ok", "components": [], "key_decisions": []},
                "assumptions": [],
                "potential_risks": [],
            }
        }

    with patch.object(BuilderAgent, "run", side_effect=mock_builder_run), \
         patch("multi_agent_builder.graph.workflow.ValidationAgent.run", return_value={"validation_result": always_fail_val.model_dump()}):

        graph = build_graph()
        initial_state = {
            "user_requirement": "Build a module",
            "human_approval": "APPROVED",
            "iteration_count": 0,
            "max_iterations": 3,
        }
        final_state = graph.invoke(initial_state)

        assert builder_calls == 3
        assert final_state.get("iteration_count") == 3
        assert final_state.get("validation_result", {}).get("status") == "FAIL"
