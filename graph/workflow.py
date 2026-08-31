"""LangGraph workflow graph assembly and conditional routing."""

import json
from typing import Literal
from langgraph.graph import StateGraph, END
from multi_agent_builder.graph.state import AgentState
from multi_agent_builder.agents.requirements import RequirementsAgent
from multi_agent_builder.agents.architecture_agent import ArchitectureAgent
from multi_agent_builder.agents.security_agent import SecurityAgent
from multi_agent_builder.agents.test_strategy_agent import TestStrategyAgent
from multi_agent_builder.agents.builder_agent import BuilderAgent
from multi_agent_builder.agents.test_agent import TestAgent
from multi_agent_builder.agents.validation_agent import ValidationAgent
from multi_agent_builder.config import settings


MAX_ITERATIONS = 3


def create_requirements_node(state: AgentState) -> dict:
    """Node handler wrapper for RequirementsAgent."""
    agent = RequirementsAgent()
    return agent.run(state)


def create_architecture_node(state: AgentState) -> dict:
    """Node handler wrapper for ArchitectureAgent."""
    agent = ArchitectureAgent()
    return agent.run(state)


def create_security_node(state: AgentState) -> dict:
    """Node handler wrapper for SecurityAgent."""
    agent = SecurityAgent()
    return agent.run(state)


def create_test_strategy_node(state: AgentState) -> dict:
    """Node handler wrapper for TestStrategyAgent."""
    agent = TestStrategyAgent()
    return agent.run(state)


def create_human_approval_node(state: AgentState) -> dict:
    """Node handler wrapper for Human Approval quality gate."""
    # If human_approval is already set in state (e.g. via test injection or programmatic run), respect it
    if state.get("human_approval") is not None:
        return {
            "human_approval": state["human_approval"],
            "human_feedback": state.get("human_feedback"),
        }

    arch_plan = state.get("architecture_plan") or {}
    sec_assess = state.get("security_assessment") or {}
    test_strat = state.get("test_strategy") or {}

    print("\n========================================")
    print("HUMAN APPROVAL REQUIRED")
    print("========================================\n")
    print("Architecture Plan:")
    print(json.dumps(arch_plan, indent=2))
    print("\nSecurity Assessment:")
    print(json.dumps(sec_assess, indent=2))
    print("\nTest Strategy:")
    print(json.dumps(test_strat, indent=2))
    print("\n========================================\n")

    try:
        user_choice = input("Approve implementation? [y/N]: ").strip()
    except (EOFError, KeyboardInterrupt):
        user_choice = "n"

    if user_choice.lower() == "y":
        return {"human_approval": "APPROVED"}

    try:
        feedback = input("Reason for rejection (optional): ").strip()
    except (EOFError, KeyboardInterrupt):
        feedback = ""

    return {
        "human_approval": "REJECTED",
        "human_feedback": feedback if feedback else "Implementation rejected by human reviewer.",
    }


def route_after_human_approval(state: AgentState) -> Literal["approved", "rejected"]:
    """Conditional routing decision function executed after Human Approval node."""
    if state.get("human_approval") == "APPROVED":
        return "approved"
    return "rejected"


def create_builder_node(state: AgentState) -> dict:
    """Node handler wrapper for BuilderAgent.

    Increments iteration_count on each execution pass.
    """
    current_iter = state.get("iteration_count", 0) + 1
    agent = BuilderAgent()
    res = agent.run(state)
    res["iteration_count"] = current_iter
    return res


def create_test_node(state: AgentState) -> dict:
    """Node handler wrapper for TestAgent."""
    agent = TestAgent()
    return agent.run(state)


def create_validation_node(state: AgentState) -> dict:
    """Node handler wrapper for ValidationAgent."""
    agent = ValidationAgent()
    return agent.run(state)


def route_after_validation(state: AgentState) -> Literal["end", "retry", "max_iterations"]:
    """Conditional routing decision function executed after ValidationAgent node.

    Behavior:
    - validation status == "PASS" -> "end"
    - validation status == "FAIL" and iteration_count < MAX_ITERATIONS -> "retry"
    - validation status == "FAIL" and iteration_count >= MAX_ITERATIONS -> "max_iterations"
    """
    val_res = state.get("validation_result") or state.get("validation_report") or {}
    status = val_res.get("status") or val_res.get("verdict", "FAIL")

    iter_count = state.get("iteration_count", 1)
    max_iters = state.get("max_iterations", MAX_ITERATIONS)

    if status == "PASS":
        return "end"

    if status == "FAIL" and iter_count < max_iters:
        return "retry"

    return "max_iterations"


def build_graph():
    """Assembles and compiles the multi-agent LangGraph workflow with parallel planning and human approval.

    Workflow topology:
                          [START]
                             │
                        requirements
                    ┌────────┼────────┐
                    ▼        ▼        ▼
               architecture security test_strategy
                    └────────┬────────┘
                             ▼
                       human_approval
                       ┌─────┴─────┐
                       ▼           ▼
                    APPROVED    REJECTED -> [END]
                       │
                       ▼
                    builder ◄──────┐
                       │           │
                       ▼           │ (retry)
                     test          │
                       │           │
                       ▼           │
                   validation ─────┘
                       │
                 (route_after_val)
                       │
                       ▼
                     [END]
    """
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("requirements", create_requirements_node)
    workflow.add_node("architecture", create_architecture_node)
    workflow.add_node("security", create_security_node)
    workflow.add_node("test_strategy", create_test_strategy_node)
    workflow.add_node("human_approval", create_human_approval_node)
    workflow.add_node("builder", create_builder_node)
    workflow.add_node("test", create_test_node)
    workflow.add_node("validation", create_validation_node)

    # Set entry point
    workflow.set_entry_point("requirements")

    # Parallel fan-out from requirements to planning agents
    workflow.add_edge("requirements", "architecture")
    workflow.add_edge("requirements", "security")
    workflow.add_edge("requirements", "test_strategy")

    # Parallel fan-in from planning agents to human_approval
    workflow.add_edge("architecture", "human_approval")
    workflow.add_edge("security", "human_approval")
    workflow.add_edge("test_strategy", "human_approval")

    # Conditional routing after human approval
    workflow.add_conditional_edges(
        "human_approval",
        route_after_human_approval,
        {
            "approved": "builder",
            "rejected": END,
        },
    )

    # Linear execution edges from builder leading to validation
    workflow.add_edge("builder", "test")
    workflow.add_edge("test", "validation")

    # Define explicit conditional edge routing after ValidationAgent
    workflow.add_conditional_edges(
        "validation",
        route_after_validation,
        {
            "end": END,
            "retry": "builder",
            "max_iterations": END,
        },
    )

    return workflow.compile()

