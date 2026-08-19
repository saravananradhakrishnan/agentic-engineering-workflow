"""LangGraph workflow graph assembly and conditional routing."""

from typing import Literal
from langgraph.graph import StateGraph, END
from multi_agent_builder.graph.state import AgentState
from multi_agent_builder.agents.requirements import RequirementsAgent
from multi_agent_builder.agents.builder_agent import BuilderAgent
from multi_agent_builder.agents.test_agent import TestAgent
from multi_agent_builder.agents.validation_agent import ValidationAgent
from multi_agent_builder.config import settings


MAX_ITERATIONS = 3


def create_requirements_node(state: AgentState) -> dict:
    """Node handler wrapper for RequirementsAgent."""
    agent = RequirementsAgent()
    return agent.run(state)


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
    """Assembles and compiles the multi-agent LangGraph workflow.

    Workflow topology:
    [START] -> requirements -> builder -> test -> validation -> (route_after_validation)
                                  ^                             │
                                  └──────── [retry] ────────────┘
                                            [end / max]: END
    """
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("requirements", create_requirements_node)
    workflow.add_node("builder", create_builder_node)
    workflow.add_node("test", create_test_node)
    workflow.add_node("validation", create_validation_node)

    # Set entry point
    workflow.set_entry_point("requirements")

    # Define linear execution edges leading to validation
    workflow.add_edge("requirements", "builder")
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
