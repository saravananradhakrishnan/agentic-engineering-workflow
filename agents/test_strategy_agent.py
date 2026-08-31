"""Test Strategy Agent implementation.

Consumes RequirementSpec and produces a structured TestStrategy.
"""

from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from multi_agent_builder.agents.base import BaseAgent
from multi_agent_builder.models.schemas import TestStrategy, RequirementSpec


TEST_STRATEGY_SYSTEM_PROMPT = """You are a Lead Quality Assurance & Test Strategist.
Your task is to review the technical RequirementSpec and design a comprehensive test strategy.

Analyze the requirements and generate:
1. Test Strategy: High-level testing philosophy and scope.
2. Functional Tests: Functional scenarios covering core features.
3. API Tests: API interface and signature validation tests.
4. Edge Cases: Boundary conditions, empty inputs, extreme values.
5. Negative Tests: Failure cases, invalid arguments, exception handling tests.
6. Non-Functional Tests: Performance, reliability, and code coverage targets.
7. Acceptance Criteria Mapping: Direct mapping of test cases to acceptance criteria.
"""


class TestStrategyAgent(BaseAgent):
    """Consumes RequirementSpec and produces a structured TestStrategy."""

    __test__ = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="TestStrategyAgent", **kwargs)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process RequirementSpec and return structured TestStrategy.

        Args:
            state: Graph state containing 'structured_requirements'.

        Returns:
            State update dictionary containing 'test_strategy'.
        """
        raw_req = state.get("structured_requirements") or {}

        if self.llm is not None:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", TEST_STRATEGY_SYSTEM_PROMPT),
                    ("human", "Requirement Specification:\n{req_spec}"),
                ]
            )
            structured_llm = self.llm.with_structured_output(TestStrategy)
            chain = prompt | structured_llm
            strategy: TestStrategy = chain.invoke({"req_spec": str(raw_req)})
            return {"test_strategy": strategy.model_dump()}

        # Offline fallback
        acc_criteria = raw_req.get("acceptance_criteria", ["General functionality"]) if isinstance(raw_req, dict) else ["General functionality"]
        fallback_strategy = TestStrategy(
            test_strategy="Automated Pytest unit testing with edge and negative case validation",
            functional_tests=["Test valid inputs return expected correct outputs"],
            api_tests=["Test main entry point signature and return types"],
            edge_cases=["Test boundary values (0, 1, negative numbers, empty strings)"],
            negative_tests=["Test invalid input types raise ValueError or TypeError"],
            non_functional_tests=["Ensure test suite runs in under 1 second"],
            acceptance_criteria_mapping=[f"Mapped to: {crit}" for crit in acc_criteria],
        )
        return {"test_strategy": fallback_strategy.model_dump()}
