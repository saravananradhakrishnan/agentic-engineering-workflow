"""Requirements Agent implementation.

Transforms natural language product requests into strongly typed Pydantic RequirementSpec.
"""

from typing import Any, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from multi_agent_builder.agents.base import BaseAgent
from multi_agent_builder.models.schemas import RequirementSpec


REQUIREMENTS_SYSTEM_PROMPT = """You are an expert Lead Systems Architect and Requirements Engineer.
Your task is to analyze high-level, natural-language product requests and convert them into a comprehensive, strongly-typed technical specification.

Decompose the request thoroughly into:
1. Application Name: A concise, professional name for the target module/app.
2. Problem Statement: Core value proposition and the problem being solved.
3. Functional Requirements: Specific capabilities, behaviors, and features required.
4. Non-Functional Requirements: Performance, security, error handling, code quality, and maintainability.
5. API Requirements: Function/method signatures, interfaces, parameters, CLI flags, or API endpoints.
6. Data Requirements: Data structures, entity models, state schemas, and storage formats.
7. Assumptions: Environment, technical constraints, or scope assumptions made.
8. Acceptance Criteria: Clear, verifiable criteria to validate successful implementation.

Be clear, practical, and highly detailed. Ensure every field is populated accurately based on the prompt.
"""


class RequirementsAgent(BaseAgent):
    """Parses raw user input into a strongly typed RequirementSpec Pydantic model."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="RequirementsAgent", **kwargs)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw requirement prompt and return structured RequirementSpec.

        Args:
            state: Graph state containing 'user_requirement'.

        Returns:
            State update dictionary containing 'structured_requirements'.
        """
        user_input = state.get("user_requirement", "")
        if not user_input:
            raise ValueError("State is missing required 'user_requirement' key.")

        # If LLM client is available, run structured output chain
        if self.llm is not None:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", REQUIREMENTS_SYSTEM_PROMPT),
                    ("human", "Product Request:\n{user_requirement}"),
                ]
            )
            structured_llm = self.llm.with_structured_output(RequirementSpec)
            chain = prompt | structured_llm

            spec: RequirementSpec = chain.invoke({"user_requirement": user_input})
            return {"structured_requirements": spec.model_dump()}

        # Fallback for offline execution when no API key is provided
        fallback_spec = RequirementSpec(
            application_name="OfflineFallbackModule",
            problem_statement=f"Offline processing for: {user_input}",
            functional_requirements=["Core requirement parsing"],
            non_functional_requirements=["Python 3.12+ execution"],
            api_requirements=["main(input: str) -> bool"],
            data_requirements=["Dictionary state tracking"],
            assumptions=["No external LLM key provided; using fallback spec"],
            acceptance_criteria=["Generates non-empty RequirementSpec"],
        )
        return {"structured_requirements": fallback_spec.model_dump()}
