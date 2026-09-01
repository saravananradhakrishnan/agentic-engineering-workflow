"""Architecture Agent implementation.

Consumes RequirementSpec and produces a structured ArchitecturePlan.
"""

from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from multi_agent_builder.agents.base import BaseAgent
from multi_agent_builder.models.schemas import ArchitecturePlan, RequirementSpec


ARCHITECTURE_SYSTEM_PROMPT = """You are a Lead Software Architect.
Your task is to review the technical RequirementSpec and construct a detailed architectural plan for the application.

Analyze the requirements and generate:
1. Architecture Style: Core pattern (e.g. Modular Monolith, Layered Architecture, Component-Driven).
2. Components: List of main software components/modules to build.
3. Technology Choices: Language, libraries, tools, runtime environment choices.
4. API Design: Method/function signatures or endpoints design.
5. Data Flow: Flow of data from inputs through components to outputs.
6. Design Decisions: Key technical decisions made and trade-offs.
7. Risks: Identified technical or architectural risks.
8. Recommendations: Actionable recommendations for the Builder Agent.
"""


class ArchitectureAgent(BaseAgent):
    """Consumes RequirementSpec and produces a structured ArchitecturePlan."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="ArchitectureAgent", **kwargs)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process RequirementSpec and return structured ArchitecturePlan.

        Args:
            state: Graph state containing 'structured_requirements'.

        Returns:
            State update dictionary containing 'architecture_plan'.
        """
        raw_req = state.get("structured_requirements") or {}

        if self.llm is not None:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", ARCHITECTURE_SYSTEM_PROMPT),
                    ("human", "Requirement Specification:\n{req_spec}"),
                ]
            )
            structured_llm = self.llm.with_structured_output(ArchitecturePlan)
            chain = prompt | structured_llm
            plan: ArchitecturePlan = chain.invoke({"req_spec": str(raw_req)})
            return {"architecture_plan": plan.model_dump()}

        # Offline fallback
        app_name = raw_req.get("application_name", "AppModule") if isinstance(raw_req, dict) else "AppModule"
        fallback_plan = ArchitecturePlan(
            architecture_style="Modular Layered Architecture",
            components=[f"{app_name}.core", f"{app_name}.utils", "tests"],
            technology_choices=["Python 3.12+", "Pytest", "Pydantic"],
            api_design="Clean modular functions with strict type annotations",
            data_flow="Input data -> Validator -> Processor -> Output result",
            design_decisions=["Decouple business logic from side effects", "Use Pydantic for validation"],
            risks=["Potential edge cases in unvalidated input"],
            recommendations=["Implement comprehensive unit tests for core logic"],
        )
        return {"architecture_plan": fallback_plan.model_dump()}
