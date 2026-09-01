"""Security Agent implementation.

Consumes RequirementSpec and produces a structured SecurityAssessment.
"""

from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from multi_agent_builder.agents.base import BaseAgent
from multi_agent_builder.models.schemas import SecurityAssessment, RequirementSpec


SECURITY_SYSTEM_PROMPT = """You are a Lead Application Security Engineer.
Your task is to review the technical RequirementSpec and perform a thorough security assessment.

Analyze the requirements and generate:
1. Security Status: Overall security posture (SAFE, NEEDS_REVIEW, CRITICAL).
2. Threats: Threat vectors and potential attack scenarios.
3. Input Validation: Sanitization and validation requirements.
4. Authentication: Required authentication measures or N/A.
5. Authorization: Access control requirements or N/A.
6. Data Protection: Data protection, privacy, and encryption guidelines.
7. Vulnerabilities: Known vulnerability risks (e.g. injection, overflow, unhandled errors).
8. Recommendations: Specific security hardening recommendations for Builder Agent.
"""


class SecurityAgent(BaseAgent):
    """Consumes RequirementSpec and produces a structured SecurityAssessment."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="SecurityAgent", **kwargs)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process RequirementSpec and return structured SecurityAssessment.

        Args:
            state: Graph state containing 'structured_requirements'.

        Returns:
            State update dictionary containing 'security_assessment'.
        """
        raw_req = state.get("structured_requirements") or {}

        if self.llm is not None:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", SECURITY_SYSTEM_PROMPT),
                    ("human", "Requirement Specification:\n{req_spec}"),
                ]
            )
            structured_llm = self.llm.with_structured_output(SecurityAssessment)
            chain = prompt | structured_llm
            assessment: SecurityAssessment = chain.invoke({"req_spec": str(raw_req)})
            return {"security_assessment": assessment.model_dump()}

        # Offline fallback
        fallback_assessment = SecurityAssessment(
            security_status="SAFE",
            threats=["Invalid input injection", "Unhandled exceptions leading to state corruption"],
            input_validation="Enforce explicit type checks and boundary validation on all entry points",
            authentication="Not required for isolated utility module",
            authorization="Not required for isolated utility module",
            data_protection="Ensure immutable input parsing and clean memory management",
            vulnerabilities=["Type error crashes", "Resource leaks on invalid inputs"],
            recommendations=[
                "Use defensive programming with explicit exception handling",
                "Sanitize all public API arguments",
            ],
        )
        return {"security_assessment": fallback_assessment.model_dump()}
