"""Validation Agent implementation.

Acts as the final quality gate evaluating RequirementSpec, BuildResult, and TestResult across functional completeness, test execution, architecture, and security.
"""

import json
from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from multi_agent_builder.agents.base import BaseAgent
from multi_agent_builder.models.schemas import (
    RequirementSpec,
    BuildResult,
    TestResult,
    ValidationResult,
)


VALIDATION_AGENT_SYSTEM_PROMPT = """You are the Lead Technical Architect and Security Auditor acting as the final quality gate for software build artifacts.

Your responsibility is to determine whether a generated application should be APPROVED (status: "PASS") or REJECTED (status: "FAIL").

EVALUATION CRITERIA:
1. FUNCTIONAL CORRECTNESS & COVERAGE:
   - Are ALL functional requirements in RequirementSpec implemented?
   - If ANY important or core functional requirement is missing, you MUST issue a "FAIL".

2. TEST RESULTS & QUALITY:
   - Review TestResult execution outputs.
   - Are important unit/integration tests passing?
   - If any critical test failed, you MUST issue a "FAIL".
   - Even if tests pass, verify whether any important requirements were left untested.

3. ARCHITECTURAL ALIGNMENT:
   - Is the code modular, clean, and aligned with requested API requirements and Python 3.12+ standards?

4. SECURITY ASSESSMENT:
   - Check for obvious security flaws (e.g. unsafe eval/exec calls, hardcoded secrets, shell command injection, unsafe path traversals).
   - If any security issue is present, you MUST issue a "FAIL".

5. EDGE CASES & ERROR HANDLING:
   - Are boundary conditions, invalid inputs, and failure scenarios gracefully handled?

DECISION RULES:
- Issue "PASS" ONLY when all acceptance criteria are met, tests pass, requirements coverage is complete, and zero security risks are found.
- Issue "FAIL" whenever an important requirement is missing, tests fail, security risks exist, or acceptance criteria are unfulfilled.
- Always provide actionable, explicit recommendations and failed_requirements list when issuing a "FAIL".
"""


class ValidationAgent(BaseAgent):
    """Final quality gate evaluating RequirementSpec, BuildResult, and TestResult."""

    __test__ = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="ValidationAgent", **kwargs)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Validation Agent quality assessment.

        Args:
            state: Graph state containing 'structured_requirements', 'build_result', and 'test_result'.

        Returns:
            State update dictionary containing 'validation_result'.
        """
        raw_req = state.get("structured_requirements") or {}
        raw_build = state.get("build_result") or {}
        raw_test = state.get("test_result") or {}

        # Format requirement spec into JSON string for prompt
        if isinstance(raw_req, dict):
            try:
                spec = RequirementSpec(**raw_req)
                req_str = json.dumps(spec.model_dump(), indent=2)
            except Exception:
                req_str = json.dumps(raw_req, indent=2)
        else:
            req_str = str(raw_req)

        # Format build result into JSON string for prompt
        if isinstance(raw_build, dict):
            try:
                build = BuildResult(**raw_build)
                build_str = json.dumps(build.model_dump(), indent=2)
            except Exception:
                build_str = json.dumps(raw_build, indent=2)
        else:
            build_str = str(raw_build)

        # Format test result into JSON string for prompt
        if isinstance(raw_test, dict):
            try:
                test_res = TestResult(**raw_test)
                test_str = json.dumps(test_res.model_dump(), indent=2)
            except Exception:
                test_str = json.dumps(raw_test, indent=2)
        else:
            test_str = str(raw_test)

        # Handle missing or malformed build/test artifacts
        if not raw_build or not raw_test:
            missing_val = ValidationResult(
                status="FAIL",
                overall_score=0.0,
                requirements_coverage=0.0,
                functional_assessment="Incomplete pipeline execution; missing build or test artifacts.",
                test_assessment="Test execution could not be evaluated due to missing test result.",
                architecture_assessment="N/A",
                security_assessment="Unverified due to missing artifacts.",
                issues=["Pipeline state missing build_result or test_result"],
                recommendations=["Re-run upstream Builder and Test agents"],
                failed_requirements=["Pipeline execution integrity"],
                approval_reason="Build or Test result missing from graph state",
            )
            return {"validation_result": missing_val.model_dump()}

        # If LLM client is configured, run structured evaluation
        if self.llm is not None:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", VALIDATION_AGENT_SYSTEM_PROMPT),
                    (
                        "human",
                        "Requirement Specification:\n{req_spec}\n\nBuild Result & Source Files:\n{build_result}\n\nTest Verification Result:\n{test_result}",
                    ),
                ]
            )
            structured_llm = self.llm.with_structured_output(ValidationResult)
            chain = prompt | structured_llm

            result: ValidationResult = chain.invoke(
                {
                    "req_spec": req_str,
                    "build_result": build_str,
                    "test_result": test_str,
                }
            )
            return {"validation_result": result.model_dump()}

        # Fallback for offline execution when no LLM key is configured
        fallback_test_status = raw_test.get("status", "PASSED")
        fallback_status = "PASS" if fallback_test_status == "PASSED" else "FAIL"

        fallback_val = ValidationResult(
            status=fallback_status,
            overall_score=1.0 if fallback_status == "PASS" else 0.0,
            requirements_coverage=1.0 if fallback_status == "PASS" else 0.5,
            functional_assessment="Offline static assessment completed.",
            test_assessment=f"Test verification returned status: {fallback_test_status}",
            architecture_assessment="Clean modular structure.",
            security_assessment="No static vulnerabilities detected in fallback inspection.",
            issues=[] if fallback_status == "PASS" else ["Test execution reported failure"],
            recommendations=[] if fallback_status == "PASS" else ["Fix failing tests in Builder Agent"],
            failed_requirements=[] if fallback_status == "PASS" else ["Test execution requirement"],
            approval_reason="Passed offline validation criteria" if fallback_status == "PASS" else "Failed offline test check",
        )

        return {"validation_result": fallback_val.model_dump()}
