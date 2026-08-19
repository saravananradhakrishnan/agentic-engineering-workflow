"""Builder Agent implementation.

Consumes a RequirementSpec and generates source code files, unit tests, and implementation metadata as a structured BuildResult.
"""

import json
from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate
from multi_agent_builder.agents.base import BaseAgent
from multi_agent_builder.models.schemas import (
    RequirementSpec,
    BuildResult,
    GeneratedFile,
    ImplementationSummary,
)
from multi_agent_builder.tools.filesystem import (
    write_file,
    read_file,
    list_files,
)


BUILDER_SYSTEM_PROMPT = """You are an Expert Software Engineer and Code Generator.
Your task is to review the provided technical RequirementSpec and construct a complete, working Python implementation along with comprehensive unit tests.

Guidelines:
1. Implementation Approach:
   - Design clean, modular Python source code adhering to Python 3.12+ best practices.
   - Address all functional and non-functional requirements.
   - Implement the specified API/function signatures and data structures.

2. File Generation:
   - Provide relative file paths (e.g., `app/calculator.py`, `tests/test_calculator.py`).
   - Include complete, runnable code inside the `content` field for each file (no placeholders or truncated snippets).
   - Provide unit tests covering positive and edge cases.

3. Structured Output:
   - Package all output into a BuildResult object containing:
     - status: "SUCCESS" or "FAILED"
     - files: List of GeneratedFile objects (source files and test files)
     - implementation_summary: Overview, components list, and key architectural decisions
     - assumptions: Key assumptions made during code construction
     - potential_risks: Potential risks, edge cases, or performance limitations
"""


class BuilderAgent(BaseAgent):
    """Consumes RequirementSpec and produces a structured BuildResult while operating on workspace files using filesystem tools."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="BuilderAgent", **kwargs)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Builder Agent logic.

        Args:
            state: Graph state containing 'structured_requirements' (RequirementSpec dict).

        Returns:
            State update dictionary containing 'build_result' and updated 'code_artifacts'.
        """
        raw_req = state.get("structured_requirements") or {}
        if isinstance(raw_req, dict):
            try:
                spec = RequirementSpec(**raw_req)
                spec_str = json.dumps(spec.model_dump(), indent=2)
            except Exception:
                spec_str = json.dumps(raw_req, indent=2)
        else:
            spec_str = str(raw_req)

        workspace_dir = state.get("workspace_dir")

        # Check for feedback from previous validation attempt (if any)
        val_res = state.get("validation_result") or {}
        prev_build = state.get("build_result") or {}

        is_retry = bool(val_res and val_res.get("status") == "FAIL")

        if is_retry:
            # Tool usage: inspect workspace files via list_files & read_file
            listed_files = list_files(base_dir=workspace_dir)
            workspace_file_contents = {}
            if listed_files.success:
                for file_path in listed_files.files:
                    read_res = read_file(file_path, base_dir=workspace_dir)
                    if read_res.success and read_res.content is not None:
                        workspace_file_contents[file_path] = read_res.content

            failed_reqs = val_res.get("failed_requirements", [])
            issues = val_res.get("issues", [])
            recs = val_res.get("recommendations", [])
            prev_files = prev_build.get("files", [])

            context_prompt = (
                "=== RETRY BUILD MODE ===\n"
                "The previous implementation FAILED validation. Use tool outputs and feedback for code revisions:\n\n"
                f"Failed Requirements:\n{json.dumps(failed_reqs, indent=2)}\n\n"
                f"Validation Issues Identified:\n{json.dumps(issues, indent=2)}\n\n"
                f"Actionable Recommendations:\n{json.dumps(recs, indent=2)}\n\n"
                f"Previous Build Implementation Files:\n{json.dumps(prev_files, indent=2)}\n\n"
                f"Existing Workspace File Contents (read via read_file tool):\n{json.dumps(workspace_file_contents, indent=2)}\n\n"
                "INSTRUCTION: Address all failed requirements, issues, and recommendations. Improve and fix the code rather than starting from scratch."
            )
        else:
            context_prompt = "=== FIRST BUILD MODE ===\nConstruct a clean initial implementation based on RequirementSpec."

        # If LLM client is configured, run structured output generation
        if self.llm is not None:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", BUILDER_SYSTEM_PROMPT),
                    (
                        "human",
                        "Requirement Specification:\n{req_spec}\n\nBuild Mode & Context:\n{context_prompt}",
                    ),
                ]
            )
            structured_llm = self.llm.with_structured_output(BuildResult)
            chain = prompt | structured_llm

            result: BuildResult = chain.invoke(
                {
                    "req_spec": spec_str,
                    "context_prompt": context_prompt,
                }
            )
        else:
            # Fallback for offline execution when no LLM key is configured
            app_name = raw_req.get("application_name", "AppModule") if isinstance(raw_req, dict) else "AppModule"
            fallback_files = [
                GeneratedFile(
                    path=f"{app_name.lower()}/main.py",
                    content=f"# Implementation for {app_name}\ndef execute():\n    return True\n",
                    purpose="Primary module entry point",
                ),
                GeneratedFile(
                    path=f"tests/test_{app_name.lower()}.py",
                    content=f"from {app_name.lower()}.main import execute\n\ndef test_execute():\n    assert execute() is True\n",
                    purpose="Unit test suite",
                ),
            ]
            fallback_summary = ImplementationSummary(
                overview=f"Modular implementation created for {app_name}",
                components=[f"{app_name.lower()}.main", f"tests.test_{app_name.lower()}"],
                key_decisions=["Decoupled core execution from test harness"],
            )
            result = BuildResult(
                status="SUCCESS",
                files=fallback_files,
                implementation_summary=fallback_summary,
                assumptions=["Standard Python 3.12 environment"],
                potential_risks=["Offline stub fallback mode; requires LLM API key for dynamic generation"],
            )

        # Tool usage: Persist all generated files into workspace/generated_app using write_file tool
        for gen_file in result.files:
            write_file(gen_file.path, gen_file.content, base_dir=workspace_dir)

        code_artifacts = [
            {
                "filename": f.path,
                "content": f.content,
                "language": "python",
            }
            for f in result.files
        ]

        return {
            "build_result": result.model_dump(),
            "code_artifacts": code_artifacts,
        }

