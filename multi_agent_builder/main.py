"""CLI Entry point for multi-agent-builder graph runner."""

import argparse
import json
import sys
from typing import Any, Dict
from multi_agent_builder.graph.workflow import build_graph
from multi_agent_builder.config import settings


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Builder: LangGraph orchestrated multi-agent code generator."
    )
    parser.add_argument(
        "--requirement",
        "-r",
        type=str,
        default="Create a Python utility module that calculates Fibonacci sequences.",
        help="Raw software requirement prompt to process.",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["groq", "gemini", "auto"],
        default=settings.LLM_PROVIDER,
        help="LLM provider to use (groq, gemini, or auto).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=settings.MAX_BUILD_RETRIES,
        help="Maximum builder retries allowed on test/validation failure.",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Launch the Uvicorn ASGI web server for multi-agent-builder API.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind Uvicorn server (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind Uvicorn server (default: 8000).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the multi-agent graph with user requirements or start Uvicorn web server."""
    args = parse_args()

    if args.server:
        import uvicorn

        print("==================================================")
        print("    STARTING MULTI-AGENT BUILDER UVICORN SERVER   ")
        print("==================================================")
        print(f"Host: {args.host}")
        print(f"Port: {args.port}")
        print("Healthcheck URL: http://{args.host}:{args.port}/health")
        print("--------------------------------------------------")
        uvicorn.run("multi_agent_builder.api:app", host=args.host, port=args.port)
        return

    print("==================================================")
    print("         MULTI-AGENT BUILDER (LangGraph)          ")
    print("==================================================")
    print(f"LLM Provider : {args.provider}")
    print(f"Max Retries  : {args.max_retries}")
    print(f"Requirement  : {args.requirement}")
    print("--------------------------------------------------")

    graph = build_graph()

    initial_state: Dict[str, Any] = {
        "user_requirement": args.requirement,
        "iteration_count": 0,
        "max_iterations": args.max_retries,
        "logs": [],
    }

    try:
        current_iter = 0
        final_state = {}

        planning_printed = False

        # Stream graph updates to track iteration progress
        for event in graph.stream(initial_state):
            for node_name, state_update in event.items():
                final_state.update(state_update)

                if node_name == "requirements":
                    print("\n========================================")
                    print("REQUIREMENTS")
                    print("========================================")
                    print("[+] Requirements Agent processed specification.")

                elif node_name in ["architecture", "security", "test_strategy"]:
                    if not planning_printed:
                        print("\n========================================")
                        print("PARALLEL PLANNING")
                        print("========================================")
                        planning_printed = True
                    if node_name == "architecture":
                        print("Architecture Agent: COMPLETE")
                    elif node_name == "security":
                        print("Security Agent: COMPLETE")
                    elif node_name == "test_strategy":
                        print("Test Strategy Agent: COMPLETE")

                elif node_name == "human_approval":
                    status = state_update.get("human_approval", "N/A")
                    if status == "REJECTED":
                        print("\n========================================")
                        print("HUMAN APPROVAL")
                        print("========================================")
                        print("Status: REJECTED")
                        feedback = state_update.get("human_feedback", "")
                        if feedback:
                            print(f"Feedback: {feedback}")
                        print("Workflow terminated.")

                elif node_name == "builder":
                    iter_num = state_update.get("iteration_count", current_iter + 1)
                    if iter_num != current_iter:
                        current_iter = iter_num
                    print("\n========================================")
                    print("BUILDER")
                    print("========================================")
                    files = state_update.get("build_result", {}).get("files", [])
                    print(f"Generated {len(files)} files.")

                elif node_name == "test":
                    print("\n========================================")
                    print("TEST")
                    print("========================================")
                    test_res = state_update.get("test_result", {})
                    status = test_res.get("status", "N/A")
                    passed = test_res.get("passed_count", 0)
                    failed = test_res.get("failed_count", 0)
                    print(f"Status: {status} (Passed: {passed}, Failed: {failed})")

                elif node_name == "validation":
                    print("\n========================================")
                    print("VALIDATION")
                    print("========================================")
                    val_res = state_update.get("validation_result", {})
                    status = val_res.get("status", "N/A")
                    reason = val_res.get("approval_reason", "N/A")
                    print(f"Status: {status}")
                    print(f"Reason: {reason}")

        print("\n========================================")
        print("[+] Workflow Execution Completed!")
        print("========================================")
        print("\n--- Final State Summary ---")

        if "structured_requirements" in final_state:
            print(
                f"\nStructured Requirements:\n{json.dumps(final_state['structured_requirements'], indent=2)}"
            )

        if "build_result" in final_state:
            print(
                f"\nBuild Result:\n{json.dumps(final_state['build_result'], indent=2)}"
            )

        if "test_result" in final_state:
            print(
                f"\nTest Result:\n{json.dumps(final_state['test_result'], indent=2)}"
            )

        if "validation_result" in final_state:
            print(
                f"\nValidation Result:\n{json.dumps(final_state['validation_result'], indent=2)}"
            )

    except Exception as e:
        print(f"\n[-] Error running graph workflow: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
