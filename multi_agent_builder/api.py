"""FastAPI web application and REST API endpoints for multi-agent-builder."""

import logging
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from multi_agent_builder.config import settings
from multi_agent_builder.graph.workflow import build_graph

logger = logging.getLogger("multi_agent_builder.api")

#Main API initilization
app = FastAPI(
    title="Multi-Agent Builder Service",
    description="REST API for multi-agent code generation and orchestration built with LangGraph",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., json_schema_extra={"example": "healthy"})


class BuildRequest(BaseModel):
    """Build workflow execution request schema."""

    user_requirement: str = Field(
        ...,
        description="Software requirement description for multi-agent builder.",
        json_schema_extra={"example": "Create a Python utility module that calculates Fibonacci sequences."},
    )
    max_retries: Optional[int] = Field(
        default=settings.MAX_BUILD_RETRIES,
        description="Maximum builder retries allowed on validation failure.",
    )
    provider: Optional[str] = Field(
        default=settings.LLM_PROVIDER,
        description="LLM provider override ('groq', 'gemini', or 'auto').",
    )


class BuildResponse(BaseModel):
    """Build workflow execution response schema."""

    status: str
    message: str
    final_state: Dict[str, Any]


@app.get("/", summary="Root status endpoint")
async def root() -> Dict[str, str]:
    """Root endpoint returning service identity and health state."""
    return {
        "service": "multi-agent-builder",
        "version": "0.1.0",
        "status": "healthy",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description="Endpoint for load balancers and AWS health checks. Must return healthy status.",
)
async def health_check() -> HealthResponse:
    """Return health check status."""
    return HealthResponse(status="healthy")


@app.post(
    "/api/v1/build",
    response_model=BuildResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger code generation build workflow",
)
async def trigger_build(request: BuildRequest) -> BuildResponse:
    """Run the multi-agent graph workflow for a user requirement."""
    try:
        graph = build_graph()
        initial_state: Dict[str, Any] = {
            "user_requirement": request.user_requirement,
            "iteration_count": 0,
            "max_iterations": request.max_retries or settings.MAX_BUILD_RETRIES,
            "logs": [],
        }

        final_state: Dict[str, Any] = {}
        for event in graph.stream(initial_state):
            for node_name, state_update in event.items():
                final_state.update(state_update)

        return BuildResponse(
            status="completed",
            message="Workflow executed successfully",
            final_state=final_state,
        )
    except Exception as e:
        logger.exception("Error executing multi-agent graph workflow")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failed: {str(e)}",
        )
