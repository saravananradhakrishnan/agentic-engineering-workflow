"""Pytest configuration and shared fixtures."""

import pytest
from typing import Dict, Any


@pytest.fixture(autouse=True)
def disable_real_llm(monkeypatch):
    """Ensure unit tests use offline fallbacks or explicit mocks rather than calling real LLM APIs."""
    monkeypatch.setattr("multi_agent_builder.agents.base.get_llm", lambda **kwargs: None)
    monkeypatch.setattr("multi_agent_builder.config.get_llm", lambda **kwargs: None)


@pytest.fixture
def sample_user_state() -> Dict[str, Any]:
    """Sample state dictionary for graph initialization."""
    return {
        "user_requirement": "Create a function to compute factorial",
        "retry_count": 0,
        "max_retries": 3,
        "logs": [],
    }
