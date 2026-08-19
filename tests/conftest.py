"""Pytest configuration and shared fixtures."""

import pytest
from typing import Dict, Any


@pytest.fixture
def sample_user_state() -> Dict[str, Any]:
    """Sample state dictionary for graph initialization."""
    return {
        "user_requirement": "Create a function to compute factorial",
        "retry_count": 0,
        "max_retries": 3,
        "logs": [],
    }
