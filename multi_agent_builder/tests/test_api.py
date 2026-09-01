"""Unit tests for FastAPI endpoints and health check URL in multi_agent_builder.api."""

import pytest
from fastapi.testclient import TestClient
from multi_agent_builder.api import app


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def test_health_check_endpoint(client):
    """Test GET /health returns 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data == {"status": "healthy"}


def test_root_endpoint(client):
    """Test GET / returns 200 OK with service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["service"] == "multi-agent-builder"
    assert json_data["status"] == "healthy"
    assert "version" in json_data


def test_trigger_build_endpoint(client, monkeypatch):
    """Test POST /api/v1/build invokes graph workflow."""
    # Mock build_graph to return a fake compiled graph for testing
    class DummyGraph:
        def stream(self, initial_state):
            yield {
                "requirements": {
                    "structured_requirements": {
                        "task_name": "Factorial",
                        "description": "Test requirement",
                    }
                }
            }

    monkeypatch.setattr("multi_agent_builder.api.build_graph", lambda: DummyGraph())

    payload = {
        "user_requirement": "Create a python function to compute factorial",
        "max_retries": 2,
    }
    response = client.post("/api/v1/build", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "completed"
    assert "structured_requirements" in json_data["final_state"]
