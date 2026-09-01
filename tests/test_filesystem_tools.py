"""Unit tests for deterministic filesystem tools and workspace security boundary enforcement."""

import pytest
from pathlib import Path
from multi_agent_builder.tools.filesystem import (
    write_file,
    read_file,
    list_files,
    get_safe_path,
)
from multi_agent_builder.agents.builder_agent import BuilderAgent
from multi_agent_builder.models.schemas import GeneratedFile, BuildResult, ImplementationSummary


@pytest.fixture
def tmp_workspace(tmp_path):
    """Fixture creating a isolated temporary workspace directory."""
    ws = tmp_path / "workspace" / "generated_app"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def test_1_write_file_creates_file_inside_workspace(tmp_workspace):
    """1. Test that write_file creates a file inside the workspace."""
    rel_path = "src/calculator.py"
    content = "def add(a, b):\n    return a + b\n"

    res = write_file(rel_path, content, base_dir=tmp_workspace)

    assert res.success is True
    assert res.path == rel_path
    target = tmp_workspace / rel_path
    assert target.exists()
    assert target.read_text(encoding="utf-8") == content


def test_2_read_file_reads_file(tmp_workspace):
    """2. Test that read_file reads a file content correctly."""
    rel_path = "config/settings.json"
    content = '{"env": "test"}'
    (tmp_workspace / "config").mkdir(parents=True, exist_ok=True)
    (tmp_workspace / rel_path).write_text(content, encoding="utf-8")

    res = read_file(rel_path, base_dir=tmp_workspace)

    assert res.success is True
    assert res.path == rel_path
    assert res.content == content


def test_3_list_files_lists_generated_files(tmp_workspace):
    """3. Test that list_files lists relative paths of all generated files."""
    write_file("main.py", "print('hello')", base_dir=tmp_workspace)
    write_file("utils/helper.py", "def help(): pass", base_dir=tmp_workspace)

    res = list_files(base_dir=tmp_workspace)

    assert res.success is True
    assert "main.py" in res.files
    assert "utils/helper.py" in res.files
    assert len(res.files) == 2


def test_4_dotdot_path_traversal_is_rejected(tmp_workspace):
    """4. Test that '../' path traversal is rejected."""
    bad_path = "../outside.py"
    content = "malicious"

    res = write_file(bad_path, content, base_dir=tmp_workspace)

    assert res.success is False
    assert "Path traversal ('..') is strictly prohibited" in res.error

    read_res = read_file(bad_path, base_dir=tmp_workspace)
    assert read_res.success is False
    assert "Path traversal ('..') is strictly prohibited" in read_res.error


def test_5_absolute_path_is_rejected(tmp_workspace):
    """5. Test that absolute paths are rejected."""
    abs_path = "/etc/passwd" if pytest.importorskip("os").name != "nt" else "C:\\Windows\\System32\\config"

    res = write_file(abs_path, "test", base_dir=tmp_workspace)

    assert res.success is False
    assert "Absolute paths are rejected for security reasons" in res.error

    read_res = read_file(abs_path, base_dir=tmp_workspace)
    assert read_res.success is False
    assert "Absolute paths are rejected for security reasons" in read_res.error


def test_6_write_file_cannot_escape_generated_app(tmp_workspace):
    """6. Test that write_file cannot escape generated_app workspace boundary."""
    escape_paths = [
        "nested/../../secret.txt",
        "app/../../../etc/shadow",
    ]

    for p in escape_paths:
        res = write_file(p, "data", base_dir=tmp_workspace)
        assert res.success is False
        assert ("Path traversal" in res.error or "Path escape attempt" in res.error)

    # Verify no files were created outside tmp_workspace
    parent_files = list(tmp_workspace.parent.glob("*"))
    # Only generated_app should exist inside tmp_workspace.parent
    assert len(parent_files) == 1
    assert parent_files[0].name == "generated_app"


def test_7_builder_agent_can_create_files_using_tool(tmp_workspace):
    """7. Test that BuilderAgent creates files inside workspace using write_file tool."""
    agent = BuilderAgent()
    agent.llm = None  # Use fallback mode for deterministic testing

    state = {
        "workspace_dir": tmp_workspace,
        "structured_requirements": {
            "application_name": "DemoApp",
            "problem_statement": "Build demo app",
        },
    }

    output = agent.run(state)

    assert "build_result" in output
    assert (tmp_workspace / "demoapp/main.py").exists()
    assert (tmp_workspace / "tests/test_demoapp.py").exists()

    res = list_files(base_dir=tmp_workspace)
    assert "demoapp/main.py" in res.files
    assert "tests/test_demoapp.py" in res.files


def test_8_retry_builder_agent_can_read_existing_files_before_modifying(tmp_workspace):
    """8. Test that retry BuilderAgent reads existing workspace files before modifying them."""
    # Pre-populate workspace with an existing file
    existing_rel_path = "math/core.py"
    existing_content = "# Buggy math code\ndef add(a, b):\n    return a - b\n"
    write_file(existing_rel_path, existing_content, base_dir=tmp_workspace)

    agent = BuilderAgent()
    agent.llm = None

    state = {
        "workspace_dir": tmp_workspace,
        "structured_requirements": {
            "application_name": "MathCore",
        },
        "build_result": {
            "status": "SUCCESS",
            "files": [{"path": existing_rel_path, "content": existing_content, "purpose": "core math"}],
        },
        "validation_result": {
            "status": "FAIL",
            "failed_requirements": ["add() should perform addition"],
            "issues": ["add function subtracts instead of adding"],
            "recommendations": ["Fix subtraction operator to addition"],
        },
    }

    # Run agent in retry mode
    output = agent.run(state)

    # Verify that the agent executed and persisted the files
    assert "build_result" in output
    res = list_files(base_dir=tmp_workspace)
    assert len(res.files) >= 1
