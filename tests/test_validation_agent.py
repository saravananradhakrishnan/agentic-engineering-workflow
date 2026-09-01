"""Unit tests for ValidationAgent and ValidationResult evaluation scenarios."""

import pytest
from unittest.mock import MagicMock
from multi_agent_builder.models.schemas import (
    RequirementSpec,
    BuildResult,
    GeneratedFile,
    ImplementationSummary,
    TestResult,
    TestCase,
    TestExecution,
    ValidationResult,
)
from multi_agent_builder.agents.validation_agent import ValidationAgent


@pytest.fixture
def sample_req_spec():
    """Sample RequirementSpec dictionary."""
    return {
        "application_name": "AuthService",
        "problem_statement": "Secure authentication service",
        "functional_requirements": ["Authenticate user", "Generate JWT token"],
        "non_functional_requirements": ["Password hashing using bcrypt", "Token expiration"],
        "api_requirements": ["login(username, password)", "verify_token(token)"],
        "data_requirements": ["User entity schema"],
        "assumptions": ["Python 3.12"],
        "acceptance_criteria": ["Successful login returns valid JWT", "Invalid password returns 401"],
    }


@pytest.fixture
def sample_build_result():
    """Sample BuildResult dictionary."""
    files = [
        GeneratedFile(
            path="auth.py",
            content="def login(username, password):\n    return 'jwt_token'\n\ndef verify_token(token):\n    return True\n",
            purpose="Auth implementation",
        )
    ]
    return BuildResult(
        status="SUCCESS",
        files=files,
        implementation_summary=ImplementationSummary(overview="Auth service", components=["auth"], key_decisions=[]),
        assumptions=[],
        potential_risks=[],
    ).model_dump()


@pytest.fixture
def sample_passed_test_result():
    """Sample passing TestResult dictionary."""
    return TestResult(
        status="PASSED",
        test_cases=[
            TestCase(id="TC-001", requirement="Authenticate user", description="Verify login", test_type="functional", expected_result="JWT token"),
        ],
        executions=[
            TestExecution(test_case_id="TC-001", status="PASSED", actual_result="login returned token"),
        ],
        passed_count=1,
        failed_count=0,
        coverage_summary="100% passed",
        issues=[],
        recommendations=[],
    ).model_dump()


def test_validation_agent_scenario_1_all_requirements_satisfied(sample_req_spec, sample_build_result, sample_passed_test_result):
    """Scenario 1: All requirements satisfied and all important tests pass -> PASS."""
    expected_val = ValidationResult(
        status="PASS",
        overall_score=1.0,
        requirements_coverage=1.0,
        functional_assessment="All functional requirements fully met.",
        test_assessment="All unit tests passed successfully.",
        architecture_assessment="Clean modular structure.",
        security_assessment="Bcrypt hashing and token validation verified.",
        issues=[],
        recommendations=[],
        failed_requirements=[],
        approval_reason="Meets all acceptance criteria and security standards.",
    )

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = expected_val
    mock_llm.with_structured_output.return_value = mock_structured_llm

    agent = ValidationAgent(llm=mock_llm)
    state = {
        "structured_requirements": sample_req_spec,
        "build_result": sample_build_result,
        "test_result": sample_passed_test_result,
    }

    with pytest.MonkeyPatch.context() as mp:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected_val
        mp.setattr("langchain_core.prompts.ChatPromptTemplate.__or__", lambda self, other: mock_chain)

        res = agent.run(state)

    assert res["validation_result"]["status"] == "PASS"
    assert res["validation_result"]["overall_score"] == 1.0
    assert len(res["validation_result"]["failed_requirements"]) == 0


def test_validation_agent_scenario_2_important_requirement_missing(sample_req_spec, sample_build_result, sample_passed_test_result):
    """Scenario 2: Important requirement missing (e.g. Password hashing missing) -> FAIL."""
    expected_val = ValidationResult(
        status="FAIL",
        overall_score=0.4,
        requirements_coverage=0.5,
        functional_assessment="Token generation present, but password hashing is missing.",
        test_assessment="Tests pass for basic login but miss password hashing verification.",
        architecture_assessment="Basic architecture.",
        security_assessment="Passwords stored/compared in plaintext.",
        issues=["Password hashing requirement not implemented"],
        recommendations=["Implement bcrypt password hashing in auth.py"],
        failed_requirements=["Password hashing using bcrypt"],
        approval_reason="Core security requirement 'Password hashing using bcrypt' is missing.",
    )

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = expected_val
    mock_llm.with_structured_output.return_value = mock_structured_llm

    agent = ValidationAgent(llm=mock_llm)
    state = {
        "structured_requirements": sample_req_spec,
        "build_result": sample_build_result,
        "test_result": sample_passed_test_result,
    }

    with pytest.MonkeyPatch.context() as mp:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected_val
        mp.setattr("langchain_core.prompts.ChatPromptTemplate.__or__", lambda self, other: mock_chain)

        res = agent.run(state)

    assert res["validation_result"]["status"] == "FAIL"
    assert "Password hashing using bcrypt" in res["validation_result"]["failed_requirements"]


def test_validation_agent_scenario_3_tests_fail(sample_req_spec, sample_build_result):
    """Scenario 3: Tests fail -> FAIL."""
    failed_test_result = TestResult(
        status="FAILED",
        test_cases=[
            TestCase(id="TC-001", requirement="Authenticate user", description="Verify login", test_type="functional", expected_result="JWT token"),
        ],
        executions=[
            TestExecution(test_case_id="TC-001", status="FAILED", actual_result="None returned", error="Authentication failed"),
        ],
        passed_count=0,
        failed_count=1,
        coverage_summary="0% passed",
        issues=["Authentication test failed"],
        recommendations=["Fix login return value"],
    ).model_dump()

    expected_val = ValidationResult(
        status="FAIL",
        overall_score=0.2,
        requirements_coverage=0.5,
        functional_assessment="Login function fails execution.",
        test_assessment="1 of 1 unit test failed.",
        architecture_assessment="Implementation present but defective.",
        security_assessment="Unverified due to failing test execution.",
        issues=["Test TC-001 failed during verification"],
        recommendations=["Fix authentication login bug in auth.py"],
        failed_requirements=["Authenticate user"],
        approval_reason="Unit test verification failed.",
    )

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = expected_val
    mock_llm.with_structured_output.return_value = mock_structured_llm

    agent = ValidationAgent(llm=mock_llm)
    state = {
        "structured_requirements": sample_req_spec,
        "build_result": sample_build_result,
        "test_result": failed_test_result,
    }

    with pytest.MonkeyPatch.context() as mp:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected_val
        mp.setattr("langchain_core.prompts.ChatPromptTemplate.__or__", lambda self, other: mock_chain)

        res = agent.run(state)

    assert res["validation_result"]["status"] == "FAIL"
    assert res["validation_result"]["issues"][0] == "Test TC-001 failed during verification"


def test_validation_agent_scenario_4_tests_pass_but_requirement_missing(sample_req_spec, sample_build_result, sample_passed_test_result):
    """Scenario 4: Tests pass but an important requirement is missing -> FAIL."""
    expected_val = ValidationResult(
        status="FAIL",
        overall_score=0.6,
        requirements_coverage=0.6,
        functional_assessment="Token verification is implemented, but token expiration mechanism is missing.",
        test_assessment="Tests pass for existing methods but do not test expiration.",
        architecture_assessment="Missing token expiration logic.",
        security_assessment="Tokens never expire, creating security risk.",
        issues=["Token expiration requirement not implemented"],
        recommendations=["Add exp claim to JWT token generation"],
        failed_requirements=["Token expiration"],
        approval_reason="Tests passed but required 'Token expiration' feature is omitted.",
    )

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = expected_val
    mock_llm.with_structured_output.return_value = mock_structured_llm

    agent = ValidationAgent(llm=mock_llm)
    state = {
        "structured_requirements": sample_req_spec,
        "build_result": sample_build_result,
        "test_result": sample_passed_test_result,
    }

    with pytest.MonkeyPatch.context() as mp:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected_val
        mp.setattr("langchain_core.prompts.ChatPromptTemplate.__or__", lambda self, other: mock_chain)

        res = agent.run(state)

    assert res["validation_result"]["status"] == "FAIL"
    assert "Token expiration" in res["validation_result"]["failed_requirements"]


def test_validation_agent_scenario_5_security_issue_identified(sample_req_spec, sample_passed_test_result):
    """Scenario 5: Security issue identified (e.g. unsafe eval() or hardcoded secret) -> FAIL."""
    unsafe_files = [
        GeneratedFile(
            path="auth.py",
            content="def login(username, password):\n    eval(username)  # Critical security flaw!\n    return 'token'\n",
            purpose="Insecure auth module",
        )
    ]
    unsafe_build = BuildResult(
        status="SUCCESS",
        files=unsafe_files,
        implementation_summary=ImplementationSummary(overview="Insecure auth", components=["auth"], key_decisions=[]),
        assumptions=[],
        potential_risks=["eval used"],
    ).model_dump()

    expected_val = ValidationResult(
        status="FAIL",
        overall_score=0.1,
        requirements_coverage=0.5,
        functional_assessment="Functionality present but dangerous.",
        test_assessment="Basic tests pass.",
        architecture_assessment="Unsafe design.",
        security_assessment="CRITICAL SECURITY FLAW: Unsanitized eval() invocation allows arbitrary code execution.",
        issues=["Critical security vulnerability: eval() used on user input"],
        recommendations=["Remove eval() call immediately and use proper parameter parsing"],
        failed_requirements=["Security assessment"],
        approval_reason="Critical security vulnerability detected in code artifacts.",
    )

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = expected_val
    mock_llm.with_structured_output.return_value = mock_structured_llm

    agent = ValidationAgent(llm=mock_llm)
    state = {
        "structured_requirements": sample_req_spec,
        "build_result": unsafe_build,
        "test_result": sample_passed_test_result,
    }

    with pytest.MonkeyPatch.context() as mp:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected_val
        mp.setattr("langchain_core.prompts.ChatPromptTemplate.__or__", lambda self, other: mock_chain)

        res = agent.run(state)

    assert res["validation_result"]["status"] == "FAIL"
    assert "CRITICAL SECURITY FLAW" in res["validation_result"]["security_assessment"]
