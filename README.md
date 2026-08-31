# Agentic Engineering Workflow

> An AI-powered software engineering workflow that uses specialized agents, parallel planning, and human-in-the-loop quality gates to transform requirements into tested and validated software.

## Overview

**Agentic Engineering Workflow** explores how multiple specialized AI agents can collaborate across the software development lifecycle.

Instead of relying on a single LLM prompt to generate an application, the system decomposes software development into specialized stages, with each agent responsible for a specific engineering capability.

```text
RequirementSpec
      │
      ├── Architecture Agent
      ├── Security Agent
      └── Test Strategy Agent
      │
      ▼
 Human Approval
      │
      ▼
  Builder Agent
      │
      ▼
   Test Agent
      │
      ▼
 Validation Agent
      │
      ▼
 Deliverable / Pass
```

The goal is to demonstrate how **agent orchestration, parallel fan-out/fan-in planning, human-in-the-loop approval, structured handoffs, automated validation, and iterative feedback loops** can be combined to create a reliable AI-assisted engineering workflow.

## Why This Project?

LLMs are increasingly capable of generating code, but generating code is only one part of software engineering.

A production-oriented engineering workflow also requires:

* Requirement understanding
* Architectural design & trade-off analysis
* Application security assessments
* Comprehensive testing strategies
* Human approval and review gates
* Automated testing & implementation
* Quality validation & iterative refactoring
* Error handling & traceability

This project explores how these responsibilities can be distributed across specialized agents and human review gates rather than delegated to a single general-purpose prompt.

## Key Concepts

* **Multi-agent orchestration**: Graph-based state machine driven by LangGraph.
* **Parallel Fan-Out / Fan-In**: Concurrent execution of Architecture, Security, and Test Strategy planning agents.
* **Human-in-the-Loop Quality Gate**: Interactive review & approval node prior to code generation.
* **Agent specialization**: Dedicated agents for requirements, architecture, security, test strategy, builder, testing, and validation.
* **Structured agent handoffs**: Pydantic schemas enforcing data contracts between agents.
* **Automated validation loops**: Quality gate evaluation triggering targeted builder retries on failure.
* **Tool execution**: Safe workspace filesystem tools for code inspection and generation.

## Project Status

Phase 3 is complete with parallel planning agents, CLI human-in-the-loop approval, and 50 passing unit tests.
