# Multi-Agent Builder

A modular, multi-agent code generation and orchestration system built using **LangGraph**, **LangChain**, **Pydantic**, with support for **Groq** and **Google Gemini** LLMs.

---

## 🎯 Architecture Overview

The system models a software development lifecycle as an agentic state graph with parallel planning and human-in-the-loop approval:

```
                      RequirementsAgent
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
    ArchitectureAgent  SecurityAgent  TestStrategyAgent
            └────────────────┬────────────────┘
                             ▼
                      Human Approval
                       ┌─────┴─────┐
                       ▼           ▼
                    APPROVED    REJECTED ─────────> [ END ]
                       │
                       ▼
                 BuilderAgent <───────────────────┐
                       │                          │
                       ▼                          │
                   TestAgent                      │
                       │                          │ (retry: status == FAIL & iter < MAX)
               ValidationAgent                    │
                       │                          │
            (route_after_validation)              │
             ├── "end" (status == PASS) ───────> [ END ]
             ├── "retry" (status == FAIL) ────────┘
             └── "max_iterations" (iter >= MAX) > [ END ]
```

---

## 📁 Directory Structure & File Purpose

```
multi-agent-builder/
├── .env.example                      # Template for environment variables (GROQ_API_KEY, GOOGLE_API_KEY, etc.)
├── .gitignore                         # Git exclusion pattern rules
├── pyproject.toml                     # Package metadata, test paths, and CLI script definitions
├── requirements.txt                   # Dependency manifests (langgraph, langchain-groq, langchain-google-genai, etc.)
│
├── multi_agent_builder/               # Main Python package
│   ├── __init__.py                    # Package initialization
│   ├── config.py                      # Environment configuration & LLM Factory (Groq / Gemini switch)
│   ├── main.py                        # CLI entry point with streaming phase output
│   ├── PROJECT.md                     # Technical architecture documentation
│   ├── README.md                      # High-level workflow documentation
│   │
│   ├── models/                        # Pydantic Schemas / Data Models
│   │   ├── __init__.py
│   │   └── schemas.py                 # Schemas: RequirementSpec, ArchitecturePlan, SecurityAssessment, TestStrategy, BuildResult, etc.
│   │
│   ├── agents/                        # Agent Definitions
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract Base Class (BaseAgent) for LLM interactions
│   │   ├── requirements.py            # RequirementsAgent: Converts raw input -> RequirementSpec
│   │   ├── architecture_agent.py      # ArchitectureAgent: Produces ArchitecturePlan
│   │   ├── security_agent.py          # SecurityAgent: Produces SecurityAssessment
│   │   ├── test_strategy_agent.py     # TestStrategyAgent: Produces TestStrategy
│   │   ├── builder_agent.py           # BuilderAgent: Consumes RequirementSpec + Planning Plans -> BuildResult
│   │   ├── test_agent.py              # TestAgent: Evaluates test execution output
│   │   └── validation_agent.py        # ValidationAgent: Quality gate issuing PASS/FAIL
│   │
│   ├── graph/                         # LangGraph Orchestration Engine
│   │   ├── __init__.py
│   │   ├── state.py                   # AgentState TypedDict (global graph state)
│   │   └── workflow.py                # StateGraph assembly with parallel fan-out/fan-in and human approval
│   │
│   └── tests/                         # Test Suite
│       ├── __init__.py
│       ├── conftest.py                # Pytest fixtures and offline LLM disabling
│       ├── test_builder_agent.py      # BuilderAgent unit tests
│       ├── test_filesystem_tools.py   # Workspace filesystem tool unit tests
│       ├── test_graph_feedback_loop.py# Graph retry loop unit tests
│       ├── test_phase3_parallel_and_human.py # Parallel planning & human-in-the-loop unit tests
│       ├── test_requirements_agent.py # RequirementsAgent unit tests
│       ├── test_test_agent.py         # TestAgent unit tests
│       └── test_validation_agent.py   # ValidationAgent unit tests
```

---

## ⚙️ Environment & LLM Configuration

The project supports both **Groq** and **Google Gemini**.

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Add your API keys in `.env`:
   ```env
   # Recommended default (Groq - LLaMA 3.3 70B / Qwen 2.5 32B)
   GROQ_API_KEY=gsk_your_groq_api_key

   # Google Gemini option
   GOOGLE_API_KEY=your_google_api_key

   # Choose 'groq', 'gemini', or 'auto' (automatically selects based on available key)
   LLM_PROVIDER=auto
   ```

---

## 🔁 How the LangGraph Workflow Operates

1. **User Requirement**: The workflow receives an initial `user_requirement` prompt.
2. **Requirements Agent Node**:
   - Takes `user_requirement` from `AgentState`.
   - Uses LLM to parse and extract structured specifications into a Pydantic `RequirementSpec`.
3. **Parallel Planning Fan-Out**:
   - LangGraph fans out execution across three independent, parallel planning agents consuming `RequirementSpec`:
     - **ArchitectureAgent**: Generates `ArchitecturePlan` (architectural style, components, data flow, API design, trade-offs).
     - **SecurityAgent**: Generates `SecurityAssessment` (threats, input validation, encryption, access control).
     - **TestStrategyAgent**: Generates `TestStrategy` (functional, API, edge case, and negative test plans).
4. **Parallel Fan-In & Human Approval Node**:
   - All three planning branches fan in to `create_human_approval_node`.
   - Displays the planning artifacts to the CLI reviewer and prompts `Approve implementation? [y/N]:`.
   - If approved (`y`), sets `human_approval = "APPROVED"` and routes to `BuilderAgent`.
   - If rejected, sets `human_approval = "REJECTED"`, records `human_feedback`, and terminates the workflow immediately at `END`.
5. **Builder Agent Node**:
   - Consumes `RequirementSpec`, `ArchitecturePlan`, `SecurityAssessment`, `TestStrategy`, plus previous retry build & validation results (if retrying).
   - Generates modular source code files and unit tests into a structured `BuildResult`.
6. **Test Agent Node**:
   - Evaluates generated test cases and execution outputs.
7. **Validation Agent Node**:
   - Acts as quality gate, evaluating code artifacts against requirements and test results to issue `PASS` or `FAIL`.
8. **Conditional Routing (`route_after_validation`)**:
   - **If `PASS`**: Workflow proceeds to `END`.
   - **If `FAIL`** and `iteration_count < max_iterations`: Increments `iteration_count` and loops back to **BuilderAgent** with feedback.
   - **If `FAIL`** and retry limit reached: Workflow terminates gracefully at `END`.

---

## 🚀 Getting Started & Running the Application

### 1. Prerequisites & Environment Setup

Ensure you have **Python 3.12+** installed.

#### Step 1: Clone the repository and navigate to the project root
```bash
cd multi-agent-builder
```

> [!IMPORTANT]
> **Working Directory Note**: Always execute commands from the **root directory** of the repository (`multi-agent-builder`), NOT from within the inner `multi_agent_builder/` subfolder. Running python module commands inside the subfolder will trigger `ModuleNotFoundError`.

#### Step 2: Create and activate a virtual environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate

# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

#### Step 3: Install dependencies
```bash
# Option A: Install in editable development mode (recommended)
pip install -e .

# Option B: Install requirements directly
pip install -r requirements.txt
```

---

### 2. Configure API Keys

1. Copy `.env.example` to create your local `.env` file:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and configure your API keys and default provider preferences:
   ```env
   # Provider API Keys
   GROQ_API_KEY=gsk_your_groq_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here

   # Provider selection: 'groq', 'gemini', or 'auto'
   MODEL_TYPE=auto

   # Model choices
   GROQ_MODEL=qwen/qwen3.6-27b
   GEMINI_MODEL=gemini-2.5-flash

   # Orchestration settings
   MAX_BUILD_RETRIES=3
   ```

---

### 3. Running the Multi-Agent Builder

#### Basic Execution (Default Prompt & Settings)
```bash
python -m multi_agent_builder.main
```

#### Web UI Execution 
```bash
uvicorn multi_agent_builder.api:app --reload
```

#### Custom Requirement & LLM Provider
Use `--requirement` (or `-r`) to pass your software prompt, and `--provider` (or `-p`) to select `groq`, `gemini`, or `auto`:

```bash
python -m multi_agent_builder.main \
  --requirement "Build a REST API that accepts two numbers and returns their sum and multiplication." \
  --provider groq
```

---

### 4. Running with Docker

You can build and run the web service locally using Docker or Docker Compose.

#### Using Docker Compose (Recommended)
```bash
# Start the containerized service (reads environment variables from your .env file)
docker-compose up --build
```

#### Using Docker CLI
```bash
# 1. Build the image
docker build -t multi-agent-builder:latest .

# 2. Run the container
docker run -d -p 8000:8000 \
  -e GOOGLE_API_KEY="your_google_api_key_here" \
  -e GROQ_API_KEY="your_groq_api_key_here" \
  -e LLM_PROVIDER="auto" \
  multi-agent-builder:latest
```

---

### 5. Running Unit Tests

Run the complete test suite using `pytest`:

```bash
pytest
```

To run tests with detailed output:
```bash
pytest -v
```