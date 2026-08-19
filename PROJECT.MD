# Multi-Agent Builder

A modular, multi-agent code generation and orchestration system built using **LangGraph**, **LangChain**, **Pydantic**, with support for **Groq** and **Google Gemini** LLMs.

---

## 🎯 Architecture Overview

The system models a software development lifecycle as an agentic state graph. Each agent has a distinct responsibility within the cycle:

```
RequirementsAgent
      ↓
BuilderAgent <──────────────────────────┐
      ↓                                 │
  TestAgent                             │
      ↓                                 │ (retry: status == FAIL & iter < MAX)
ValidationAgent                         │
      ↓                                 │
(route_after_validation)                │
 ├── "end" (status == PASS) ──────────> [ END ]
 ├── "retry" (status == FAIL) ──────────┘
 └── "max_iterations" (iter >= MAX) ──> [ END ]
```

---

## 📁 Directory Structure & File Purpose

```
multi-agent-builder/
├── .env.example                      # Template for environment variables (GROQ_API_KEY, GOOGLE_API_KEY, etc.)
├── .gitignore                         # Git exclusion pattern rules
├── pyproject.toml                     # Package metadata and CLI script definitions
├── requirements.txt                   # Dependency manifests (langgraph, langchain-groq, langchain-google-genai, etc.)
├── README.md                          # Architecture and workflow documentation
│
├── multi_agent_builder/               # Main Python package
│   ├── __init__.py                    # Package initialization
│   ├── config.py                      # Environment configuration & LLM Factory (Groq / Gemini switch)
│   ├── main.py                        # CLI entry point for running the agentic graph
│   │
│   ├── models/                        # Pydantic Schemas / Data Models
│   │   ├── __init__.py
│   │   └── schemas.py                 # Pydantic types: RequirementSpec, BuildResult, GeneratedFile, ImplementationSummary, etc.
│   │
│   ├── agents/                        # Agent Definitions
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract Base Class (BaseAgent) for LLM interactions
│   │   ├── requirements.py            # RequirementsAgent: Converts raw input -> RequirementSpec
│   │   ├── builder_agent.py          # BuilderAgent: Consumes RequirementSpec -> BuildResult
│   │   ├── test.py                    # TestAgent (Stub)
│   │   └── validation.py              # ValidationAgent (Stub)
│   │
│   └── graph/                         # LangGraph Orchestration Engine
│       ├── __init__.py
│       ├── state.py                   # AgentState TypedDict (global graph state)
│       └── workflow.py                # StateGraph assembly, node bindings, and retry conditional edge logic
│
└── tests/                             # Test Suite
    ├── __init__.py
    ├── conftest.py                    # Pytest fixtures
    ├── test_placeholder.py            # Core state and graph unit tests
    ├── test_requirements_agent.py     # RequirementsAgent & RequirementSpec unit tests
    └── test_builder_agent.py          # BuilderAgent & BuildResult mock unit tests
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
   # Recommended default (Groq - LLaMA 3.3 70B)
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
3. **Builder Agent Node**:
   - Consumes `RequirementSpec` from `AgentState`.
   - Determines technical approach, generates source files and unit tests, returning a structured `BuildResult`.
4. **Test Agent Node**:
   - Evaluates test execution output.
5. **Validation Agent Node**:
   - Evaluates code artifacts against requirements and test results to issue `PASS` or `FAIL`.
6. **Conditional Routing (`should_retry`)**:
   - **If `PASS`**: Workflow proceeds to `END` and returns final `BuildResult`.
   - **If `FAIL`** and `retry_count < max_retries`: Increment `retry_count` and loop back to the **Builder Agent** with feedback.
   - **If `FAIL`** and retry limit reached: Workflow terminates gracefully at `END` with accumulated logs.

---

## 🚀 Getting Started & Running the Application

### 1. Prerequisites & Environment Setup

Ensure you have **Python 3.10+** installed.

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

#### Custom Requirement & LLM Provider

Use `--requirement` (or `-r`) to pass your software prompt, and `--provider` (or `-p`) to select `groq`, `gemini`, or `auto`:

```bash
python -m multi_agent_builder.main \
  --requirement "Build a REST API that accepts two numbers and returns their sum and multiplication. It should validate that the inputs are numbers and return a proper error for invalid input" \
  --provider groq
```

#### Specifying Max Build Retries
```bash
python -m multi_agent_builder.main \
  --requirement "Create a Python utility module for validating email addresses" \
  --provider gemini \
  --max-retries 5
```

---

### 4. Running Unit Tests

Run the full test suite using `pytest`:

```bash
pytest
```

To run tests with detailed output:
```bash
pytest -v
```

