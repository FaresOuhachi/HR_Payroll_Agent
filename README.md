# HR Payroll Agent

An enterprise-grade agentic AI system that automates HR and payroll operations using multi-agent orchestration, retrieval-augmented generation (RAG), human-in-the-loop approvals, and full observability.

## Overview

HR Payroll Agent is a production-ready system that combines LLM-powered AI agents with robust enterprise infrastructure to handle payroll calculations, employee data management, and HR policy compliance. The system uses a **Router Agent** to classify user intent and delegate to specialist agents, each equipped with domain-specific tools that perform precise calculations and database lookups.

Key design principles:
- **Accuracy over hallucination** — Tools handle all arithmetic and data retrieval, not the LLM
- **Human oversight** — High-risk operations pause for human approval before execution
- **Auditability** — Every tool call, agent decision, and user action is logged
- **Security-first** — RBAC, JWT auth, input validation, and tool governance at every layer

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Chat UI)                      │
│                  WebSocket / SSE Connection                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    FastAPI Application                        │
│  ┌─────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  Auth   │  │  Guardrails  │  │    Observability       │  │
│  │  (JWT)  │  │  (Validate)  │  │  (Traces/Logs/Metrics) │  │
│  └─────────┘  └──────────────┘  └────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Agent Orchestration                        │
│                                                              │
│  ┌──────────────┐    ┌─────────────────────────────────┐    │
│  │ Router Agent │───►│  Specialist Agents (LangGraph)  │    │
│  │  (Classify)  │    │  ┌─────────┐  ┌────────────┐   │    │
│  └──────────────┘    │  │ Payroll │  │  Employee   │   │    │
│                      │  │  Agent  │  │   Agent     │   │    │
│                      │  └────┬────┘  └─────┬──────┘   │    │
│                      └───────┼─────────────┼──────────┘    │
│                              │             │                │
│  ┌───────────────────────────▼─────────────▼────────────┐  │
│  │                    Payroll Tools                       │  │
│  │  get_employee_info  │  calculate_gross_pay            │  │
│  │  calculate_deductions  │  calculate_net_pay           │  │
│  │  get_leave_balance  │  calculate_department_payroll   │  │
│  │  search_employees_by_department                       │  │
│  └──────────────────────────┬───────────────────────────┘  │
└─────────────────────────────┼──────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────┐
│                      Data Layer                              │
│  ┌──────────────────┐  ┌───────────┐  ┌────────────────┐   │
│  │  PostgreSQL 16   │  │  pgvector │  │    Redis 7     │   │
│  │  (Relational DB) │  │  (RAG)    │  │  (Cache/Rate)  │   │
│  └──────────────────┘  └───────────┘  └────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

## Core Features

### Multi-Agent Orchestration
LangGraph-based stateful agent system with a **Router Agent** that classifies user intent (payroll, employee, compliance, general) and routes to specialist agents. Each agent operates as a directed graph with conditional edges, tool nodes, and automatic state checkpointing to PostgreSQL.

### Payroll Calculation Engine
Seven specialized tools the agent can invoke:
- **get_employee_info** — Retrieve full employee profile (salary, tax, benefits)
- **calculate_gross_pay** — Compute monthly or annual gross pay
- **calculate_deductions** — Itemized tax, health insurance, and retirement deductions
- **calculate_net_pay** — Complete monthly pay breakdown (gross → deductions → net)
- **get_leave_balance** — Check PTO days remaining
- **search_employees_by_department** — List all employees in a department
- **calculate_department_payroll** — Total monthly payroll cost for a department

### RAG Knowledge Base
HR policy documents are chunked (800 chars, 200 overlap), embedded via OpenAI, and stored in pgvector for similarity search. The agent retrieves relevant policies before answering compliance questions about compensation, leave, or regulatory requirements.

### Human-in-the-Loop Approvals
Operations are automatically classified by risk level. High-risk actions (e.g., payroll runs above thresholds) pause execution and create approval records. Managers approve or reject via the API, and the agent resumes from its checkpointed state.

### Real-Time Chat Interface
WebSocket-based chat UI with streaming events (reasoning steps, tool calls, tool results, final messages). Includes SSE fallback for environments without WebSocket support.

### Authentication & RBAC
JWT-based stateless authentication with a 3-tier role hierarchy:

| Role | Capabilities |
|------|-------------|
| **Admin** | Full system access — user management, audit logs, all operations |
| **Manager** | Run payroll, approve/reject operations, view all employees, reports |
| **Employee** | View own profile and payslips, use the chatbot for personal queries |

### Guardrails & Safety
- **Input Validation** — Pydantic models validate all API requests
- **Tool Governance** — Agents restricted to an allowlist of permitted tools
- **Output Validation** — Agent responses validated before delivery
- **Audit Logging** — Every tool call and agent execution recorded to database

### Conversation Memory & Checkpointing
Multi-turn conversations persisted to PostgreSQL. LangGraph checkpoints full agent state after each node transition, enabling resumable execution and conversation continuity across sessions.

### Observability
- **Distributed Tracing** — OpenTelemetry with OTLP exporter
- **Structured Logging** — JSON logs via structlog with correlation IDs
- **Metrics** — Prometheus-compatible metrics endpoint
- **LLM Tracing** — LangSmith integration for LangChain/LangGraph observability

### Evaluation Framework
Scripts to create evaluation datasets and run LLM-based evaluations using LangSmith, measuring agent accuracy, tool selection, and response quality.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | FastAPI, Uvicorn, Pydantic |
| Database | PostgreSQL 16 + pgvector, SQLAlchemy (async), Alembic |
| LLM / Agents | LangChain, LangGraph, Groq (Llama 3.1), OpenAI (embeddings) |
| Caching | Redis 7 |
| Auth | python-jose (JWT), passlib (bcrypt) |
| Real-Time | WebSockets, Server-Sent Events |
| Observability | OpenTelemetry, structlog, Prometheus, LangSmith |
| Testing | pytest, pytest-asyncio, httpx |
| Infrastructure | Docker Compose |

## Project Structure

```
hr-payroll-agent/
├── src/
│   ├── main.py                    # FastAPI entry point & lifespan
│   ├── config.py                  # Pydantic settings from .env
│   ├── api/                       # HTTP & WebSocket endpoints
│   │   ├── router.py              # Router aggregator
│   │   ├── agents.py              # Agent execution
│   │   ├── websocket.py           # WebSocket real-time chat
│   │   ├── stream.py              # SSE streaming fallback
│   │   ├── employees.py           # Employee CRUD
│   │   ├── approvals.py           # Approval workflow
│   │   ├── documents.py           # RAG document endpoints
│   │   ├── auth.py                # Login & token generation
│   │   └── health.py              # Health & readiness checks
│   ├── agents/                    # LangGraph agent logic
│   │   ├── router_agent.py        # Intent classification & routing
│   │   ├── payroll_agent.py       # Payroll specialist agent
│   │   ├── state.py               # Agent state definitions
│   │   └── callbacks.py           # Streaming event callbacks
│   ├── tools/
│   │   └── payroll_tools.py       # 7 payroll calculation tools
│   ├── db/
│   │   ├── engine.py              # Async SQLAlchemy engine
│   │   ├── models.py              # ORM models
│   │   └── repositories.py        # Database queries
│   ├── auth/
│   │   ├── jwt.py                 # JWT token creation/verification
│   │   ├── rbac.py                # Permission matrix & checks
│   │   └── dependencies.py        # FastAPI auth dependencies
│   ├── rag/
│   │   ├── embeddings.py          # Embedding model setup
│   │   ├── vectorstore.py         # pgvector interface
│   │   ├── retriever.py           # Similarity search retrieval
│   │   └── ingestion.py           # Document chunking & ingestion
│   ├── guardrails/
│   │   ├── input_validator.py     # Request validation
│   │   ├── tool_governor.py       # Tool allowlist enforcement
│   │   ├── output_validator.py    # Response validation
│   │   └── approval_workflow.py   # Risk classification & approvals
│   ├── memory/
│   │   ├── conversation.py        # Chat history management
│   │   ├── long_term.py           # Long-term memory storage
│   │   └── checkpointer.py        # LangGraph state persistence
│   ├── cache/
│   │   └── redis_client.py        # Redis caching client
│   ├── observability/
│   │   ├── tracing.py             # OpenTelemetry setup
│   │   ├── logging.py             # Structured JSON logging
│   │   └── metrics.py             # Prometheus metrics
│   └── evaluation/
│       └── evaluators.py          # LLM evaluation metrics
├── scripts/
│   ├── seed_data.py               # Populate DB with sample data
│   ├── ingest_policies.py         # Ingest HR policies into vector DB
│   ├── create_eval_dataset.py     # Create evaluation dataset
│   └── run_evaluation.py          # Run agent evaluations
├── migrations/                    # Alembic database migrations
├── tests/
│   ├── test_agents/               # Agent logic tests
│   ├── test_api/                  # API endpoint tests
│   ├── test_tools/                # Tool execution tests
│   ├── test_rag/                  # RAG functionality tests
│   └── test_guardrails/           # Guardrail tests
├── sample_data/
│   ├── employees.json             # 10 sample employees
│   └── policies/                  # HR policy documents
│       ├── compensation_policy.md
│       ├── leave_policy.md
│       └── compliance_rules.md
├── frontend/
│   ├── index.html                 # Chat interface
│   ├── app.js                     # WebSocket client
│   ├── styles.css                 # UI styling
│   └── architecture.html          # Architecture diagram
├── .env.example                   # Environment variable template
├── docker-compose.yaml            # PostgreSQL + Redis
├── alembic.ini                    # Migration config
├── Makefile                       # Common commands
└── requirements.txt               # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- API Keys:
  - **OpenAI** — for text embeddings ([platform.openai.com](https://platform.openai.com))
  - **Groq** — for LLM inference ([console.groq.com](https://console.groq.com))
  - **LangSmith** (optional) — for LLM observability ([smith.langchain.com](https://smith.langchain.com))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/hr-payroll-agent.git
cd hr-payroll-agent

# 2. Set up environment variables
cp .env.example .env
# Edit .env and fill in your API keys

# 3. Start PostgreSQL and Redis
make docker-up

# 4. Install Python dependencies
make setup

# 5. Run database migrations
make migrate

# 6. Seed sample data (10 employees + test users)
make seed

# 7. Ingest HR policy documents into the vector store
make ingest

# 8. Start the development server
make run
```

The application will be available at `http://localhost:8000` with the chat UI served at the root.

### Makefile Commands

| Command | Description |
|---------|-------------|
| `make setup` | Install Python dependencies |
| `make docker-up` | Start PostgreSQL + Redis containers |
| `make docker-down` | Stop and remove containers |
| `make migrate` | Run database migrations |
| `make migration msg="description"` | Generate a new migration |
| `make seed` | Populate DB with sample data |
| `make ingest` | Ingest HR policies into vector store |
| `make run` | Start FastAPI dev server (port 8000) |
| `make test` | Run all tests |
| `make test-cov` | Run tests with coverage report |
| `make clean` | Remove Python cache files |

## API Reference

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `POST` | `/agents/execute` | Execute agent synchronously | JWT |
| `WS` | `/agents/ws/{thread_id}` | Real-time agent chat | JWT |
| `GET` | `/agents/stream` | SSE streaming fallback | JWT |
| `GET` | `/employees` | List employees (with filters) | JWT |
| `GET` | `/employees/{employee_code}` | Get employee details | JWT |
| `GET` | `/approvals` | List pending approvals | JWT |
| `POST` | `/approvals/{approval_id}/approve` | Approve an operation | JWT |
| `POST` | `/approvals/{approval_id}/reject` | Reject an operation | JWT |
| `POST` | `/documents/ingest` | Ingest policy documents | JWT |
| `GET` | `/documents/search` | Search the knowledge base | JWT |
| `POST` | `/auth/login` | Get JWT access token | None |
| `GET` | `/health` | Health check | None |
| `GET` | `/health/ready` | Readiness check | None |

Interactive API docs available at `http://localhost:8000/docs` (Swagger UI).

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `REDIS_URL` | Redis connection string | Yes |
| `OPENAI_API_KEY` | OpenAI API key (for embeddings) | Yes |
| `GROQ_API_KEY` | Groq API key (for LLM inference) | Yes |
| `JWT_SECRET_KEY` | Secret key for signing JWT tokens | Yes |
| `JWT_ALGORITHM` | JWT signing algorithm (default: HS256) | No |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiry in minutes (default: 60) | No |
| `LANGSMITH_TRACING` | Enable LangSmith tracing | No |
| `LANGSMITH_API_KEY` | LangSmith API key | No |
| `LANGSMITH_ENDPOINT` | LangSmith API endpoint | No |
| `LANGSMITH_PROJECT` | LangSmith project name | No |
| `APP_NAME` | Application name | No |
| `APP_ENV` | Environment (development/production) | No |
| `DEBUG` | Enable debug mode | No |
| `LOG_LEVEL` | Logging level (default: INFO) | No |

## Testing

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run a specific test suite
pytest tests/test_agents/ -v
pytest tests/test_tools/ -v
pytest tests/test_rag/ -v
```

## License

This project is licensed under the MIT License.
