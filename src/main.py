"""
FastAPI Application Entry Point
=============================================================================
CONCEPT: FastAPI Application Lifecycle

FastAPI applications follow this lifecycle:
  1. Startup — Initialize resources (DB connections, caches, etc.)
  2. Request handling — Process HTTP requests using routes
  3. Shutdown — Clean up resources (close DB connections, flush logs)

We use the `lifespan` context manager pattern (recommended over the older
`@app.on_event("startup")` pattern). This ensures proper cleanup even if
the server crashes.

CONCEPT: ASGI (Asynchronous Server Gateway Interface)
FastAPI is an ASGI framework — it handles requests asynchronously.
This means while one request is waiting for a DB query, the server
can handle other requests. This is much more efficient than traditional
WSGI (like Flask/Django) for I/O-bound workloads.

Run with: uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
=============================================================================
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import settings
from src.api.router import api_router
from src.db.engine import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    CONCEPT: Resource Management
    Everything before `yield` runs on startup.
    Everything after `yield` runs on shutdown.
    This pattern guarantees cleanup happens, similar to try/finally.
    """
    # === STARTUP ===
    print(f"Starting {settings.app_name} ({settings.app_env})")

    # LangSmith tracing status
    if settings.langsmith_tracing and settings.langsmith_api_key:
        print(f"LangSmith tracing enabled (project: {settings.langsmith_project})")
    else:
        print("LangSmith tracing disabled")

    # Verify database connection
    async with engine.begin() as conn:
        from sqlalchemy import text
        await conn.execute(text("SELECT 1"))
        print("Database connection verified")

    # Initialize the payroll graph with PostgreSQL checkpointer.
    # The checkpointer context manager must stay open for the app's lifetime,
    # so we use `async with` inside the lifespan context.
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from src.agents.payroll_agent import init_payroll_graph

    db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    async with AsyncPostgresSaver.from_conn_string(db_url) as checkpointer:
        await checkpointer.setup()
        init_payroll_graph(checkpointer)
        print("Agent checkpointer initialized")

        yield  # Application is running and handling requests

    # === SHUTDOWN ===
    print("Shutting down...")
    await engine.dispose()  # Close all DB connections in the pool
    print("Database connections closed")


# =============================================================================
# Create the FastAPI application
# =============================================================================
app = FastAPI(
    title=settings.app_name,
    description=(
        "Agentic AI system for HR & Payroll automation. "
        "Features multi-agent orchestration, RAG, human-in-the-loop approvals, "
        "and full observability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Middleware
# =============================================================================
# CONCEPT: CORS (Cross-Origin Resource Sharing)
# Browsers block requests from one domain to another by default (security).
# CORS middleware tells the browser "it's OK for the frontend to call this API".
# In production, restrict `allow_origins` to your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # In production: ["https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Mount Routers
# =============================================================================
app.include_router(api_router)


# =============================================================================
# Serve Frontend Static Files (Phase 8)
# =============================================================================
# Mount the frontend directory so the chat UI is accessible at http://localhost:8000/
# StaticFiles serves HTML/CSS/JS without any build step.
import os
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
