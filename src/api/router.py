"""
API Router Aggregator
=============================================================================
CONCEPT: Router Organization

FastAPI lets you split routes across multiple files using APIRouter.
This file aggregates all routers into one, which is then mounted
on the main FastAPI app. This keeps the codebase organized:
  - health.py → /health, /ready
  - employees.py → /employees/*
  - agents.py → /agents/* (Phase 2)
  - approvals.py → /approvals/* (Phase 5)
  - auth.py → /auth/* (Phase 7)
  - documents.py → /documents/* (Phase 3)
=============================================================================
"""

from fastapi import APIRouter

from src.api.health import router as health_router
from src.api.employees import router as employees_router
from src.api.agents import router as agents_router
from src.api.documents import router as documents_router
from src.api.approvals import router as approvals_router
from src.api.auth import router as auth_router
from src.api.websocket import router as websocket_router
from src.api.stream import router as stream_router

# Main API router — aggregates all sub-routers
api_router = APIRouter()

# Mount each sub-router
api_router.include_router(health_router)
api_router.include_router(employees_router)
api_router.include_router(agents_router)          # Phase 2: Agent execution
api_router.include_router(documents_router)       # Phase 3: RAG document ingestion & search
api_router.include_router(approvals_router)       # Phase 5: Human-in-the-loop approvals
api_router.include_router(auth_router)            # Phase 7: Authentication
api_router.include_router(websocket_router)       # Phase 8: WebSocket real-time chat
api_router.include_router(stream_router)          # Phase 8: SSE streaming fallback
