"""
Health Check Endpoints
=============================================================================
CONCEPT: Health Checks

Production systems need two types of health checks:
  1. /health (Liveness) — "Is the process running?"
     Used by Kubernetes to know if the container is alive.
     If this fails, K8s restarts the container.

  2. /ready (Readiness) — "Can it handle requests?"
     Checks if dependencies (DB, Redis) are accessible.
     If this fails, K8s stops sending traffic to this instance
     (but doesn't restart it — it might just be waiting for DB).

WHY separate endpoints?
  A server might be running (/health OK) but not ready to serve
  (e.g., DB connection pool is initializing, or Redis is down).
=============================================================================
"""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.engine import get_db_session

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """
    Liveness probe — Is the service running?
    Always returns 200 if the process is alive.
    """
    return {"status": "ok", "service": "hr-payroll-agent"}


@router.get("/ready")
async def readiness_check(db: AsyncSession = Depends(get_db_session)):
    """
    Readiness probe — Can the service handle requests?
    Checks database connectivity by running a simple query.
    """
    checks = {}

    # Check PostgreSQL
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {str(e)}"

    all_ok = all(v == "ok" for v in checks.values())

    return {
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
    }
