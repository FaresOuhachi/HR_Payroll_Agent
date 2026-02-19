"""
Employee API Endpoints
=============================================================================
CONCEPT: RESTful API Design

REST (Representational State Transfer) is an architectural style for APIs.
Key principles:
  1. Resources — Everything is a "resource" (employees, payroll runs, etc.)
  2. HTTP Methods — Map to CRUD operations:
     GET    = Read (list or get one)
     POST   = Create
     PUT    = Update (full replace)
     PATCH  = Update (partial)
     DELETE = Delete
  3. Status Codes — Communicate outcome:
     200 = OK, 201 = Created, 404 = Not Found, 422 = Validation Error

CONCEPT: Pydantic Schemas
FastAPI uses Pydantic models for:
  - Request validation — Automatically validates incoming JSON
  - Response serialization — Controls what fields are returned
  - Documentation — Auto-generates OpenAPI/Swagger docs
=============================================================================
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.engine import get_db_session
from src.db import repositories as repo

router = APIRouter(prefix="/employees", tags=["Employees"])


# =============================================================================
# Pydantic Schemas — Define request/response shapes
# =============================================================================
class EmployeeResponse(BaseModel):
    """
    Response schema for employee data.

    CONCEPT: Response schemas control what data is exposed to the client.
    We might have sensitive fields in the DB (SSN, bank account) that
    should never appear in API responses. The schema acts as a filter.
    """
    id: UUID
    employee_code: str
    full_name: str
    email: str | None
    department: str
    position: str | None
    salary_info: dict
    benefits_info: dict
    is_active: bool

    class Config:
        from_attributes = True  # Allow creating from SQLAlchemy model


class EmployeeListResponse(BaseModel):
    """Paginated list of employees."""
    employees: list[EmployeeResponse]
    total: int


# =============================================================================
# Endpoints
# =============================================================================
@router.get("", response_model=EmployeeListResponse)
async def list_employees(
    department: str | None = Query(None, description="Filter by department"),
    limit: int = Query(50, ge=1, le=100, description="Max results per page"),
    offset: int = Query(0, ge=0, description="Skip N results (pagination)"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    List all active employees with optional filtering.

    CONCEPT: Pagination
    For large datasets, we don't return everything at once.
    limit=50, offset=0 → first 50 results
    limit=50, offset=50 → next 50 results
    This prevents memory issues and slow responses.
    """
    employees = await repo.list_employees(
        db, department=department, limit=limit, offset=offset
    )
    total = await repo.count_employees(db, department=department)
    return EmployeeListResponse(
        employees=[EmployeeResponse.model_validate(e) for e in employees],
        total=total,
    )


@router.get("/search")
async def search_employees(
    q: str = Query(..., min_length=1, description="Search term"),
    db: AsyncSession = Depends(get_db_session),
):
    """Search employees by name or department."""
    employees = await repo.search_employees(db, q)
    return {
        "results": [EmployeeResponse.model_validate(e) for e in employees],
        "count": len(employees),
    }


@router.get("/{employee_code}", response_model=EmployeeResponse)
async def get_employee(
    employee_code: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get a specific employee by their code.

    CONCEPT: HTTP 404
    If the resource doesn't exist, we return 404 Not Found.
    This is the standard REST response for "resource not found".
    """
    employee = await repo.get_employee_by_code(db, employee_code)
    if not employee:
        raise HTTPException(status_code=404, detail=f"Employee {employee_code} not found")
    return EmployeeResponse.model_validate(employee)
