"""
Data Access Layer (Repositories)
=============================================================================
CONCEPT: Repository Pattern

The Repository pattern separates data access logic from business logic.
Instead of writing SQL queries directly in API routes or agent code,
we centralize all database operations here. This provides:

  1. Single source of truth — All queries in one place
  2. Testability — Mock the repository in tests, not the database
  3. Reusability — Same query used by API routes AND agent tools
  4. Abstraction — Business logic doesn't care if data comes from
     PostgreSQL, Redis cache, or an external API

Each function here is a thin wrapper around SQLAlchemy queries.
=============================================================================
"""

from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    Employee,
    User,
    PayrollRun,
    PayrollItem,
    AgentExecution,
    Approval,
    ToolAuditLog,
    ConversationMessage,
)


# =============================================================================
# Employee Repository
# =============================================================================
async def get_employee_by_code(db: AsyncSession, employee_code: str) -> Employee | None:
    """Fetch an employee by their code (e.g., 'EMP001')."""
    result = await db.execute(
        select(Employee).where(Employee.employee_code == employee_code)
    )
    return result.scalar_one_or_none()


async def get_employee_by_id(db: AsyncSession, employee_id: UUID) -> Employee | None:
    """Fetch an employee by their UUID."""
    result = await db.execute(
        select(Employee).where(Employee.id == employee_id)
    )
    return result.scalar_one_or_none()


async def list_employees(
    db: AsyncSession,
    department: str | None = None,
    is_active: bool = True,
    limit: int = 100,
    offset: int = 0,
) -> list[Employee]:
    """
    List employees with optional filtering.

    CONCEPT: Query building with SQLAlchemy
    We construct the query incrementally — adding filters only when needed.
    This pattern avoids complex conditional SQL strings.
    """
    query = select(Employee).where(Employee.is_active == is_active)

    if department:
        query = query.where(Employee.department == department)

    query = query.order_by(Employee.employee_code).limit(limit).offset(offset)
    result = await db.execute(query)
    return list(result.scalars().all())


async def search_employees(db: AsyncSession, search_term: str) -> list[Employee]:
    """Search employees by name or department (case-insensitive)."""
    pattern = f"%{search_term}%"
    result = await db.execute(
        select(Employee).where(
            (Employee.full_name.ilike(pattern)) | (Employee.department.ilike(pattern))
        )
    )
    return list(result.scalars().all())


async def count_employees(db: AsyncSession, department: str | None = None) -> int:
    """Count total active employees, optionally filtered by department."""
    query = select(func.count(Employee.id)).where(Employee.is_active == True)
    if department:
        query = query.where(Employee.department == department)
    result = await db.execute(query)
    return result.scalar_one()


# =============================================================================
# User Repository
# =============================================================================
async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
    """Fetch a user by username (for authentication)."""
    result = await db.execute(
        select(User).where(User.username == username)
    )
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: UUID) -> User | None:
    """Fetch a user by their UUID."""
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    return result.scalar_one_or_none()


# =============================================================================
# Agent Execution Repository
# =============================================================================
async def create_agent_execution(db: AsyncSession, **kwargs) -> AgentExecution:
    """Record a new agent execution."""
    execution = AgentExecution(**kwargs)
    db.add(execution)
    await db.commit()
    await db.refresh(execution)
    return execution


async def update_agent_execution(
    db: AsyncSession, execution_id: UUID, **kwargs
) -> AgentExecution | None:
    """Update an agent execution record."""
    execution = await db.get(AgentExecution, execution_id)
    if execution:
        for key, value in kwargs.items():
            setattr(execution, key, value)
        await db.commit()
        await db.refresh(execution)
    return execution


# =============================================================================
# Approval Repository
# =============================================================================
async def create_approval(db: AsyncSession, **kwargs) -> Approval:
    """Create a new approval request."""
    approval = Approval(**kwargs)
    db.add(approval)
    await db.commit()
    await db.refresh(approval)
    return approval


async def get_pending_approvals(db: AsyncSession) -> list[Approval]:
    """List all pending approvals."""
    result = await db.execute(
        select(Approval)
        .where(Approval.status == "pending")
        .order_by(Approval.created_at.desc())
    )
    return list(result.scalars().all())


async def get_approval_by_id(db: AsyncSession, approval_id: UUID) -> Approval | None:
    """Fetch a specific approval by ID."""
    return await db.get(Approval, approval_id)


# =============================================================================
# Tool Audit Log Repository
# =============================================================================
async def log_tool_call(db: AsyncSession, **kwargs) -> ToolAuditLog:
    """Record a tool call in the audit log."""
    log_entry = ToolAuditLog(**kwargs)
    db.add(log_entry)
    await db.commit()
    return log_entry


# =============================================================================
# Conversation History Repository
# =============================================================================
async def get_conversation_history(
    db: AsyncSession, thread_id: str, limit: int = 10
) -> list[ConversationMessage]:
    """
    Get recent messages for a conversation thread.

    CONCEPT: Conversation Memory
    We fetch the last N messages to provide context for the agent.
    This is the "sliding window" approach — the agent sees recent
    history but not the entire conversation (to stay within token limits).
    """
    result = await db.execute(
        select(ConversationMessage)
        .where(ConversationMessage.thread_id == thread_id)
        .order_by(ConversationMessage.created_at.desc())
        .limit(limit)
    )
    # Reverse to get chronological order
    messages = list(result.scalars().all())
    messages.reverse()
    return messages


async def add_conversation_message(
    db: AsyncSession, thread_id: str, role: str, content: str, metadata: dict | None = None
) -> ConversationMessage:
    """Add a message to a conversation thread."""
    message = ConversationMessage(
        thread_id=thread_id,
        role=role,
        content=content,
        metadata_=metadata or {},
    )
    db.add(message)
    await db.commit()
    return message
