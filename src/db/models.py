"""
Database Models (SQLAlchemy ORM)
=============================================================================
CONCEPT: ORM Models

Each class below maps to a PostgreSQL table. SQLAlchemy translates:
  - Class attributes → Table columns
  - Python types → SQL types (str → VARCHAR, float → NUMERIC, etc.)
  - Relationships → Foreign keys + JOINs

WHY ORM instead of raw SQL?
  1. Type safety — Python catches errors at development time
  2. Migrations — Alembic tracks schema changes automatically
  3. Relationships — Navigate between tables using Python attributes
  4. Security — Parameterized queries prevent SQL injection by default

TABLE DESIGN OVERVIEW:
  - users: Authentication & authorization (roles: admin, manager, employee)
  - employees: HR data (name, department, salary, tax info)
  - payroll_runs: Monthly payroll batches
  - payroll_items: Individual pay calculations within a run
  - agent_executions: Audit trail of every AI agent run
  - approvals: Human-in-the-loop approval requests
  - tool_audit_log: Record of every tool the agent called
  - documents: RAG knowledge base (text chunks + vector embeddings)
  - conversation_history: Chat memory for multi-turn conversations
=============================================================================
"""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from src.db.engine import Base


def utcnow():
    """Helper to get current UTC time."""
    return datetime.now(timezone.utc)


# =============================================================================
# Users — Authentication & Authorization
# =============================================================================
# CONCEPT: Users table stores credentials and roles for JWT-based auth.
# Roles control what each user can do:
#   admin    → Full access (all agents, all data, approve anything)
#   manager  → Run payroll agents, approve operations, view all employees
#   employee → Query own data only via employee agent
# =============================================================================
class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="employee")  # admin, manager, employee
    full_name = Column(String(255))
    email = Column(String(255), unique=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


# =============================================================================
# Employees — Core HR Data
# =============================================================================
# CONCEPT: This is the domain data that AI agents query and manipulate.
# salary_info and tax_info use JSONB — a PostgreSQL type that stores
# flexible JSON data with indexing support. This is useful for semi-structured
# data that varies between employees (different tax brackets, benefits, etc.)
# =============================================================================
class Employee(Base):
    __tablename__ = "employees"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_code = Column(String(20), unique=True, nullable=False, index=True)  # e.g., "EMP001"
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True)
    department = Column(String(100), nullable=False)
    position = Column(String(100))
    hire_date = Column(DateTime(timezone=True))
    salary_info = Column(JSONB, default=dict)   # {annual_salary, currency, pay_frequency}
    tax_info = Column(JSONB, default=dict)      # {tax_bracket, filing_status, deductions}
    benefits_info = Column(JSONB, default=dict)  # {health_insurance, retirement_pct, pto_days}
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    payroll_items = relationship("PayrollItem", back_populates="employee")


# =============================================================================
# Payroll Runs & Items
# =============================================================================
# CONCEPT: Payroll is processed in "runs" (e.g., January 2025 payroll).
# Each run contains items — one per employee — with their pay breakdown.
# This two-table design allows:
#   1. Batch processing (run entire company payroll at once)
#   2. Audit trail (see every calculation)
#   3. Approval workflow (run must be approved before processing)
# =============================================================================
class PayrollRun(Base):
    __tablename__ = "payroll_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    status = Column(String(50), default="draft")  # draft, calculated, pending_approval, approved, processed
    total_gross = Column(Float, default=0.0)
    total_net = Column(Float, default=0.0)
    total_deductions = Column(Float, default=0.0)
    employee_count = Column(Integer, default=0)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    items = relationship("PayrollItem", back_populates="payroll_run")


class PayrollItem(Base):
    __tablename__ = "payroll_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    payroll_run_id = Column(UUID(as_uuid=True), ForeignKey("payroll_runs.id"), nullable=False)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id"), nullable=False)
    gross_pay = Column(Float, nullable=False)
    tax_amount = Column(Float, default=0.0)
    insurance_amount = Column(Float, default=0.0)
    retirement_amount = Column(Float, default=0.0)
    other_deductions = Column(Float, default=0.0)
    net_pay = Column(Float, nullable=False)
    breakdown = Column(JSONB, default=dict)  # Detailed calculation breakdown
    created_at = Column(DateTime(timezone=True), default=utcnow)

    # Relationships
    payroll_run = relationship("PayrollRun", back_populates="items")
    employee = relationship("Employee", back_populates="payroll_items")


# =============================================================================
# Agent Executions — Audit Trail
# =============================================================================
# CONCEPT: Every time an AI agent runs, we record the full execution.
# This is critical for:
#   1. Debugging — "Why did the agent give this answer?"
#   2. Compliance — "Who ran what, when, and what was the result?"
#   3. Performance — Track latency, token usage, error rates
#   4. Billing — Attribute LLM costs to specific operations
# =============================================================================
class AgentExecution(Base):
    __tablename__ = "agent_executions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(String(255), index=True)  # Conversation thread for multi-turn
    agent_type = Column(String(50), nullable=False)  # router, payroll, employee, compliance
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    user_input = Column(Text, nullable=False)
    agent_output = Column(Text)
    status = Column(String(50), default="running")  # running, completed, failed, pending_approval
    input_data = Column(JSONB, default=dict)   # Full structured input
    output_data = Column(JSONB, default=dict)  # Full structured output
    metadata_ = Column("metadata", JSONB, default=dict)  # Token usage, latency, etc.
    started_at = Column(DateTime(timezone=True), default=utcnow)
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=utcnow)


# =============================================================================
# Approvals — Human-in-the-Loop
# =============================================================================
# CONCEPT: For high-risk operations (e.g., batch payroll > $50k), the agent
# pauses and creates an approval request. A human reviews and decides.
# This is the "human-in-the-loop" pattern — the AI proposes, humans approve.
#
# The workflow:
#   1. Agent detects high-risk operation
#   2. Agent state is checkpointed (saved to DB)
#   3. Approval record created with status="pending"
#   4. Human reviews via API/UI
#   5. On approve: agent resumes from checkpoint
#   6. On reject: agent returns rejection message
# =============================================================================
class Approval(Base):
    __tablename__ = "approvals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    execution_id = Column(UUID(as_uuid=True), ForeignKey("agent_executions.id"), nullable=False)
    approval_type = Column(String(50), nullable=False)  # financial, data_change, compliance
    risk_level = Column(String(20), nullable=False)      # low, medium, high, critical
    payload = Column(JSONB, default=dict)    # What's being approved (amount, operation, etc.)
    status = Column(String(20), default="pending")  # pending, approved, rejected
    requested_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    decided_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    decision_reason = Column(Text)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    decided_at = Column(DateTime(timezone=True))


# =============================================================================
# Tool Audit Log — Every Tool Call Recorded
# =============================================================================
# CONCEPT: Observability requires knowing exactly what the agent did.
# Every tool call (calculate_pay, get_employee, etc.) is logged with:
#   - What tool was called
#   - What inputs were provided
#   - What output was returned
#   - How long it took
#   - Whether any guardrails were triggered
# =============================================================================
class ToolAuditLog(Base):
    __tablename__ = "tool_audit_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    execution_id = Column(UUID(as_uuid=True), ForeignKey("agent_executions.id"), nullable=False)
    tool_name = Column(String(100), nullable=False)
    tool_input = Column(JSONB, default=dict)
    tool_output = Column(JSONB, default=dict)
    duration_ms = Column(Integer)
    status = Column(String(50), default="success")  # success, error, blocked
    guardrail_violations = Column(JSONB, default=list)
    created_at = Column(DateTime(timezone=True), default=utcnow)


# =============================================================================
# Documents — RAG Knowledge Base (Vector Store)
# =============================================================================
# CONCEPT: RAG (Retrieval-Augmented Generation) stores document chunks
# as vector embeddings. When a user asks a question, we:
#   1. Convert the question to a vector (embedding)
#   2. Find the most similar document chunks (cosine similarity)
#   3. Feed those chunks to the LLM as context
#
# The `embedding` column uses pgvector's Vector type (1536 dimensions for
# OpenAI's text-embedding-3-small model). pgvector supports HNSW indexes
# for fast approximate nearest neighbor search.
#
# WHY 1536 dimensions?
#   OpenAI's text-embedding-3-small produces 1536-dimensional vectors.
#   Each dimension captures a semantic feature of the text.
#   Similar texts have vectors that point in similar directions (high cosine similarity).
# =============================================================================
class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)       # The actual text chunk
    embedding = Column(Vector(1536))             # Vector embedding (1536 dims for OpenAI)
    source = Column(String(255))                 # Source file name (e.g., "leave_policy.md")
    section = Column(String(255))                # Section within the document
    metadata_ = Column("metadata", JSONB, default=dict)  # Additional metadata
    created_at = Column(DateTime(timezone=True), default=utcnow)


# Create an HNSW index for fast vector similarity search
# CONCEPT: HNSW (Hierarchical Navigable Small World) is an approximate
# nearest neighbor algorithm. It's not 100% exact but is much faster
# than brute-force search (O(log n) vs O(n)).
# vector_cosine_ops = use cosine similarity for distance metric
Index(
    "idx_documents_embedding_hnsw",
    Document.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 200},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)


# =============================================================================
# Conversation History — Chat Memory
# =============================================================================
# CONCEPT: For multi-turn conversations, the agent needs to remember
# what was said earlier. This table stores each message in a conversation,
# identified by thread_id. The agent loads recent messages as context
# before processing each new request.
# =============================================================================
class ConversationMessage(Base):
    __tablename__ = "conversation_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(String(255), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=utcnow)
