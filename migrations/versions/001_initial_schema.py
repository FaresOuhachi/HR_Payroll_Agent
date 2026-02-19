"""Initial schema — all tables for the HR Payroll Agent system.

CONCEPT: This migration creates all database tables from scratch.
It also enables the pgvector extension, which adds vector data types
and similarity search operators to PostgreSQL.

Revision ID: 001
Create Date: 2025-01-01
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# Revision identifiers
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    # CONCEPT: PostgreSQL extensions add custom data types and functions.
    # pgvector adds the 'vector' type and operators like <-> (cosine distance).
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # --- Users ---
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("username", sa.String(100), unique=True, nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", sa.String(50), nullable=False, server_default="employee"),
        sa.Column("full_name", sa.String(255)),
        sa.Column("email", sa.String(255), unique=True),
        sa.Column("is_active", sa.Boolean, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_users_username", "users", ["username"])

    # --- Employees ---
    op.create_table(
        "employees",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("employee_code", sa.String(20), unique=True, nullable=False),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255), unique=True),
        sa.Column("department", sa.String(100), nullable=False),
        sa.Column("position", sa.String(100)),
        sa.Column("hire_date", sa.DateTime(timezone=True)),
        sa.Column("salary_info", postgresql.JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("tax_info", postgresql.JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("benefits_info", postgresql.JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("is_active", sa.Boolean, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_employees_code", "employees", ["employee_code"])

    # --- Payroll Runs ---
    op.create_table(
        "payroll_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(50), server_default="draft"),
        sa.Column("total_gross", sa.Float, server_default="0"),
        sa.Column("total_net", sa.Float, server_default="0"),
        sa.Column("total_deductions", sa.Float, server_default="0"),
        sa.Column("employee_count", sa.Integer, server_default="0"),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    # --- Payroll Items ---
    op.create_table(
        "payroll_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("payroll_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("payroll_runs.id"), nullable=False),
        sa.Column("employee_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("employees.id"), nullable=False),
        sa.Column("gross_pay", sa.Float, nullable=False),
        sa.Column("tax_amount", sa.Float, server_default="0"),
        sa.Column("insurance_amount", sa.Float, server_default="0"),
        sa.Column("retirement_amount", sa.Float, server_default="0"),
        sa.Column("other_deductions", sa.Float, server_default="0"),
        sa.Column("net_pay", sa.Float, nullable=False),
        sa.Column("breakdown", postgresql.JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    # --- Agent Executions ---
    op.create_table(
        "agent_executions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("thread_id", sa.String(255)),
        sa.Column("agent_type", sa.String(50), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id")),
        sa.Column("user_input", sa.Text, nullable=False),
        sa.Column("agent_output", sa.Text),
        sa.Column("status", sa.String(50), server_default="running"),
        sa.Column("input_data", postgresql.JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("output_data", postgresql.JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("metadata", postgresql.JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_agent_executions_thread_id", "agent_executions", ["thread_id"])

    # --- Approvals ---
    op.create_table(
        "approvals",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("execution_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("agent_executions.id"), nullable=False),
        sa.Column("approval_type", sa.String(50), nullable=False),
        sa.Column("risk_level", sa.String(20), nullable=False),
        sa.Column("payload", postgresql.JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("status", sa.String(20), server_default="pending"),
        sa.Column("requested_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id")),
        sa.Column("decided_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id")),
        sa.Column("decision_reason", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("decided_at", sa.DateTime(timezone=True)),
    )

    # --- Tool Audit Log ---
    op.create_table(
        "tool_audit_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("execution_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("agent_executions.id"), nullable=False),
        sa.Column("tool_name", sa.String(100), nullable=False),
        sa.Column("tool_input", postgresql.JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("tool_output", postgresql.JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("duration_ms", sa.Integer),
        sa.Column("status", sa.String(50), server_default="success"),
        sa.Column("guardrail_violations", postgresql.JSONB, server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    # --- Documents (RAG Vector Store) ---
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(1536)),  # pgvector: 1536 dims for OpenAI embeddings
        sa.Column("source", sa.String(255)),
        sa.Column("section", sa.String(255)),
        sa.Column("metadata", postgresql.JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    # HNSW index for fast vector similarity search
    op.execute("""
        CREATE INDEX idx_documents_embedding_hnsw
        ON documents
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200)
    """)

    # --- Conversation History ---
    op.create_table(
        "conversation_history",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("thread_id", sa.String(255), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("metadata", postgresql.JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_conversation_history_thread_id", "conversation_history", ["thread_id"])


def downgrade() -> None:
    op.drop_table("conversation_history")
    op.drop_table("documents")
    op.drop_table("tool_audit_log")
    op.drop_table("approvals")
    op.drop_table("agent_executions")
    op.drop_table("payroll_items")
    op.drop_table("payroll_runs")
    op.drop_table("employees")
    op.drop_table("users")
    op.execute("DROP EXTENSION IF EXISTS vector")
