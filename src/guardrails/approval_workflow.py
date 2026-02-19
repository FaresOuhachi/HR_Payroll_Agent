"""
Human-in-the-Loop Approval Workflow
=============================================================================
CONCEPT: Human-in-the-Loop (HITL)

Human-in-the-loop is a design pattern where an AI system pauses execution
and requests a human decision before proceeding with high-risk actions.
The AI PROPOSES, the human DECIDES.

WHY IS HITL ESSENTIAL FOR AUTONOMOUS AI IN HR/PAYROLL?

  1. Financial Irreversibility
     Payroll operations move real money. Once salaries are deposited into
     bank accounts, reversing them is extremely difficult (requires bank
     cooperation, employee consent, and regulatory approval). A $50,000
     error in a payroll run cannot simply be "undone."

  2. Legal Liability
     HR/payroll decisions have legal consequences: tax withholding errors
     can trigger IRS penalties, incorrect termination processing can lead
     to wrongful termination lawsuits, and benefits miscalculations can
     violate ERISA regulations. A human must sign off on actions that
     carry legal liability.

  3. AI Hallucination Risk
     LLMs can confidently produce incorrect outputs. In a payroll context,
     the agent might calculate a 50% tax rate when it should be 22%, or
     include terminated employees in a payroll run. Human review catches
     these errors before they become real-world problems.

  4. Regulatory Compliance
     Many financial regulations (SOX, SOC 2, GDPR) require human oversight
     of automated financial processes. HITL provides the audit trail that
     auditors and regulators need: "A human reviewed and approved this
     action at this time for this reason."

  5. Trust Building
     When deploying AI in an organization, trust is built incrementally.
     Starting with human-in-the-loop for all operations, then gradually
     reducing oversight as the system proves reliable, is the recommended
     approach. It's much harder to regain trust after an autonomous AI
     makes a costly mistake.

THE APPROVAL WORKFLOW:

  Step 1: Agent detects a high-risk operation
          (via ToolGovernor.classify_risk())
          |
  Step 2: Agent's execution state is checkpointed (saved to database)
          The agent PAUSES — it does NOT proceed with the operation.
          |
  Step 3: An approval record is created with status="pending"
          The record includes: what operation, what amount, who requested it,
          and a snapshot of the data (payload).
          |
  Step 4: A human reviewer (manager or admin) sees the pending approval
          via the /approvals API endpoints or a dashboard UI.
          |
  Step 5: The reviewer approves or rejects with a reason.
          |
  Step 6a: If APPROVED → The agent resumes from its checkpoint and
           executes the operation.
  Step 6b: If REJECTED → The agent returns a rejection message to the
           original user, including the reviewer's reason.

  This workflow is stored in the `approvals` table (see src/db/models.py).

=============================================================================
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.db import repositories as repo

logger = logging.getLogger(__name__)


class ApprovalWorkflow:
    """
    Manages the human-in-the-loop approval workflow for high-risk operations.

    ARCHITECTURE:
    This class is a thin service layer over the approval repository. It adds:
      1. Business logic validation (can't approve an already-decided approval)
      2. Timestamp management (automatically sets decided_at)
      3. Logging for audit trail

    WHY A SEPARATE CLASS?
    The repository (src/db/repositories.py) handles raw database CRUD.
    This class handles BUSINESS LOGIC:
      - "Is this approval still pending?"
      - "Does this decision make sense?"
      - "What timestamp should we record?"

    Separating these concerns means:
      - The repository stays simple and reusable
      - Business rules are testable without database mocking
      - Multiple callers (API, agent, CLI) share the same business logic

    USAGE:
        # In the agent's tool execution node:
        workflow = ApprovalWorkflow(db_session)

        # Request approval for a high-risk operation
        approval = await workflow.request_approval(
            execution_id=current_execution_id,
            approval_type="financial",
            risk_level="high",
            payload={
                "operation": "department_payroll",
                "department": "Engineering",
                "total_amount": 75000.00,
                "employee_count": 12,
            },
            requested_by=user_id,
        )

        # The agent now pauses. Later, a human processes the decision:
        result = await workflow.process_decision(
            approval_id=approval.id,
            decision="approved",
            approver_id=manager_id,
            reason="Reviewed amounts — all within expected ranges.",
        )
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize the approval workflow with a database session.

        CONCEPT: Dependency Injection
        The database session is injected from outside. This means:
          - In API routes: session comes from FastAPI's Depends(get_db_session)
          - In tests: session comes from a test database fixture
          - In agent nodes: session comes from async_session_maker()

        The workflow doesn't care WHERE the session comes from — it just
        uses it to talk to the database.

        Args:
            db: An async SQLAlchemy session for database operations.
        """
        self.db = db

    async def request_approval(
        self,
        execution_id: UUID,
        approval_type: str,
        risk_level: str,
        payload: dict,
        requested_by: UUID | None = None,
    ):
        """
        Create a new approval request for a high-risk operation.

        This is called by the agent when it determines that an operation
        requires human review before proceeding.

        HOW IT WORKS:
          1. Creates a new record in the `approvals` table with status="pending"
          2. The payload contains all information the reviewer needs to make
             a decision (operation type, amounts, affected employees, etc.)
          3. Returns the created Approval ORM object

        WHAT GOES IN THE PAYLOAD?
        The payload is a JSONB field that stores structured data about the
        operation. It should include everything a reviewer needs:
          - operation: What kind of operation (e.g., "department_payroll")
          - amounts: Financial figures involved
          - affected_entities: Which employees, departments, etc.
          - context: Any relevant context (e.g., "Monthly payroll for January")
          - comparison: Comparison to previous periods (if available)

        Args:
            execution_id: The UUID of the agent execution that triggered
                          this approval request. Links back to the
                          agent_executions table for full audit trail.
            approval_type: Category of the approval. Common types:
                           "financial" — Payroll runs, salary changes
                           "data_change" — Employee record modifications
                           "compliance" — Policy exceptions, audit actions
            risk_level: The classified risk level ("low", "medium", "high",
                        "critical"). Determines who can approve:
                          "high" → manager role required
                          "critical" → admin role required
            payload: JSONB dict with operation details for the reviewer.
            requested_by: UUID of the user whose action triggered this
                          approval. None if the system triggered it
                          automatically.

        Returns:
            The created Approval ORM object with id, status="pending", etc.
        """
        logger.info(
            "Approval requested: type=%s, risk=%s, execution=%s, requested_by=%s",
            approval_type,
            risk_level,
            execution_id,
            requested_by,
        )

        approval = await repo.create_approval(
            self.db,
            execution_id=execution_id,
            approval_type=approval_type,
            risk_level=risk_level,
            payload=payload,
            status="pending",
            requested_by=requested_by,
        )

        logger.info(
            "Approval created: id=%s, status=pending",
            approval.id,
        )

        return approval

    async def process_decision(
        self,
        approval_id: UUID,
        decision: str,
        approver_id: UUID,
        reason: str,
    ) -> dict:
        """
        Process a human's approval or rejection decision.

        This is called by the API when a reviewer clicks "Approve" or
        "Reject" in the dashboard.

        HOW IT WORKS:
          1. Fetch the approval record by ID
          2. Validate that it's still pending (can't change a past decision)
          3. Update the record with the decision, approver, reason, and timestamp
          4. Commit the changes to the database
          5. Return a summary dict

        BUSINESS RULES:
          - Only "pending" approvals can be decided (idempotency protection)
          - Decision must be "approved" or "rejected" (no other states)
          - Reason is required (for audit trail — reviewers must explain WHY)
          - decided_at is set automatically to the current UTC time

        WHAT HAPPENS AFTER A DECISION?
          After this method returns, the caller (API endpoint) should:
            - If approved: Resume the agent execution from its checkpoint
            - If rejected: Notify the original user with the rejection reason
          The actual resumption logic lives in the agent execution layer,
          not in this workflow class (separation of concerns).

        Args:
            approval_id: UUID of the approval to decide.
            decision: "approved" or "rejected".
            approver_id: UUID of the user making the decision.
            reason: Human-readable explanation for the decision.
                    Required for audit compliance.

        Returns:
            A dict with the approval details and decision outcome.

        Raises:
            ValueError: If the approval is not found or not in "pending" status,
                        or if the decision is not "approved"/"rejected".
        """
        # ------------------------------------------------------------------
        # Step 1: Fetch the approval record
        # ------------------------------------------------------------------
        approval = await repo.get_approval_by_id(self.db, approval_id)

        if not approval:
            logger.error("Approval not found: id=%s", approval_id)
            raise ValueError(f"Approval {approval_id} not found")

        # ------------------------------------------------------------------
        # Step 2: Validate the current state
        # ------------------------------------------------------------------
        # CONCEPT: State Machine Validation
        # An approval can only transition: pending → approved/rejected
        # It cannot go from approved → rejected or vice versa.
        # This prevents double-processing and ensures auditability.
        # ------------------------------------------------------------------
        if approval.status != "pending":
            logger.warning(
                "Attempted to decide non-pending approval: id=%s, current_status=%s",
                approval_id,
                approval.status,
            )
            raise ValueError(
                f"Approval {approval_id} is already '{approval.status}'. "
                f"Only 'pending' approvals can be decided."
            )

        # ------------------------------------------------------------------
        # Step 3: Validate the decision value
        # ------------------------------------------------------------------
        valid_decisions = {"approved", "rejected"}
        if decision not in valid_decisions:
            raise ValueError(
                f"Invalid decision '{decision}'. Must be one of: {valid_decisions}"
            )

        # ------------------------------------------------------------------
        # Step 4: Update the approval record
        # ------------------------------------------------------------------
        now = datetime.now(timezone.utc)

        approval.status = decision
        approval.decided_by = approver_id
        approval.decision_reason = reason
        approval.decided_at = now

        await self.db.commit()
        await self.db.refresh(approval)

        logger.info(
            "Approval decided: id=%s, decision=%s, decided_by=%s, reason='%s'",
            approval_id,
            decision,
            approver_id,
            reason,
        )

        # ------------------------------------------------------------------
        # Step 5: Return a summary
        # ------------------------------------------------------------------
        return {
            "approval_id": str(approval.id),
            "execution_id": str(approval.execution_id),
            "approval_type": approval.approval_type,
            "risk_level": approval.risk_level,
            "status": approval.status,
            "decided_by": str(approver_id),
            "decision_reason": reason,
            "decided_at": now.isoformat(),
            "payload": approval.payload,
        }

    async def get_pending(self):
        """
        Retrieve all approval requests that are still waiting for a decision.

        This is called by the API to populate the approval dashboard.
        Reviewers see this list and can click into each approval to
        review its details and make a decision.

        ORDERING: Results are ordered by created_at DESC (newest first)
        as defined in the repository's get_pending_approvals function.

        Returns:
            A list of Approval ORM objects with status="pending",
            ordered by creation time (newest first).
        """
        pending = await repo.get_pending_approvals(self.db)

        logger.info("Retrieved %d pending approvals", len(pending))

        return pending
