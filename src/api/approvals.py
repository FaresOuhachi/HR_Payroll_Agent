"""
Approval API Endpoints — Human-in-the-Loop REST Interface
=============================================================================
CONCEPT: Approval REST API

These endpoints provide the HTTP interface for the human-in-the-loop
approval workflow. They allow human reviewers (managers, admins) to:
  1. View pending approvals — What operations are waiting for a decision?
  2. Inspect approval details — What exactly is being requested?
  3. Approve an operation — "Yes, proceed with this payroll run."
  4. Reject an operation — "No, the amounts look incorrect."

HOW THIS CONNECTS TO THE AGENT SYSTEM:

  Agent                    Database                   Human (via API)
  -----                    --------                   ---------------
  1. Detects high-risk  →  2. Creates approval     →  3. GET /approvals
     operation                (status=pending)           (see pending list)
                                                     4. GET /approvals/{id}
                                                        (review details)
                                                     5. POST /approvals/{id}/approve
                                                        or /reject
                           6. Updates approval      ←  (status=approved/rejected)
  7. Resumes execution  ←     (agent reads decision)

ENDPOINT DESIGN:
  - GET /approvals           → List (collection endpoint)
  - GET /approvals/{id}      → Read one (resource endpoint)
  - POST /approvals/{id}/approve → Action (RPC-style sub-resource)
  - POST /approvals/{id}/reject  → Action (RPC-style sub-resource)

  WHY POST for approve/reject instead of PATCH?
  These are "actions" (verbs), not "updates" (data changes). Using POST
  for actions is a common REST convention. The action sub-resource pattern
  (/approvals/{id}/approve) makes the intent clear and self-documenting.
  Some teams use PATCH with a status field, but the action pattern prevents
  accidental state transitions (you can't PATCH to an invalid status).

=============================================================================
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.engine import get_db_session
from src.guardrails.approval_workflow import ApprovalWorkflow

router = APIRouter(prefix="/approvals", tags=["Approvals"])


# =============================================================================
# Pydantic Schemas — Request and Response Shapes
# =============================================================================
# CONCEPT: API Contracts
#
# These schemas define the exact shape of data flowing in and out of the API.
# They serve multiple purposes:
#   1. Validation — FastAPI rejects malformed requests automatically
#   2. Documentation — Auto-generated Swagger/OpenAPI docs
#   3. Serialization — Control what database fields are exposed
#   4. Type Safety — IDE autocomplete and error detection
# =============================================================================


class ApprovalResponse(BaseModel):
    """
    Response schema for a single approval record.

    This controls which fields from the Approval ORM model are exposed
    in the API. Note that we include the payload (operation details) so
    reviewers can see exactly what they're approving.
    """
    id: UUID
    execution_id: UUID
    approval_type: str
    risk_level: str
    payload: dict
    status: str
    requested_by: UUID | None = None
    decided_by: UUID | None = None
    decision_reason: str | None = None
    created_at: str    # ISO 8601 timestamp string
    decided_at: str | None = None

    class Config:
        from_attributes = True  # Allow creating from SQLAlchemy ORM objects


class ApprovalListResponse(BaseModel):
    """Response schema for listing multiple approvals."""
    approvals: list[ApprovalResponse]
    count: int


class DecisionRequest(BaseModel):
    """
    Request schema for approving or rejecting an approval.

    WHY IS REASON REQUIRED?
    In financial systems, every decision must be justified for audit purposes.
    Regulators (IRS, SOX auditors, etc.) may ask: "Why was this $200,000
    payroll run approved?" The reason field provides that answer.

    approver_id is included in the request body rather than extracted from
    a JWT token. In a production system, you would extract the approver's
    identity from the authentication token instead (see Phase 7: Auth).
    """
    approver_id: UUID = Field(
        ...,
        description="UUID of the user making the decision (will come from JWT in production)"
    )
    reason: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Explanation for why this decision was made (required for audit trail)"
    )


# =============================================================================
# Helper — Convert ORM object to response schema
# =============================================================================
def _approval_to_response(approval) -> ApprovalResponse:
    """
    Convert an Approval ORM object to an ApprovalResponse Pydantic model.

    WHY NOT JUST USE from_attributes=True?
    The Approval model has datetime fields that need to be converted to
    ISO 8601 strings. SQLAlchemy datetime objects are not automatically
    serialized by Pydantic in the way we want (we want ISO strings, not
    Python datetime repr). This helper handles the conversion explicitly.

    Args:
        approval: An Approval SQLAlchemy ORM instance.

    Returns:
        An ApprovalResponse Pydantic model ready for JSON serialization.
    """
    return ApprovalResponse(
        id=approval.id,
        execution_id=approval.execution_id,
        approval_type=approval.approval_type,
        risk_level=approval.risk_level,
        payload=approval.payload or {},
        status=approval.status,
        requested_by=approval.requested_by,
        decided_by=approval.decided_by,
        decision_reason=approval.decision_reason,
        created_at=approval.created_at.isoformat() if approval.created_at else "",
        decided_at=approval.decided_at.isoformat() if approval.decided_at else None,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get("", response_model=ApprovalListResponse)
async def list_pending_approvals(
    db: AsyncSession = Depends(get_db_session),
):
    """
    List all pending approval requests.

    This is the main endpoint for the approval dashboard. Reviewers see
    a list of all operations waiting for their decision, ordered by
    creation time (newest first).

    CONCEPT: Collection Endpoint
    In REST, a GET request to a collection URL (no ID) returns a list.
    We filter to only "pending" approvals because that's what reviewers
    care about — already-decided approvals are historical data.

    Returns:
        ApprovalListResponse with the list of pending approvals and count.
    """
    workflow = ApprovalWorkflow(db)
    pending = await workflow.get_pending()

    return ApprovalListResponse(
        approvals=[_approval_to_response(a) for a in pending],
        count=len(pending),
    )


@router.get("/{approval_id}", response_model=ApprovalResponse)
async def get_approval(
    approval_id: UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get details of a specific approval request.

    Reviewers use this to inspect the full details of an approval before
    making their decision. The payload field contains all the operation
    details (amounts, affected employees, etc.).

    CONCEPT: Resource Endpoint
    In REST, a GET request with an ID returns a single resource.
    We return 404 if the approval doesn't exist — standard REST practice.

    Args:
        approval_id: UUID of the approval to retrieve.

    Returns:
        ApprovalResponse with full approval details.

    Raises:
        HTTPException 404: If the approval doesn't exist.
    """
    from src.db import repositories as repo

    approval = await repo.get_approval_by_id(db, approval_id)
    if not approval:
        raise HTTPException(
            status_code=404,
            detail=f"Approval {approval_id} not found"
        )

    return _approval_to_response(approval)


@router.post("/{approval_id}/approve", response_model=dict)
async def approve_request(
    approval_id: UUID,
    request: DecisionRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Approve a pending approval request.

    When a reviewer approves an operation, this endpoint:
      1. Validates the approval exists and is still pending
      2. Records the decision, approver, and reason
      3. Sets the decided_at timestamp
      4. Returns the updated approval summary

    After approval, the agent system should resume the paused execution
    and carry out the approved operation. That resumption is handled
    by the agent orchestration layer, not by this endpoint.

    CONCEPT: Action Sub-Resource
    /approvals/{id}/approve is an "action" endpoint. Instead of PATCHing
    the status field directly, we expose the action as a dedicated URL.
    This makes the API self-documenting and prevents invalid state
    transitions (you can't accidentally set status to "banana").

    Args:
        approval_id: UUID of the approval to approve.
        request: DecisionRequest with approver_id and reason.

    Returns:
        Dict with the approval details and decision outcome.

    Raises:
        HTTPException 404: If the approval doesn't exist.
        HTTPException 400: If the approval is not in "pending" status.
    """
    workflow = ApprovalWorkflow(db)

    try:
        result = await workflow.process_decision(
            approval_id=approval_id,
            decision="approved",
            approver_id=request.approver_id,
            reason=request.reason,
        )
        return result

    except ValueError as e:
        # ValueError is raised by process_decision for:
        #   - Approval not found
        #   - Approval not in pending status
        error_message = str(e)
        if "not found" in error_message:
            raise HTTPException(status_code=404, detail=error_message)
        raise HTTPException(status_code=400, detail=error_message)


@router.post("/{approval_id}/reject", response_model=dict)
async def reject_request(
    approval_id: UUID,
    request: DecisionRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Reject a pending approval request.

    When a reviewer rejects an operation, this endpoint:
      1. Validates the approval exists and is still pending
      2. Records the rejection decision, approver, and reason
      3. Sets the decided_at timestamp
      4. Returns the updated approval summary

    After rejection, the agent system should notify the original user
    that their operation was denied and include the rejection reason.
    The user can then modify their request and try again.

    WHY SEPARATE APPROVE AND REJECT ENDPOINTS?
    We could have a single POST /approvals/{id}/decide endpoint with
    the decision in the body. But separate endpoints:
      - Make permissions clearer (future: approve requires manager role)
      - Make API usage stats more informative ("/approve called 50 times")
      - Prevent accidental approvals (harder to send wrong action)
      - Are more self-documenting in Swagger/OpenAPI

    Args:
        approval_id: UUID of the approval to reject.
        request: DecisionRequest with approver_id and reason.

    Returns:
        Dict with the approval details and rejection outcome.

    Raises:
        HTTPException 404: If the approval doesn't exist.
        HTTPException 400: If the approval is not in "pending" status.
    """
    workflow = ApprovalWorkflow(db)

    try:
        result = await workflow.process_decision(
            approval_id=approval_id,
            decision="rejected",
            approver_id=request.approver_id,
            reason=request.reason,
        )
        return result

    except ValueError as e:
        error_message = str(e)
        if "not found" in error_message:
            raise HTTPException(status_code=404, detail=error_message)
        raise HTTPException(status_code=400, detail=error_message)
