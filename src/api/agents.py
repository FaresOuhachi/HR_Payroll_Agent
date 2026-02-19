"""
Agent API Endpoints
=============================================================================
CONCEPT: Agent Execution API

This is the main interface for interacting with the AI agents.
The client sends a user message, and the API:
  1. Routes it to the appropriate agent
  2. Executes the agent (which may call multiple tools)
  3. Returns the agent's response with metadata

CONCEPT: Request/Response Design
We use Pydantic models to define the exact shape of:
  - What the client sends (AgentRequest)
  - What the server returns (AgentResponse)
This provides automatic validation, documentation, and type safety.
=============================================================================
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agents.router_agent import route_and_execute

router = APIRouter(prefix="/agents", tags=["Agents"])


# =============================================================================
# Request/Response Schemas
# =============================================================================
class AgentRequest(BaseModel):
    """
    What the client sends to execute an agent.

    CONCEPT: Pydantic validation
    - `input` is required (the user's message)
    - `thread_id` is optional (for multi-turn conversations)
    - `agent_type` is optional (skip router and go directly to a specialist)
    """
    input: str = Field(..., description="The user's message or question")
    thread_id: str | None = Field(None, description="Conversation thread ID for multi-turn")
    agent_type: str | None = Field(None, description="Force routing to a specific agent type")


class AgentResponse(BaseModel):
    """What the server returns after agent execution."""
    response: str = Field(..., description="The agent's response text")
    agent_type: str = Field(..., description="Which agent handled the request")
    thread_id: str = Field(..., description="Thread ID for continuing this conversation")
    tools_used: list[str] = Field(default_factory=list, description="Tools called during execution")
    duration_ms: int = Field(..., description="Total execution time in milliseconds")
    status: str = Field(..., description="Execution status")
    classification: dict = Field(default_factory=dict, description="Intent classification details")
    reasoning_steps: list[dict] = Field(default_factory=list, description="Agent reasoning steps")


# =============================================================================
# Endpoints
# =============================================================================
@router.post("/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest):
    """
    Execute an AI agent to process a user request.

    The system will:
    1. Classify the user's intent (payroll, employee, compliance, general)
    2. Route to the appropriate specialist agent
    3. The agent will call tools as needed (database lookups, calculations)
    4. Return the agent's response with full metadata

    CONCEPT: This endpoint is the gateway to the entire agent system.
    All agent interactions flow through here.
    """
    try:
        result = await route_and_execute(
            user_input=request.input,
            thread_id=request.thread_id,
        )

        return AgentResponse(
            response=result["response"],
            agent_type=result["agent_type"],
            thread_id=result["thread_id"],
            tools_used=result["tools_used"],
            duration_ms=result["duration_ms"],
            status=result["status"],
            classification=result["classification"],
            reasoning_steps=result["reasoning_steps"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}"
        )
