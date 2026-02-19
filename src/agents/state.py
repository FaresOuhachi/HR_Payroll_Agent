"""
Agent State Definitions
=============================================================================
CONCEPT: LangGraph State

In LangGraph, state is the data that flows through the agent graph.
It's defined as a TypedDict — a Python dictionary with typed keys.

WHY TypedDict?
  - Type safety: IDE autocomplete and error checking
  - Documentation: Clear definition of what data exists at each step
  - Serialization: Easy to save/load from database (checkpointing)

HOW STATE FLOWS:
  1. User input creates initial state
  2. Each node receives state, processes it, and returns updated state
  3. Edges route state to the next node based on conditions
  4. Final node produces the output

CONCEPT: Annotated[list, operator.add]
This tells LangGraph to APPEND to the list instead of replacing it.
Without this, each node would overwrite the previous tool results.
With it, results accumulate as the agent works through steps.
=============================================================================
"""

import operator
from typing import Annotated, Any, Literal, TypedDict


class ToolCall(TypedDict):
    """Represents a single tool call the agent wants to make."""
    tool_name: str
    tool_args: dict[str, Any]


class ToolResult(TypedDict):
    """Represents the result of executing a tool."""
    tool_name: str
    tool_args: dict[str, Any]
    result: Any
    duration_ms: int
    status: str  # "success" or "error"


class ReasoningStep(TypedDict):
    """A single step in the agent's reasoning process (for UI display)."""
    step: str           # "classifying", "retrieving_context", "planning", "executing_tool", etc.
    data: dict[str, Any]
    timestamp: float


class AgentState(TypedDict):
    """
    The main state object that flows through all LangGraph agent graphs.

    CONCEPT: This is the "memory" of a single agent execution.
    Every node in the graph can read and write to this state.
    """
    # --- Input ---
    user_input: str                    # The original user message
    thread_id: str                     # Conversation thread ID
    user_id: str | None                # Authenticated user ID

    # --- Routing ---
    # Which specialist agent should handle this request
    target_agent: str                  # "payroll", "employee", "compliance", "general"
    classification_confidence: float   # How confident the router is (0.0 - 1.0)

    # --- Context ---
    # Retrieved information to help the agent answer
    conversation_history: list[dict]   # Previous messages in this thread
    retrieved_context: list[dict]      # RAG results (policy documents, etc.)

    # --- Tool Use ---
    # Tools the agent plans to call and their results
    planned_tools: list[ToolCall]
    # Annotated with operator.add → results accumulate across nodes
    tool_results: Annotated[list[ToolResult], operator.add]

    # --- Reasoning (for UI streaming) ---
    reasoning_steps: Annotated[list[ReasoningStep], operator.add]

    # --- Safety ---
    requires_approval: bool            # Does this operation need human approval?
    risk_level: str                    # "low", "medium", "high", "critical"
    guardrail_violations: list[str]    # Any guardrail issues detected

    # --- Output ---
    final_response: str                # The agent's final answer to the user
    status: str                        # "running", "completed", "pending_approval", "failed"
    error: str | None                  # Error message if something went wrong

    # --- Metrics ---
    total_tokens: int                  # Total LLM tokens used
    total_duration_ms: int             # Total execution time
