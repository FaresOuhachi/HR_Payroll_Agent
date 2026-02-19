"""
Payroll Agent — LangGraph StateGraph for Payroll Operations
=============================================================================
CONCEPT: LangGraph StateGraph

A StateGraph is a directed graph where:
  - NODES are Python functions that process state
  - EDGES connect nodes (can be conditional)
  - STATE flows through the graph, getting modified at each node

The agent follows this flow:
  1. Receive user input
  2. LLM decides which tools to call (tool calling / function calling)
  3. Execute the tools
  4. LLM synthesizes results into a response
  5. If LLM wants more tools → loop back to step 3
  6. If done → return response

CONCEPT: ReAct Pattern (Reasoning + Acting)
The agent alternates between:
  - REASONING: LLM thinks about what to do next
  - ACTING: Execute a tool based on that reasoning
  - OBSERVING: Look at the tool results
  - Repeat until the task is complete

This is implemented as a cycle in the graph with a conditional edge
that checks: "Did the LLM call more tools, or is it done?"
=============================================================================
"""

import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.config import settings
from src.tools.payroll_tools import PAYROLL_TOOLS

# ============================================================================
# CONCEPT: Messages-based State
# LangGraph's prebuilt ToolNode works with a messages-based state.
# Each LLM interaction and tool result is stored as a message.
# This is the standard pattern for conversational agents.
# ============================================================================
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class PayrollAgentState(TypedDict):
    """
    State for the payroll agent.

    CONCEPT: add_messages reducer
    The `Annotated[list, add_messages]` tells LangGraph to APPEND new messages
    to the existing list instead of replacing it. This preserves the full
    conversation history as the agent reasons through multiple steps.
    """
    messages: Annotated[list, add_messages]


# ============================================================================
# LLM Configuration
# ============================================================================
# CONCEPT: ChatGroq with tool binding
# `.bind_tools(PAYROLL_TOOLS)` tells the LLM about our available tools.
# The LLM's response will include tool_calls when it wants to use a tool.
# Groq provides ultra-fast inference for open-source models like Llama 3.1.
# The API is OpenAI-compatible, so LangChain's ChatGroq works seamlessly.
llm = ChatGroq(
    model=settings.groq_model,
    temperature=0,               # 0 = deterministic (no randomness)
    api_key=settings.groq_api_key,
).bind_tools(PAYROLL_TOOLS)


# System prompt that defines the agent's behavior
PAYROLL_SYSTEM_PROMPT = """You are an expert HR Payroll Assistant for Vane LLC, an HR SaaS company.

Your job is to help with payroll-related questions and calculations. You have access to tools
that can look up employee information, calculate pay, and analyze department payroll.

GUIDELINES:
- Always look up employee data using tools before making calculations
- Use employee codes (e.g., EMP001) when calling tools
- Provide clear breakdowns of all calculations
- Include currency in your responses
- If you're unsure about something, say so rather than guessing
- When presenting payroll calculations, format numbers clearly with $ and commas

AVAILABLE INFORMATION:
- Employee details (name, department, salary, tax info, benefits)
- Gross pay calculations (annual salary / 12 for monthly)
- Deduction calculations (tax, health insurance, retirement)
- Net pay calculations (gross - all deductions)
- Leave/PTO balances
- Department-level payroll totals"""


# ============================================================================
# Node Functions
# ============================================================================
async def call_model(state: PayrollAgentState) -> dict:
    """
    CONCEPT: The "brain" of the agent — calls the LLM to decide what to do.

    The LLM receives:
      1. System prompt (defines behavior)
      2. Conversation history (previous messages)
      3. Tool definitions (what tools are available)

    The LLM responds with either:
      a) A text message (final answer) → go to END
      b) Tool calls (needs more info) → go to tools node
    """
    messages = state["messages"]

    # Prepend system prompt if not already there
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=PAYROLL_SYSTEM_PROMPT)] + messages

    response = await llm.ainvoke(
        messages,
        config={"run_name": "payroll_llm_call"},
    )

    return {"messages": [response]}


def should_continue(state: PayrollAgentState) -> str:
    """
    CONCEPT: Conditional Edge — determines the next node.

    After the LLM responds, we check:
    - If it made tool_calls → route to "tools" node (execute the tools)
    - If it didn't → route to END (the response is the final answer)

    This creates the ReAct loop:
    call_model → (has tools?) → tools → call_model → (has tools?) → END
    """
    last_message = state["messages"][-1]

    # If the LLM made tool calls, we need to execute them
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # Otherwise, the LLM is done — it produced a final text response
    return END


# ============================================================================
# Build the Graph
# ============================================================================
def create_payroll_agent() -> StateGraph:
    """
    CONCEPT: Building a LangGraph StateGraph

    The graph has two nodes:
    1. "agent" — Calls the LLM to reason about what to do
    2. "tools" — Executes the tools the LLM requested

    And edges:
    - START → agent (begin with LLM reasoning)
    - agent → tools (if LLM wants to call tools)
    - agent → END (if LLM has the final answer)
    - tools → agent (after tools run, let LLM reason about results)

    This creates a loop:
    START → agent → tools → agent → tools → agent → END

    VISUAL:
        START
          │
          ▼
      ┌──────┐
      │ agent│ ◄─────┐
      └──┬───┘       │
         │           │
    (conditional)    │
      │      │       │
      ▼      ▼       │
    END   ┌──────┐   │
          │tools │ ──┘
          └──────┘
    """
    # Create graph with our state type
    graph = StateGraph(PayrollAgentState)

    # Add nodes
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(PAYROLL_TOOLS))

    # Add edges
    graph.add_edge(START, "agent")           # Start with the agent
    graph.add_conditional_edges(             # Agent decides: tools or done?
        "agent",
        should_continue,
        {
            "tools": "tools",               # Go execute tools
            END: END,                        # We're done
        }
    )
    graph.add_edge("tools", "agent")         # After tools, back to agent

    return graph


# Module-level graph reference — initialized without checkpointer for import-time
# availability, then re-initialized with checkpointer at app startup via init_payroll_graph().
payroll_graph = create_payroll_agent().compile()


def init_payroll_graph(checkpointer):
    """
    Re-compile the payroll graph WITH the PostgreSQL checkpointer.

    Called at app startup after the checkpointer context is opened in lifespan.
    Once initialized, the graph automatically saves/loads conversation
    state per thread_id.
    """
    global payroll_graph
    payroll_graph = create_payroll_agent().compile(checkpointer=checkpointer)
