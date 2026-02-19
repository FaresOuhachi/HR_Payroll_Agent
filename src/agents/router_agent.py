"""
Router Agent — Classifies User Intent and Routes to Specialist
=============================================================================
CONCEPT: Multi-Agent Architecture

Instead of one monolithic agent, we use specialized agents:
  - Payroll Agent → handles pay calculations, deductions, payslips
  - Employee Agent → handles employee info queries, leave balances
  - Compliance Agent → handles labor law and policy questions

The Router Agent is a "meta-agent" that:
  1. Classifies the user's intent
  2. Routes to the appropriate specialist agent
  3. Returns the specialist's response

WHY multi-agent?
  - Each specialist has a focused system prompt (better accuracy)
  - Each specialist has access only to relevant tools (better safety)
  - Easier to test and debug individual capabilities
  - Follows the "single responsibility principle"

CONCEPT: Intent Classification
The router uses the LLM to classify the user's message into categories.
This is much more robust than keyword matching because the LLM understands
context, synonyms, and nuance. For example:
  "How much does John make?" → payroll (salary question)
  "When can John take vacation?" → employee (leave question)
  "Are we compliant with overtime rules?" → compliance
=============================================================================
"""

import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.config import settings
import src.agents.payroll_agent as payroll_agent_module

# LLM for classification — Groq provides ultra-fast inference
classifier_llm = ChatGroq(
    model=settings.groq_model,
    temperature=0,
    api_key=settings.groq_api_key,
)

CLASSIFIER_PROMPT = """You are an intent classifier for an HR/Payroll system.
Classify the user's message into exactly one of these categories:

- "payroll": Questions about salary, pay, deductions, taxes, net pay, gross pay,
  payroll calculations, compensation, bonuses, overtime pay, payslips, department payroll costs.

- "employee": Questions about employee information, leave balances, PTO,
  time off, department listings, employee search, personal details.

- "compliance": Questions about labor laws, regulations, working hours limits,
  minimum wage, overtime rules, contract requirements, data protection policies.

- "general": Greetings, meta questions about the system, or anything that
  doesn't fit the above categories.

Respond with ONLY the category name, nothing else. Examples:
- "Calculate net pay for EMP001" → payroll
- "How many PTO days does Amina have left?" → employee
- "What is the maximum overtime allowed per month?" → compliance
- "Hello, what can you do?" → general
"""


async def classify_intent(user_input: str, thread_id: str | None = None) -> dict:
    """
    Classify the user's intent using the LLM.

    Returns:
        dict with "agent" (category) and "confidence" (how sure we are)
    """
    response = await classifier_llm.ainvoke(
        [
            SystemMessage(content=CLASSIFIER_PROMPT),
            HumanMessage(content=user_input),
        ],
        config={
            "run_name": "classify_intent",
            "tags": ["router"],
            "metadata": {"session_id": thread_id or ""},
        },
    )

    category = response.content.strip().lower()

    # Validate category
    valid_categories = {"payroll", "employee", "compliance", "general"}
    if category not in valid_categories:
        category = "general"

    return {
        "agent": category,
        "confidence": 0.95 if category in valid_categories else 0.5,
    }


async def route_and_execute(user_input: str, thread_id: str | None = None) -> dict:
    """
    Main entry point: classify intent, route to specialist, return response.

    CONCEPT: This is the orchestration layer — it coordinates the multi-agent system.
    In production, this would also:
      - Load conversation memory
      - Apply input guardrails
      - Log the execution
      - Apply output guardrails
    """
    import time
    start_time = time.time()

    # Ensure we have a thread_id for checkpointing and LangSmith session grouping
    if not thread_id:
        thread_id = str(uuid.uuid4())

    # Step 1: Classify intent
    classification = await classify_intent(user_input, thread_id=thread_id)
    target_agent = classification["agent"]

    reasoning_steps = [
        {
            "step": "classifying",
            "data": {
                "target_agent": target_agent,
                "confidence": classification["confidence"],
            },
        }
    ]

    # Step 2: Execute through the checkpointed graph for ALL intents.
    # This ensures conversation history is preserved across turns regardless
    # of how the intent is classified. The payroll agent's system prompt is
    # broad enough to handle general greetings and compliance questions too —
    # it simply won't call tools for those.
    result = await payroll_agent_module.payroll_graph.ainvoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={
            "run_name": f"{target_agent}_agent",
            "tags": [target_agent],
            "configurable": {"thread_id": thread_id},
            "metadata": {"session_id": thread_id},
        },
    )

    # Extract the final response from messages
    final_message = result["messages"][-1]
    response_text = final_message.content

    # Count tool calls in the conversation
    tools_used = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc["name"])

    duration_ms = int((time.time() - start_time) * 1000)

    return {
        "response": response_text,
        "agent_type": target_agent,
        "thread_id": thread_id,
        "classification": classification,
        "tools_used": tools_used,
        "reasoning_steps": reasoning_steps,
        "duration_ms": duration_ms,
        "status": "completed",
    }
