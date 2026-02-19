"""
WebSocket Endpoint — Real-Time Chat with Agent Reasoning
=============================================================================
CONCEPT: WebSocket Communication

HTTP is request-response: client sends request → server sends response.
WebSocket is bidirectional: both sides can send messages at any time.

WHY WebSocket for chat?
  1. Real-time streaming — Send reasoning steps as they happen
  2. No polling — Client doesn't need to keep asking "are you done?"
  3. Persistent connection — No overhead of reconnecting per message
  4. Bidirectional — User can send new messages while agent is working

PROTOCOL:
  Client → Server: {"type": "message", "content": "Calculate pay for EMP001"}
  Server → Client: {"event_type": "reasoning", "step": "classifying", ...}
  Server → Client: {"event_type": "tool_call", "step": "executing_tool", ...}
  Server → Client: {"event_type": "tool_result", "step": "executing_tool", ...}
  Server → Client: {"event_type": "message", "step": "response", "data": {"content": "..."}}
  Server → Client: {"event_type": "done", "step": "complete", "data": {"duration_ms": ...}}
=============================================================================
"""

import asyncio
import json
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.agents.callbacks import StreamingCallbackHandler
from src.agents.router_agent import classify_intent
import src.agents.payroll_agent as payroll_agent_module
from langchain_core.messages import HumanMessage

router = APIRouter(tags=["WebSocket"])


@router.websocket("/agents/ws/{thread_id}")
async def agent_websocket(websocket: WebSocket, thread_id: str):
    """
    WebSocket endpoint for real-time agent chat.

    CONCEPT: WebSocket Lifecycle
    1. Client connects → await websocket.accept()
    2. Loop: receive messages, process, send responses
    3. Client disconnects → WebSocketDisconnect exception
    4. Clean up resources

    The thread_id in the URL enables multi-turn conversations.
    Same thread_id = same conversation context.
    """
    await websocket.accept()

    try:
        while True:
            # Wait for user message
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)
            user_input = data.get("content", "")

            if not user_input:
                continue

            # Create callback handler for this request
            callback = StreamingCallbackHandler()

            # Process in background so we can stream events
            # CONCEPT: asyncio.create_task runs the agent in a separate coroutine
            # while we stream events from the callback queue to the WebSocket.
            agent_task = asyncio.create_task(
                _process_agent_request(user_input, thread_id, callback)
            )

            # Stream events to client as they arrive
            async for event in callback.events():
                await websocket.send_text(event.to_ws())

            # Wait for agent task to complete (should already be done)
            await agent_task

    except WebSocketDisconnect:
        pass  # Client disconnected — clean exit
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "event_type": "error",
                "step": "error",
                "data": {"message": str(e)},
            }))
        except Exception:
            pass


async def _process_agent_request(
    user_input: str,
    thread_id: str,
    callback: StreamingCallbackHandler,
):
    """
    Process a user request through the agent pipeline, emitting events via callback.

    This function orchestrates the full agent flow:
    1. Classify intent → emit "classifying" event
    2. Route to agent → emit "routing" event
    3. Agent calls tools → emit "tool_call"/"tool_result" events
    4. Agent generates response → emit "message" event
    5. Done → emit "done" event
    """
    start_time = time.time()
    tokens_used = 0

    try:
        # Step 1: Classify intent
        await callback.emit("reasoning", "classifying", {"status": "running"})
        classification = await classify_intent(user_input)
        target_agent = classification["agent"]
        await callback.emit("reasoning", "classifying", {
            "status": "done",
            "agent": target_agent,
            "confidence": classification["confidence"],
        })

        # Step 2: Route to agent
        await callback.emit("reasoning", "routing", {
            "target": target_agent,
        })

        # Step 3: Execute agent
        if target_agent in ("payroll", "employee"):
            await callback.emit("reasoning", "executing_agent", {
                "agent": "payroll",
                "status": "running",
            })

            result = await payroll_agent_module.payroll_graph.ainvoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={
                    "configurable": {"thread_id": thread_id},
                    "metadata": {"session_id": thread_id},
                },
            )

            # Extract tool calls and emit events for each
            tools_used = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tools_used.append(tc["name"])
                        await callback.emit("tool_call", "executing_tool", {
                            "tool": tc["name"],
                            "args": tc.get("args", {}),
                            "status": "done",
                        })

            # Get final response
            final_message = result["messages"][-1]
            response_text = final_message.content

        else:
            # For general/compliance, simple LLM response
            from langchain_groq import ChatGroq
            from langchain_core.messages import SystemMessage
            from src.config import settings

            llm = ChatGroq(model=settings.groq_model, temperature=0, api_key=settings.groq_api_key)
            response = await llm.ainvoke([
                SystemMessage(content="You are an HR Payroll AI Assistant for Vane LLC."),
                HumanMessage(content=user_input),
            ])
            response_text = response.content
            tools_used = []

        # Step 4: Emit response
        await callback.emit("message", "response", {
            "content": response_text,
        })

        # Step 5: Done
        duration_ms = int((time.time() - start_time) * 1000)
        await callback.done({
            "duration_ms": duration_ms,
            "tokens": tokens_used,
            "tools_used": tools_used,
            "agent_type": target_agent,
        })

    except Exception as e:
        await callback.error(str(e))
