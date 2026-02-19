"""
SSE Streaming Endpoint — HTTP-Based Alternative to WebSocket
=============================================================================
CONCEPT: Server-Sent Events (SSE)

SSE is a simpler alternative to WebSocket for server→client streaming.
The server sends a stream of events over a single HTTP connection.

WHEN to use SSE vs WebSocket:
  - SSE: One-way streaming (server → client). Simpler. Auto-reconnects.
    Good for: streaming agent responses, live updates, notifications.
  - WebSocket: Two-way (both directions). More complex. Manual reconnect.
    Good for: interactive chat, gaming, collaborative editing.

We support BOTH so the frontend can use the best option available:
  - Primary: WebSocket (full bidirectional chat)
  - Fallback: SSE (if WebSocket is blocked by proxy/firewall)

SSE FORMAT:
  event: reasoning
  data: {"step": "classifying", "data": {"agent": "payroll"}}

  event: message
  data: {"step": "response", "data": {"content": "The net pay is..."}}

  event: done
  data: {"step": "complete", "data": {"duration_ms": 3200}}
=============================================================================
"""

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agents.callbacks import StreamingCallbackHandler
from src.api.websocket import _process_agent_request

router = APIRouter(tags=["Streaming"])


class StreamRequest(BaseModel):
    """Request body for SSE streaming endpoint."""
    input: str = Field(..., description="The user's message")
    thread_id: str | None = Field(None, description="Conversation thread ID")


@router.post("/agents/stream")
async def stream_agent_response(request: StreamRequest):
    """
    Stream agent response as Server-Sent Events.

    CONCEPT: StreamingResponse
    FastAPI's StreamingResponse accepts an async generator.
    It sends each yielded chunk to the client immediately,
    without waiting for the full response.

    The client uses the EventSource API:
        const source = new EventSource('/agents/stream');
        source.addEventListener('reasoning', (e) => { ... });
        source.addEventListener('message', (e) => { ... });

    Note: Since EventSource only supports GET, we use fetch() with
    ReadableStream for POST requests (see frontend/app.js).
    """
    thread_id = request.thread_id or "default"

    callback = StreamingCallbackHandler()

    # Start agent processing in background
    asyncio.create_task(
        _process_agent_request(request.input, thread_id, callback)
    )

    async def event_generator():
        """Yield SSE-formatted events as they arrive from the agent."""
        async for event in callback.events():
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
