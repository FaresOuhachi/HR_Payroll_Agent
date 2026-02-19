"""
Streaming Callback Handler — Real-Time Agent Reasoning to UI
=============================================================================
CONCEPT: Callbacks and Event Streaming

When an AI agent processes a request, it goes through multiple steps:
  1. Classify intent
  2. Retrieve context (RAG)
  3. Plan tool calls
  4. Execute tools
  5. Synthesize response

Without streaming, the user sees NOTHING until step 5 completes.
With streaming, we push events to the UI at each step, so the user
sees the agent's reasoning process in real-time.

HOW IT WORKS:
  - Each agent node calls `callback.emit(event_type, data)`
  - The callback pushes the event to an asyncio.Queue
  - The WebSocket/SSE handler reads from the queue and sends to the browser
  - The browser JavaScript renders events into the reasoning panel

This is similar to how ChatGPT shows "Searching..." or "Analyzing..."
while the model is working.
=============================================================================
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReasoningEvent:
    """
    A single event in the agent's reasoning process.

    CONCEPT: Event-Driven Architecture
    Instead of returning one big response, we emit small events as work happens.
    This enables:
      - Real-time UI updates (user sees progress)
      - Debugging (inspect each step)
      - Audit trail (log every decision)
    """
    event_type: str        # "reasoning", "tool_call", "tool_result", "message", "done", "error"
    step: str              # "classifying", "retrieving_context", "planning", "executing_tool", etc.
    data: dict[str, Any]   # Event-specific data
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """
        Format as Server-Sent Event (SSE).

        CONCEPT: SSE (Server-Sent Events)
        SSE is a standard for streaming data from server to browser.
        Format: "event: <type>\ndata: <json>\n\n"
        The browser's EventSource API parses this automatically.
        """
        payload = {
            "step": self.step,
            "data": self.data,
            "timestamp": self.timestamp,
        }
        return f"event: {self.event_type}\ndata: {json.dumps(payload)}\n\n"

    def to_ws(self) -> str:
        """Format as WebSocket message (JSON)."""
        return json.dumps({
            "event_type": self.event_type,
            "step": self.step,
            "data": self.data,
            "timestamp": self.timestamp,
        })


class StreamingCallbackHandler:
    """
    Collects reasoning events and makes them available for streaming to the UI.

    CONCEPT: asyncio.Queue
    An asyncio Queue is a thread-safe, async-compatible buffer.
    The agent PUTS events into the queue, and the WebSocket handler
    GETS events from it. This decouples the agent from the transport layer.

    Usage:
        callback = StreamingCallbackHandler()

        # In agent nodes:
        await callback.emit("reasoning", "classifying", {"agent": "payroll"})

        # In WebSocket handler:
        async for event in callback.events():
            await websocket.send_text(event.to_ws())
    """

    def __init__(self):
        self.queue: asyncio.Queue[ReasoningEvent | None] = asyncio.Queue()
        self.events_log: list[ReasoningEvent] = []  # Keep full log for debugging
        self._done = False

    async def emit(self, event_type: str, step: str, data: dict[str, Any] | None = None):
        """
        Emit a reasoning event.

        Called by agent nodes to report progress:
            await callback.emit("reasoning", "classifying", {"agent": "payroll", "confidence": 0.95})
            await callback.emit("tool_call", "executing_tool", {"tool": "calculate_net_pay", "args": {...}})
            await callback.emit("tool_result", "executing_tool", {"tool": "calculate_net_pay", "result": {...}})
            await callback.emit("message", "response", {"content": "The net pay is..."})
            await callback.emit("done", "complete", {"duration_ms": 3200, "tokens": 1240})
        """
        event = ReasoningEvent(
            event_type=event_type,
            step=step,
            data=data or {},
        )
        self.events_log.append(event)
        await self.queue.put(event)

    async def done(self, summary: dict[str, Any] | None = None):
        """Signal that the agent is done processing."""
        await self.emit("done", "complete", summary or {})
        self._done = True
        await self.queue.put(None)  # Sentinel value to stop iteration

    async def error(self, error_message: str):
        """Signal an error occurred."""
        await self.emit("error", "error", {"message": error_message})
        self._done = True
        await self.queue.put(None)

    async def events(self):
        """
        Async generator that yields events as they arrive.

        Usage:
            async for event in callback.events():
                if event is None:
                    break
                await websocket.send_text(event.to_ws())
        """
        while True:
            event = await self.queue.get()
            if event is None:
                break
            yield event

    def get_log(self) -> list[dict]:
        """Get the full event log (for debugging/audit)."""
        return [
            {
                "event_type": e.event_type,
                "step": e.step,
                "data": e.data,
                "timestamp": e.timestamp,
            }
            for e in self.events_log
        ]
