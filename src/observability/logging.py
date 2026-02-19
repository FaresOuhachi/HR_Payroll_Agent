"""
Structured Logging with structlog
=============================================================================
CONCEPT: Why Structured (JSON) Logging Instead of Plain Text?

Traditional (plain text) log:
    2025-01-15 10:30:45 INFO Processing payroll for EMP001 in department Engineering

Structured (JSON) log:
    {
        "timestamp": "2025-01-15T10:30:45.123Z",
        "level": "info",
        "event": "processing_payroll",
        "employee_id": "EMP001",
        "department": "Engineering",
        "agent_type": "payroll",
        "thread_id": "conv-abc-123",
        "trace_id": "4bf92f3577b34da6"
    }

The JSON log contains the same information but in a MACHINE-PARSEABLE format.
This is critical for production systems because:

  1. SEARCHABILITY — With plain text, searching for "all errors related to
     EMP001" requires fragile regex. With JSON, you query:
       jq 'select(.employee_id == "EMP001" and .level == "error")'
     Or in Elasticsearch/Loki:
       employee_id:"EMP001" AND level:"error"

  2. AGGREGATION — "How many errors per agent type in the last hour?"
     Impossible with plain text. Trivial with structured logs:
       SELECT agent_type, COUNT(*) FROM logs
       WHERE level='error' AND timestamp > NOW() - INTERVAL '1 hour'
       GROUP BY agent_type

  3. CORRELATION — Structured logs include trace_id and thread_id,
     letting you correlate logs with distributed traces and conversation
     history. When a user reports a bug, you find their thread_id, then
     pull ALL related logs, traces, and metrics instantly.

  4. ALERTING — Log aggregation systems (ELK, Loki, Datadog) can parse
     JSON fields and trigger alerts based on specific field values:
       Alert when: level == "error" AND agent_type == "payroll"
                   AND occurrences > 5 in 10 minutes

  5. DEBUGGING AGENTS — AI agents are non-deterministic. The same input
     can produce different outputs. Structured logs capture:
       - What the user asked (event="user_input", content="...")
       - What the LLM decided (event="llm_response", intent="...")
       - What tools were called (event="tool_call", tool="...")
       - What the final response was (event="agent_response", content="...")
     This makes it possible to reconstruct and debug agent behavior.

WHY STRUCTLOG?
  structlog is Python's best structured logging library. It provides:
    - Automatic JSON formatting
    - Context binding (add fields once, they appear in all subsequent logs)
    - Processor pipelines (transform log entries before output)
    - Integration with stdlib logging (works alongside existing loggers)
    - Async-safe (important for FastAPI)

STRUCTLOG PIPELINE:
  When you call logger.info("event", key=value), structlog runs the log
  entry through a chain of "processors":

    Raw event  ->  [add_timestamp]  ->  [add_log_level]  ->  [add_caller_info]
               ->  [JSONRenderer]   ->  Final JSON output

  Each processor enriches the log entry with additional context.
=============================================================================
"""

import logging
import sys

import structlog

from src.config import settings


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_logging_configured: bool = False


def setup_logging() -> None:
    """
    Configure structlog for structured JSON logging.

    This function sets up the entire logging pipeline. It should be called
    once during application startup (in main.py's lifespan function).

    THE PROCESSOR CHAIN:
      Each log entry passes through these processors in order:

      1. structlog.contextvars.merge_contextvars
         Merges any context variables bound via structlog.contextvars.bind().
         This lets you set context once (e.g., request_id, user_id) and
         have it automatically included in all subsequent log entries
         within that async context.

      2. structlog.stdlib.filter_by_level
         Drops log entries below the configured level (e.g., if level is
         INFO, DEBUG messages are silently discarded).

      3. structlog.stdlib.add_logger_name
         Adds the logger's name (e.g., "src.agents.payroll_agent") to
         the log entry. Useful for filtering logs by component.

      4. structlog.stdlib.add_log_level
         Adds the log level as a string (e.g., "info", "error") to
         the log entry.

      5. structlog.stdlib.PositionalArgumentsFormatter
         Formats positional arguments (like printf-style formatting).
         Ensures backward compatibility with stdlib logging patterns.

      6. structlog.processors.TimeStamper(fmt="iso")
         Adds an ISO-8601 timestamp (e.g., "2025-01-15T10:30:45.123456Z").
         ISO format is the standard for machine-parseable timestamps.

      7. structlog.processors.StackInfoRenderer
         If the log entry includes stack_info=True, renders the stack
         trace as a string. Useful for debugging.

      8. structlog.processors.format_exc_info
         If an exception is being logged, formats the traceback and adds
         it to the log entry.

      9. structlog.processors.UnicodeDecoder
         Ensures all byte strings are decoded to Unicode (prevents
         encoding errors in JSON output).

      10. structlog.stdlib.ProcessorFormatter.wrap_for_formatter
          Prepares the log entry for the stdlib formatter (bridges
          structlog's pipeline with Python's logging module).
    """
    global _logging_configured

    if _logging_configured:
        return

    # Determine log level from settings
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # -------------------------------------------------------------------
    # Define the shared processor chain
    # -------------------------------------------------------------------
    # These processors run for EVERY log entry, regardless of which
    # logger emitted it. They enrich the entry with metadata.
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    # -------------------------------------------------------------------
    # Configure structlog
    # -------------------------------------------------------------------
    structlog.configure(
        processors=shared_processors,
        # Use stdlib's LoggerFactory so structlog integrates with Python's
        # built-in logging module. This means:
        #   - structlog logs are handled by stdlib handlers
        #   - Third-party libraries using stdlib logging are also captured
        #   - You get one unified logging pipeline
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # -------------------------------------------------------------------
    # Configure the stdlib root logger
    # -------------------------------------------------------------------
    # This sets up the final output format. We use structlog's
    # ProcessorFormatter to render logs as JSON. The formatter runs
    # AFTER the shared processors and produces the final output string.
    formatter = structlog.stdlib.ProcessorFormatter(
        # The final processor decides the output format:
        #   JSONRenderer() -> {"timestamp": "...", "level": "info", ...}
        #   ConsoleRenderer() -> colored human-readable output
        # We use JSON for production and console for development.
        processor=structlog.dev.ConsoleRenderer()
        if settings.debug
        else structlog.processors.JSONRenderer(),
    )

    # Create a handler that writes to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Reduce noise from verbose third-party libraries
    # These libraries log at DEBUG/INFO level by default, which clutters
    # the output. We set them to WARNING so only important messages appear.
    for noisy_logger in ["uvicorn", "uvicorn.access", "sqlalchemy.engine", "httpcore", "httpx"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _logging_configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Factory function to get a named structured logger.

    CONCEPT: Logger Naming Convention
    Use the module's __name__ as the logger name. This creates a hierarchy:
      - "src" (root package)
      - "src.agents" (agents package)
      - "src.agents.payroll_agent" (specific module)

    This hierarchy lets you:
      - Filter logs by component: logger_name:"src.agents.*"
      - Set different log levels per component (noisy module -> WARNING)

    PARAMETERS:
      name: Logger name, typically __name__ of the calling module.

    RETURNS:
      A structlog BoundLogger instance with the given name.

    USAGE:
        from src.observability.logging import get_logger

        logger = get_logger(__name__)

        # Simple log
        logger.info("payroll_calculated", employee_id="EMP001", net_pay=4500.00)
        # Output: {"timestamp": "...", "level": "info", "logger": "src.agents.payroll",
        #          "event": "payroll_calculated", "employee_id": "EMP001", "net_pay": 4500.0}

        # Log with context binding
        # bind() returns a new logger with extra fields attached.
        # All subsequent logs from this logger include these fields.
        request_logger = logger.bind(
            user_id="user-abc-123",
            thread_id="conv-xyz-789",
            trace_id="4bf92f3577b34da6",
        )
        request_logger.info("agent_started", agent_type="payroll")
        # Output includes user_id, thread_id, AND trace_id automatically

        # Error logging with exception info
        try:
            result = await risky_operation()
        except Exception:
            logger.exception("operation_failed", operation="calculate_tax")
            # Output includes the full stack trace

    ASYNC CONTEXT VARIABLES:
        For request-scoped context in async FastAPI handlers, use
        structlog.contextvars instead of bind():

            import structlog
            structlog.contextvars.clear_contextvars()
            structlog.contextvars.bind_contextvars(
                request_id="req-123",
                user_id="user-abc",
            )
            # Now ALL loggers in this async context include request_id and user_id
            # This works across async function calls without passing the logger around
    """
    return structlog.get_logger(name)
