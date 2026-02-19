"""
Distributed Tracing with OpenTelemetry
=============================================================================
CONCEPT: What is Distributed Tracing?

In a monolithic application, debugging is relatively simple — you read one
log file. But in agentic AI systems, a single user request can trigger a
chain of operations:

    User Request
      -> Router Agent (decides which sub-agent to call)
        -> Payroll Agent (processes the payroll task)
          -> LLM API call (OpenAI / Anthropic)
          -> Database query (get employee data)
          -> Tool execution (calculate_pay)
        -> RAG retrieval (search policy documents)
      -> Response sent back

Without tracing, when something goes wrong, you have no way to follow the
full journey of a request across these components. Distributed tracing
solves this by assigning a unique **trace ID** to each request and tracking
every operation (called a **span**) within that trace.

KEY TERMINOLOGY:
  - Trace: The entire journey of a request through the system. Identified
    by a unique trace_id. Think of it as a tree of operations.

  - Span: A single unit of work within a trace. Each span has:
      * A name (e.g., "payroll_agent.process")
      * A start time and end time (duration)
      * Attributes (key-value metadata like employee_id, token_count)
      * A parent span (forming a tree structure)
      * A status (OK, ERROR)

  - Tracer: The object that creates spans. You get one per module/component.

  - TracerProvider: The global configuration that manages all tracers.
    It decides where spans are sent (console, Jaeger, Datadog, etc.)

  - Exporter: Sends completed spans to a backend for storage and
    visualization. We use ConsoleSpanExporter for development (prints to
    stdout) but in production you would use JaegerExporter, OTLPExporter,
    or a vendor-specific exporter (Datadog, New Relic, etc.)

WHY OPENTELEMETRY?
  OpenTelemetry (OTel) is the industry-standard, vendor-neutral framework
  for observability. It provides a single API for tracing, metrics, and
  logs. By using OTel, you avoid vendor lock-in — you can switch from
  Jaeger to Datadog without changing your application code.

EXAMPLE TRACE for "Calculate payroll for EMP001":
  [Trace: abc-123]
    |-- [Span: router_agent.route] 12ms
    |     |-- [Span: llm.classify_intent] 850ms (tokens: 150)
    |-- [Span: payroll_agent.process] 2340ms
    |     |-- [Span: db.get_employee] 5ms
    |     |-- [Span: tool.calculate_pay] 3ms
    |     |-- [Span: llm.generate_response] 1800ms (tokens: 420)
    |-- [Span: response.format] 2ms

From this trace you can immediately see:
  - The LLM calls dominate latency (850ms + 1800ms = 2650ms out of 2354ms)
  - The database query is fast (5ms)
  - If there's an error, you know exactly which span failed
=============================================================================
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    BatchSpanProcessor,
)
from opentelemetry.sdk.resources import Resource


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
# We track whether tracing has been initialized to avoid setting up the
# provider multiple times (e.g., if setup_tracing() is called from both
# main.py and a test file).
_tracing_initialized: bool = False


def setup_tracing(
    service_name: str = "hr-payroll-agent",
    use_batch_processor: bool = False,
) -> TracerProvider:
    """
    Initialize the OpenTelemetry TracerProvider with a ConsoleSpanExporter.

    WHAT THIS FUNCTION DOES:
      1. Creates a Resource that identifies this service (so spans from
         different microservices can be distinguished in a shared backend).
      2. Creates a TracerProvider — the central configuration for all tracing.
      3. Attaches a ConsoleSpanExporter — prints completed spans to stdout.
         In production, you would replace this with an OTLPSpanExporter
         pointing to your tracing backend (Jaeger, Tempo, Datadog, etc.)
      4. Sets this provider as the global default so any call to
         `trace.get_tracer()` anywhere in the application uses it.

    PARAMETERS:
      service_name: Identifies this service in the tracing backend.
          When you have multiple services (API, worker, scheduler), each
          gets its own name so you can filter traces by service.
      use_batch_processor: If True, use BatchSpanProcessor which buffers
          spans and exports them in batches (better for production).
          If False, use SimpleSpanProcessor which exports immediately
          (better for development — you see spans instantly).

    RETURNS:
      The configured TracerProvider instance.

    USAGE:
        # In your main.py lifespan or startup:
        from src.observability.tracing import setup_tracing
        setup_tracing()

        # Then in any module:
        from src.observability.tracing import get_tracer
        tracer = get_tracer(__name__)

        async def process_payroll(employee_id: str):
            with tracer.start_as_current_span("payroll.process") as span:
                span.set_attribute("employee_id", employee_id)
                # ... do work ...
                span.set_attribute("result", "success")
    """
    global _tracing_initialized

    # Guard against double initialization.
    # In production apps, setup_tracing() is called once during startup.
    # This guard prevents accidental duplicate exporters (which would
    # cause duplicate span output).
    if _tracing_initialized:
        return trace.get_tracer_provider()

    # -----------------------------------------------------------------------
    # Step 1: Define the Resource
    # -----------------------------------------------------------------------
    # A Resource describes the entity producing telemetry. At minimum, it
    # should include `service.name`. You can add more attributes like
    # `service.version`, `deployment.environment`, etc.
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": "development",
        }
    )

    # -----------------------------------------------------------------------
    # Step 2: Create the TracerProvider
    # -----------------------------------------------------------------------
    # The TracerProvider is the factory for Tracers. It holds the
    # configuration (resource, exporters, samplers) and creates Tracer
    # instances on demand.
    provider = TracerProvider(resource=resource)

    # -----------------------------------------------------------------------
    # Step 3: Configure the Exporter and Processor
    # -----------------------------------------------------------------------
    # CONCEPT: Exporter vs Processor
    #   - Exporter: WHERE to send spans (console, Jaeger, OTLP endpoint)
    #   - Processor: HOW to send spans (immediately vs batched)
    #
    # SimpleSpanProcessor: Exports each span immediately when it ends.
    #   Pros: See spans instantly (great for development/debugging).
    #   Cons: Adds latency to every operation (bad for production).
    #
    # BatchSpanProcessor: Buffers spans and exports in batches.
    #   Pros: Minimal performance impact (ideal for production).
    #   Cons: Small delay before spans appear in the backend.
    exporter = ConsoleSpanExporter()

    if use_batch_processor:
        # Production: batch spans for efficiency
        processor = BatchSpanProcessor(exporter)
    else:
        # Development: export immediately for instant feedback
        processor = SimpleSpanProcessor(exporter)

    provider.add_span_processor(processor)

    # -----------------------------------------------------------------------
    # Step 4: Set as the Global TracerProvider
    # -----------------------------------------------------------------------
    # This makes the provider available via `trace.get_tracer()` anywhere
    # in the application without passing the provider around explicitly.
    # This is the "singleton" pattern for observability — there should be
    # exactly one TracerProvider per process.
    trace.set_tracer_provider(provider)
    _tracing_initialized = True

    return provider


def get_tracer(name: str) -> trace.Tracer:
    """
    Get a named Tracer instance for creating spans.

    CONCEPT: Why Named Tracers?
    Each module/component gets its own named tracer. This serves two purposes:
      1. Span attribution — You can tell which module created a span.
      2. Selective instrumentation — You can enable/disable tracing per module.

    The name is typically the module's __name__ (e.g., "src.agents.payroll_agent").

    PARAMETERS:
      name: A descriptive name for the tracer, usually the module path.

    RETURNS:
      A Tracer instance that can create spans.

    USAGE:
        tracer = get_tracer("src.agents.payroll_agent")

        # Create a span for an operation
        with tracer.start_as_current_span("calculate_gross_pay") as span:
            span.set_attribute("employee_id", "EMP001")
            span.set_attribute("period", "2025-01")
            result = compute_gross_pay(...)
            span.set_attribute("gross_pay", result)

        # Spans automatically capture:
        #   - Start time and end time
        #   - Parent-child relationships (nested spans)
        #   - Exception information (if the span's context raises)

    HOW PARENT-CHILD SPANS WORK:
        When you nest `start_as_current_span` calls, OpenTelemetry
        automatically links them:

        with tracer.start_as_current_span("agent.process") as parent:
            # This span becomes the "current" span
            with tracer.start_as_current_span("db.query") as child:
                # child.parent == parent (automatic!)
                ...
            with tracer.start_as_current_span("llm.call") as child2:
                # child2.parent == parent (automatic!)
                ...

        Result:
          agent.process (parent)
            ├── db.query (child)
            └── llm.call (child2)
    """
    return trace.get_tracer(name)
