"""
Application Metrics with Prometheus
=============================================================================
CONCEPT: What are Metrics and Why Track Them?

Metrics are numerical measurements collected over time. Unlike logs (which
record individual events) or traces (which follow individual requests),
metrics give you a bird's-eye view of system health:

  - "How many agent executions happened in the last hour?"
  - "What's the 95th percentile LLM API latency?"
  - "How many tokens have we consumed today?"
  - "Which tools are called most frequently?"

Metrics enable:
  1. ALERTING — "Notify me if error rate exceeds 5%"
  2. DASHBOARDS — Real-time Grafana dashboards showing system health
  3. CAPACITY PLANNING — "We're using 1M tokens/day, how much will it cost?"
  4. PERFORMANCE OPTIMIZATION — "Which tool is the bottleneck?"

PROMETHEUS + GRAFANA ARCHITECTURE:
  Your App  ──(exposes /metrics)──>  Prometheus  ──(queries)──>  Grafana
                                     (scrapes &                  (dashboards
                                      stores)                    & alerts)

  1. Your app exposes metrics at a /metrics HTTP endpoint
  2. Prometheus periodically scrapes (pulls) this endpoint
  3. Grafana queries Prometheus to render dashboards and fire alerts

METRIC TYPES (Prometheus):

  1. COUNTER — A value that only goes up (never decreases).
     Examples: total requests, total errors, total tokens consumed.
     Counters are reset to zero when the application restarts.
     You use rate() in Prometheus to compute "requests per second".

     USE: "How many payroll calculations have we done?"
          agent_executions_total{agent_type="payroll", status="success"}

  2. HISTOGRAM — Measures the distribution of values (like latency).
     Automatically creates multiple time series:
       - <name>_bucket{le="X"} — count of observations <= X
       - <name>_count — total number of observations
       - <name>_sum — sum of all observed values
     You can compute averages, percentiles (p50, p95, p99), etc.

     USE: "What's the 95th percentile LLM API latency?"
          histogram_quantile(0.95, rate(llm_latency_seconds_bucket[5m]))

  3. GAUGE — A value that goes up AND down (current state).
     Examples: current memory usage, active connections, queue depth.
     Not used in this module, but mentioned for completeness.

  4. SUMMARY — Similar to Histogram but computes quantiles client-side.
     Generally, Histograms are preferred because they are more flexible
     for server-side aggregation.

WHY THESE SPECIFIC METRICS?
  For an agentic AI system, the key operational concerns are:

  - agent_executions_total: Are agents succeeding or failing? What's
    the success rate by agent type? This is your primary health metric.

  - llm_tokens_counter: LLM API costs are directly proportional to
    token usage. Tracking this lets you estimate costs and set budgets.

  - llm_latency_histogram: LLM API calls are the #1 source of latency
    in agentic systems (often 1-5 seconds per call). Monitoring latency
    distribution helps you detect degradation and optimize.

  - tool_execution_histogram: If a tool (e.g., database query) starts
    taking 10x longer than usual, you want to know immediately.
=============================================================================
"""

from prometheus_client import Counter, Histogram


# =============================================================================
# Counter: Agent Executions
# =============================================================================
# Tracks every agent execution, labeled by agent type and outcome.
#
# LABELS explain:
#   agent_type: "router", "payroll", "employee", "compliance"
#   status: "success", "error", "timeout", "pending_approval"
#
# Example Prometheus queries:
#   Total successful payroll runs:
#     agent_executions_total{agent_type="payroll", status="success"}
#
#   Error rate over last 5 minutes:
#     rate(agent_executions_total{status="error"}[5m])
#       / rate(agent_executions_total[5m])
#
#   Execution rate by agent type:
#     sum by (agent_type) (rate(agent_executions_total[5m]))
# =============================================================================
agent_executions_total = Counter(
    name="agent_executions_total",
    documentation="Total number of agent executions, partitioned by type and status.",
    labelnames=["agent_type", "status"],
)


# =============================================================================
# Counter: LLM Token Usage
# =============================================================================
# Tracks cumulative token consumption, labeled by model.
# This is essential for cost monitoring:
#   - GPT-4o: ~$5 per 1M input tokens, ~$15 per 1M output tokens
#   - GPT-4o-mini: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
#
# Example Prometheus queries:
#   Total tokens consumed by gpt-4o in the last 24 hours:
#     increase(llm_tokens_total{model="gpt-4o"}[24h])
#
#   Token consumption rate (tokens per second):
#     rate(llm_tokens_total[5m])
# =============================================================================
llm_tokens_counter = Counter(
    name="llm_tokens_total",
    documentation="Total LLM tokens consumed, partitioned by model.",
    labelnames=["model"],
)


# =============================================================================
# Histogram: LLM API Latency
# =============================================================================
# Measures the time distribution of LLM API calls.
#
# BUCKET DESIGN:
#   We define custom buckets aligned with typical LLM response times:
#   - 0.1s to 0.5s: Fast responses (simple classifications, short answers)
#   - 0.5s to 2.0s: Typical responses (most agent interactions)
#   - 2.0s to 10.0s: Slow responses (complex reasoning, long outputs)
#   - 10s+: Timeouts or degraded service
#
#   Each bucket counts "how many requests completed in <= X seconds".
#   Prometheus uses these buckets to estimate percentiles.
#
# Example Prometheus queries:
#   95th percentile latency:
#     histogram_quantile(0.95, rate(llm_latency_seconds_bucket[5m]))
#
#   Average latency:
#     rate(llm_latency_seconds_sum[5m]) / rate(llm_latency_seconds_count[5m])
#
#   Requests taking longer than 5 seconds:
#     llm_latency_seconds_bucket{le="5.0"} - compared to _count
# =============================================================================
llm_latency_histogram = Histogram(
    name="llm_latency_seconds",
    documentation="Latency of LLM API calls in seconds.",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 30.0),
)


# =============================================================================
# Histogram: Tool Execution Latency
# =============================================================================
# Measures how long each tool takes to execute.
#
# BUCKET DESIGN:
#   Tools are typically much faster than LLM calls:
#   - 1ms to 50ms: In-memory operations, simple calculations
#   - 50ms to 500ms: Database queries, API calls
#   - 500ms to 5s: Complex operations (batch processing, external APIs)
#
# The `tool_name` label lets you compare tool performance:
#   - calculate_pay: should be <10ms (pure computation)
#   - get_employee: ~5-20ms (database query)
#   - search_documents: ~50-200ms (vector similarity search)
#
# Example Prometheus queries:
#   Average execution time per tool:
#     rate(tool_execution_seconds_sum[5m])
#       / rate(tool_execution_seconds_count[5m])
#
#   Slowest tool (95th percentile):
#     histogram_quantile(0.95, sum by (tool_name, le)
#       (rate(tool_execution_seconds_bucket[5m])))
# =============================================================================
tool_execution_histogram = Histogram(
    name="tool_execution_seconds",
    documentation="Latency of tool executions in seconds, partitioned by tool name.",
    labelnames=["tool_name"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


# =============================================================================
# Helper Functions
# =============================================================================
# These convenience functions wrap the raw metric objects so that callers
# don't need to know the details of prometheus_client's API. They also
# handle unit conversion (milliseconds -> seconds) since application code
# typically measures in milliseconds but Prometheus convention uses seconds.
# =============================================================================


def record_agent_execution(
    agent_type: str,
    status: str,
    duration_ms: float,
    tokens: int = 0,
    model: str = "gpt-4o-mini",
) -> None:
    """
    Record a completed agent execution in metrics.

    This function should be called at the end of every agent run, regardless
    of whether it succeeded or failed. It updates multiple metrics at once:
      1. Increments the execution counter (by agent_type and status)
      2. Records LLM latency in the histogram (converted to seconds)
      3. Adds token consumption to the token counter

    PARAMETERS:
      agent_type: The type of agent that ran.
          Values: "router", "payroll", "employee", "compliance"
      status: The outcome of the execution.
          Values: "success", "error", "timeout", "pending_approval"
      duration_ms: Total execution time in milliseconds.
          We convert to seconds for Prometheus (convention).
      tokens: Total tokens consumed (input + output) during this execution.
          Defaults to 0 if the agent didn't make LLM calls.
      model: The LLM model used (for token cost attribution).
          Defaults to "gpt-4o-mini".

    USAGE:
        import time
        start = time.perf_counter()

        result = await payroll_agent.run(user_input)

        duration_ms = (time.perf_counter() - start) * 1000
        record_agent_execution(
            agent_type="payroll",
            status="success",
            duration_ms=duration_ms,
            tokens=result.token_usage.total,
            model="gpt-4o-mini",
        )

    WHY RECORD ALL THREE METRICS TOGETHER?
      In practice, you always want to update these atomically. If you
      increment the counter but forget the histogram, your latency
      dashboard will be incomplete. Bundling them in one function
      prevents inconsistencies.
    """
    # 1. Increment the execution counter with labels
    agent_executions_total.labels(
        agent_type=agent_type,
        status=status,
    ).inc()

    # 2. Record the LLM call latency (convert ms -> seconds)
    # Prometheus convention is to use seconds as the base unit.
    # This makes PromQL queries consistent (no unit confusion).
    duration_seconds = duration_ms / 1000.0
    llm_latency_histogram.observe(duration_seconds)

    # 3. Track token consumption for cost monitoring
    if tokens > 0:
        llm_tokens_counter.labels(model=model).inc(tokens)


def record_tool_execution(tool_name: str, duration_ms: float) -> None:
    """
    Record a tool execution's latency.

    Call this after every tool invocation to track tool performance over time.
    The histogram data enables you to:
      - Set alerts when a tool becomes slow
      - Compare tool performance across deployments
      - Identify bottleneck tools in the agent pipeline

    PARAMETERS:
      tool_name: The name of the tool that was executed.
          Examples: "calculate_pay", "get_employee", "search_documents",
                    "get_department_summary"
      duration_ms: Execution time in milliseconds.

    USAGE:
        import time
        start = time.perf_counter()

        result = await calculate_pay(employee)

        duration_ms = (time.perf_counter() - start) * 1000
        record_tool_execution("calculate_pay", duration_ms)

    WHAT THE DATA LOOKS LIKE IN PROMETHEUS:
      After calling record_tool_execution("calculate_pay", 8.5):

      tool_execution_seconds_bucket{tool_name="calculate_pay", le="0.01"} = 1
      tool_execution_seconds_bucket{tool_name="calculate_pay", le="0.025"} = 1
      ...
      tool_execution_seconds_count{tool_name="calculate_pay"} = 1
      tool_execution_seconds_sum{tool_name="calculate_pay"} = 0.0085
    """
    duration_seconds = duration_ms / 1000.0
    tool_execution_histogram.labels(tool_name=tool_name).observe(duration_seconds)
