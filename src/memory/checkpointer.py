"""
LangGraph PostgreSQL Checkpointer
=============================================================================
CONCEPT: What Is Checkpointing?

Checkpointing is the mechanism by which a LangGraph agent persists its entire
execution state (messages, tool results, intermediate values, node position)
to durable storage after every single node transition. Think of it as an
automatic "save game" system for your AI agent.

Without checkpointing, every piece of agent state lives only in memory.
If the process crashes, the network hiccups, or a user closes their browser,
all progress is lost and the conversation must start from scratch.

WHY IS CHECKPOINTING CRITICAL?
  1. Human-in-the-Loop (HITL) workflows:
     When the agent encounters a high-risk operation (e.g., processing
     a $200,000 payroll batch), it must PAUSE and wait for a human
     to approve or reject the action. This pause might last minutes,
     hours, or even days. The agent's state must survive across that
     entire waiting period. Checkpointing saves the state to PostgreSQL,
     so when the human finally approves, we can reload exactly where
     we left off and resume execution seamlessly.

  2. Resumable / Long-Running Agents:
     Some agent tasks span multiple interactions. For example, a user
     might start a payroll inquiry, leave for lunch, and come back to
     continue. Checkpointing allows the agent to pick up right where
     it stopped — same conversation history, same tool results, same
     reasoning context.

  3. Fault Tolerance:
     If the server restarts, the agent can resume from the last
     checkpoint instead of replaying the entire conversation. This is
     especially important for operations that have side effects (e.g.,
     if a tool already sent an email, we don't want to send it again).

  4. Multi-Turn Conversation Continuity:
     Each invocation of the agent graph continues from the previous
     checkpoint for that thread_id. This means the agent remembers
     the full conversation history and all intermediate state without
     the caller needing to replay anything.

HOW DOES LANGGRAPH SAVE STATE AT EACH NODE TRANSITION?
  LangGraph's execution model works like this:

  1. Before a node runs, the current state is loaded from the checkpointer
     (or initialized from the input if this is the first invocation).

  2. The node function executes and returns a state update (a dict of
     changes to apply to the current state).

  3. LangGraph merges the update into the current state using reducers
     (e.g., `Annotated[list, add_messages]` appends to a list instead
     of replacing it).

  4. The MERGED state is serialized to JSON and written to PostgreSQL
     with a unique checkpoint_id. The previous checkpoint is kept too,
     forming a linked list of states (enabling "time travel" debugging).

  5. The next node in the graph loads this checkpoint and continues.

  The checkpoint record includes:
    - thread_id: Which conversation this belongs to
    - checkpoint_id: Unique ID for this specific state snapshot
    - parent_checkpoint_id: The previous checkpoint (for history)
    - channel_values: The actual serialized state (messages, tool results, etc.)
    - channel_versions: Version numbers for each state key (for conflict detection)

WHAT IS AsyncPostgresSaver?
  AsyncPostgresSaver is LangGraph's built-in checkpointer that stores
  state in PostgreSQL. It requires a psycopg (v3) async connection —
  NOT asyncpg. This is because the LangGraph checkpointing library
  uses psycopg3's native async support and its specific SQL dialect
  for upserting checkpoint records.

  The connection string must use the `postgresql://` scheme (not
  `postgresql+asyncpg://`) because psycopg3 handles its own async
  I/O and does not go through SQLAlchemy's engine layer.

CONNECTION STRING FORMAT:
  SQLAlchemy uses:   postgresql+asyncpg://user:pass@host:port/db
  psycopg3 uses:     postgresql://user:pass@host:port/db

  We convert from the SQLAlchemy format (stored in settings.database_url)
  to the psycopg3 format by stripping the "+asyncpg" driver suffix.
=============================================================================
"""

from src.config import settings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


def _get_psycopg_connection_string() -> str:
    """
    Convert the SQLAlchemy database URL to a psycopg3-compatible format.

    CONCEPT: Driver-Specific Connection Strings
    SQLAlchemy connection strings encode the driver in the scheme:
      - postgresql+asyncpg://...  → uses the asyncpg driver (SQLAlchemy async)
      - postgresql+psycopg2://... → uses psycopg2 (legacy sync)
      - postgresql://...          → uses the default driver

    The langgraph-checkpoint-postgres package uses psycopg v3 directly
    (not through SQLAlchemy), so it expects a plain postgresql:// URL.

    Example:
      Input:  postgresql+asyncpg://hr_admin:secret@localhost:5432/hr_payroll
      Output: postgresql://hr_admin:secret@localhost:5432/hr_payroll
    """
    url = settings.database_url

    # Strip the "+asyncpg" driver suffix so psycopg3 can use this URL directly.
    # We also handle "+psycopg2" just in case someone configures the sync driver.
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    url = url.replace("postgresql+psycopg2://", "postgresql://")
    url = url.replace("postgresql+psycopg://", "postgresql://")

    return url


async def get_checkpointer() -> AsyncPostgresSaver:
    """
    Create and initialize an AsyncPostgresSaver for LangGraph checkpointing.

    CONCEPT: Checkpointer Lifecycle
    The checkpointer needs to:
      1. Establish an async connection to PostgreSQL (via psycopg3)
      2. Create the checkpoint tables if they don't exist (.setup())
      3. Be ready to save/load state for any thread_id

    The .setup() call creates these tables (if they don't exist):
      - checkpoints:        Stores the serialized agent state snapshots
      - checkpoint_blobs:   Stores large binary data referenced by checkpoints
      - checkpoint_writes:  Stores pending writes (for crash recovery)
      - checkpoint_migrations: Tracks schema version for future upgrades

    USAGE WITH LANGGRAPH:
    Once you have a checkpointer, you pass it when compiling the graph:

        checkpointer = await get_checkpointer()
        graph = create_payroll_agent().compile(checkpointer=checkpointer)

        # Now every invocation is automatically checkpointed:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Calculate pay for EMP001")]},
            config={"configurable": {"thread_id": "user-123-session-abc"}}
        )

        # Later, continue the SAME conversation (state is loaded from checkpoint):
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="What about EMP002?")]},
            config={"configurable": {"thread_id": "user-123-session-abc"}}
        )

    HUMAN-IN-THE-LOOP EXAMPLE:
    When the agent pauses for approval, the state is already checkpointed.
    When the human approves, we resume from the exact same state:

        # Agent hits approval node → state is checkpointed automatically
        # ... hours pass ...
        # Human approves → we resume:
        result = await graph.ainvoke(
            None,  # No new input — just resume from checkpoint
            config={"configurable": {"thread_id": "approval-thread-xyz"}}
        )

    Returns:
        AsyncPostgresSaver: A fully initialized checkpointer ready for use
                            with LangGraph graph compilation.
    """
    connection_string = _get_psycopg_connection_string()

    # Create the checkpointer with the psycopg3 connection string.
    #
    # CONCEPT: AsyncPostgresSaver.from_conn_string()
    # This factory method creates a connection pool internally using psycopg3's
    # AsyncConnectionPool. The pool manages multiple async connections so that
    # concurrent agent invocations don't block each other.
    #
    # Under the hood, it roughly does:
    #   pool = AsyncConnectionPool(conninfo=connection_string)
    #   return AsyncPostgresSaver(pool)
    # from_conn_string() returns an async context manager in newer versions
    # of langgraph-checkpoint-postgres. We enter it and keep the reference.
    checkpointer = AsyncPostgresSaver.from_conn_string(connection_string)

    # If from_conn_string returns a context manager, enter it
    if hasattr(checkpointer, "__aenter__"):
        checkpointer = await checkpointer.__aenter__()

    # Create the checkpoint tables in PostgreSQL if they don't already exist.
    #
    # CONCEPT: Idempotent Schema Setup
    # .setup() uses CREATE TABLE IF NOT EXISTS, so it's safe to call multiple
    # times. In production you'd typically call this once at application startup,
    # but it's harmless to call it every time — it's a no-op if tables exist.
    await checkpointer.setup()

    return checkpointer
