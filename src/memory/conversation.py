"""
Conversation Memory — Short-Term / Episodic Memory for Multi-Turn Dialogue
=============================================================================
CONCEPT: Why Does Conversation Memory Matter?

Large Language Models are stateless by design. Each API call to GPT-4o (or
any LLM) is independent — the model has zero memory of previous interactions
unless you explicitly provide prior messages in the prompt. This means that
without a memory system, every user message is treated as if it's the very
first message in the conversation.

Consider this exchange WITHOUT memory:
  User: "What is Sarah's salary?"
  Agent: "Sarah Johnson (EMP003) earns $92,000/year."
  User: "What about her tax deductions?"
  Agent: "I don't know who you're referring to."  ← BROKEN! Lost context.

WITH conversation memory:
  User: "What is Sarah's salary?"
  Agent: "Sarah Johnson (EMP003) earns $92,000/year."
  User: "What about her tax deductions?"
  Agent: "Sarah Johnson's monthly tax deduction is $1,533.33."  ← Works!

The second exchange works because we loaded the previous messages and
included them in the prompt, giving the LLM the context it needs.

CONCEPT: Sliding Window vs. Summarization — Two Memory Strategies

Strategy 1: Sliding Window (used by get_history)
  Keep the last N messages (e.g., 10) and discard older ones.
  Pros:
    - Simple to implement
    - Predictable token usage (N messages * avg tokens per message)
    - Fast — just a database query with LIMIT
  Cons:
    - Hard cutoff — if important context was in message #11, it's gone
    - No graceful degradation

Strategy 2: Summarization (used by summarize_history)
  When the conversation exceeds a threshold, use the LLM to compress
  older messages into a concise summary, then keep the summary + recent
  messages.

  Before summarization (20 messages, ~4000 tokens):
    [msg1, msg2, msg3, ..., msg15, msg16, msg17, msg18, msg19, msg20]

  After summarization (~800 tokens):
    [summary_of_msg1_to_msg15, msg16, msg17, msg18, msg19, msg20]

  Pros:
    - Preserves key information from the entire conversation
    - Graceful degradation — important facts survive in the summary
    - Keeps token usage bounded while retaining more context
  Cons:
    - Requires an extra LLM call (adds latency and cost)
    - Summary quality depends on the LLM — some nuance may be lost
    - More complex to implement

This module implements BOTH strategies. The agent uses the sliding window
by default (fast, simple), and triggers summarization when the conversation
grows too long (preserves important context).

CONCEPT: Thread-Based Isolation
Each conversation has a unique thread_id. This ensures:
  - Different users don't see each other's conversations
  - The same user can have multiple independent conversations
  - State is cleanly separated per interaction

ARCHITECTURE NOTE:
This class does NOT directly execute SQL queries. Instead, it delegates to
the repository layer (src.db.repositories), which owns all database access.
This follows the Repository Pattern — the memory module is a higher-level
abstraction that composes repository calls with LLM-based summarization.
=============================================================================
"""

import logging

from groq import AsyncGroq
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.db.repositories import get_conversation_history, add_conversation_message

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages short-term conversation memory for multi-turn agent interactions.

    CONCEPT: This class bridges the gap between the database (where messages
    are stored) and the agent (which needs formatted message lists).

    The database stores raw messages with metadata (timestamps, thread IDs).
    The agent needs a clean list of {"role": "user", "content": "..."} dicts
    that can be fed directly into the LLM prompt.

    This class handles:
      1. Fetching recent messages in the right format (sliding window)
      2. Storing new messages as the conversation progresses
      3. Summarizing old messages when the conversation grows too long

    USAGE:
        memory = ConversationMemory(db_session)

        # Load recent context for the agent
        history = await memory.get_history("thread-abc-123", limit=10)
        # Returns: [{"role": "user", "content": "..."}, {"role": "assistant", ...}]

        # After the agent responds, store the exchange
        await memory.add_message("thread-abc-123", "user", "What is EMP001's salary?")
        await memory.add_message("thread-abc-123", "assistant", "EMP001 earns $85,000/year.")

        # Periodically compress old messages
        await memory.summarize_history("thread-abc-123")
    """

    # -------------------------------------------------------------------------
    # Configuration constants
    # -------------------------------------------------------------------------

    # When the conversation exceeds this many messages, we trigger summarization.
    # This keeps token usage bounded while preserving important context.
    #
    # WHY 20? A rough heuristic:
    #   - Average message is ~100 tokens
    #   - 20 messages = ~2000 tokens of context
    #   - GPT-4o has 128k context, but we want to leave room for:
    #     - System prompt (~500 tokens)
    #     - RAG context (~2000 tokens)
    #     - Tool definitions (~1000 tokens)
    #     - Agent reasoning + response (~2000 tokens)
    #   - 2000 tokens for history is a reasonable budget
    SUMMARIZATION_THRESHOLD: int = 20

    # When summarizing, keep the last N messages verbatim (don't summarize them).
    # These recent messages are the most relevant for the current exchange.
    KEEP_RECENT_COUNT: int = 5

    def __init__(self, db: AsyncSession) -> None:
        """
        Initialize with a database session.

        CONCEPT: Dependency Injection
        We receive the database session from the caller (typically FastAPI's
        dependency injection system). This means:
          - The memory class doesn't create its own connections
          - The caller controls the session lifecycle (commit, rollback, close)
          - Testing is easy: inject a mock session

        Args:
            db: An async SQLAlchemy session for database operations.
        """
        self.db = db

        # Initialize the Groq client for summarization.
        # We only create this when needed (summarize_history), but having it
        # ready avoids repeated instantiation.
        self._groq_client = AsyncGroq(api_key=settings.groq_api_key)

    # =========================================================================
    # Public API
    # =========================================================================

    async def get_history(self, thread_id: str, limit: int = 10) -> list[dict]:
        """
        Retrieve recent conversation messages for a thread.

        CONCEPT: Sliding Window Memory
        This implements the sliding window strategy — we fetch the last `limit`
        messages from the database and return them as simple dicts that can be
        directly used in an LLM prompt.

        The messages are returned in chronological order (oldest first), which
        is what LLMs expect — the conversation should read naturally from top
        to bottom.

        HOW IT WORKS:
          1. Query the database for the last N messages (ORDER BY created_at DESC LIMIT N)
          2. Reverse the result to get chronological order
          3. Convert ORM objects to simple dicts (role + content)

        The repository function (get_conversation_history) handles steps 1 and 2.
        This method handles step 3 — the format conversion.

        Args:
            thread_id: Unique identifier for the conversation thread. This is
                       typically generated when a user starts a new conversation
                       (e.g., "user-42-session-abc123").
            limit: Maximum number of messages to retrieve. Defaults to 10.
                   Higher values provide more context but consume more tokens.

        Returns:
            A list of message dicts in chronological order, e.g.:
            [
                {"role": "system", "content": "Summary of earlier conversation..."},
                {"role": "user", "content": "What is EMP001's salary?"},
                {"role": "assistant", "content": "EMP001 earns $85,000/year."},
                {"role": "user", "content": "What about their tax deductions?"},
            ]
        """
        # Fetch from the repository layer (which handles the SQL query).
        # The repository returns ConversationMessage ORM objects, ordered
        # chronologically (it queries DESC and then reverses).
        messages = await get_conversation_history(self.db, thread_id, limit)

        # Convert ORM objects to plain dicts for the LLM.
        # The LLM only needs "role" and "content" — it doesn't need IDs,
        # timestamps, or metadata. Keeping the format minimal reduces noise
        # and token usage.
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """
        Store a new message in the conversation history.

        CONCEPT: Append-Only Message Log
        Conversation history is append-only — we never modify or delete
        individual messages (except during summarization, where old messages
        are replaced by a summary). This design:
          1. Preserves a complete audit trail
          2. Avoids race conditions (no updates, only inserts)
          3. Makes debugging easy (replay the full conversation)

        This method is called twice per agent interaction:
          1. Once for the user's message (role="user")
          2. Once for the agent's response (role="assistant")

        Sometimes a third call stores system messages (role="system"),
        such as summaries or injected context.

        Args:
            thread_id: Conversation thread identifier.
            role: Message author — one of "user", "assistant", or "system".
                  - "user": Human input
                  - "assistant": Agent/LLM response
                  - "system": System-generated context (summaries, instructions)
            content: The message text.
            metadata: Optional dict of extra information to store with the
                      message (e.g., token count, model used, latency).
        """
        await add_conversation_message(
            self.db,
            thread_id=thread_id,
            role=role,
            content=content,
            metadata=metadata,
        )

        logger.debug(
            "Stored %s message for thread %s (%d chars)",
            role,
            thread_id,
            len(content),
        )

    async def summarize_history(self, thread_id: str) -> None:
        """
        Summarize older messages when the conversation grows too long.

        CONCEPT: LLM-Based Conversation Compression
        When a conversation exceeds SUMMARIZATION_THRESHOLD messages, we:
          1. Fetch ALL messages for the thread
          2. Split into "old" (to summarize) and "recent" (to keep verbatim)
          3. Send the old messages to the LLM with a summarization prompt
          4. Replace the old messages with a single "system" summary message
          5. Keep the recent messages intact

        WHY NOT JUST USE A BIGGER SLIDING WINDOW?
          - Token costs scale linearly with context length
          - Longer contexts slow down inference (attention is O(n^2) in length)
          - Most old messages are not relevant to the current question
          - A good summary captures the key facts in 10-20% of the tokens

        EXAMPLE:
          Before (22 messages, ~4400 tokens of context):
            user: "How many employees in Engineering?"
            assistant: "There are 5 employees in Engineering."
            user: "List them."
            assistant: "1. John Doe (EMP001)..."
            ... (18 more messages about various topics) ...
            user: "What about Finance department?"
            assistant: "Finance has 3 employees."

          After summarization (6 messages, ~1200 tokens):
            system: "Summary: The user asked about Engineering (5 employees:
                     John Doe EMP001, ...). They also inquired about ..."
            user: (recent message 1)
            assistant: (recent message 2)
            user: (recent message 3)
            assistant: (recent message 4)
            user: (recent message 5)

        WHEN TO CALL THIS:
          - After every agent interaction, check message count
          - If count > SUMMARIZATION_THRESHOLD, call this method
          - The agent orchestrator (in src/agents/) typically handles this

        Args:
            thread_id: Conversation thread identifier.
        """
        # Step 1: Fetch the FULL conversation history (no limit).
        # We need all messages to produce a comprehensive summary.
        all_messages = await get_conversation_history(
            self.db, thread_id, limit=1000  # Practical upper bound
        )

        # Step 2: Check if summarization is actually needed.
        if len(all_messages) <= self.SUMMARIZATION_THRESHOLD:
            logger.debug(
                "Thread %s has %d messages (threshold: %d) — skipping summarization",
                thread_id,
                len(all_messages),
                self.SUMMARIZATION_THRESHOLD,
            )
            return

        # Step 3: Split into old messages (to summarize) and recent messages (to keep).
        #
        # CONCEPT: The split point
        # We keep the last KEEP_RECENT_COUNT messages verbatim because they
        # contain the most immediately relevant context. Everything before
        # that gets compressed into a summary.
        split_point = len(all_messages) - self.KEEP_RECENT_COUNT
        old_messages = all_messages[:split_point]
        # recent_messages are kept as-is; we don't need to do anything with them.

        logger.info(
            "Summarizing %d old messages for thread %s (keeping %d recent)",
            len(old_messages),
            thread_id,
            self.KEEP_RECENT_COUNT,
        )

        # Step 4: Format old messages for the summarization prompt.
        formatted_history = "\n".join(
            f"{msg.role}: {msg.content}" for msg in old_messages
        )

        # Step 5: Call the LLM to produce a summary.
        #
        # CONCEPT: Summarization Prompt Engineering
        # The prompt is carefully designed to:
        #   a) Extract key facts (names, numbers, decisions)
        #   b) Preserve entities (employee codes, department names)
        #   c) Note any unresolved questions or ongoing topics
        #   d) Be concise — the whole point is to reduce tokens
        #
        # We use a lower temperature (0.2) for summarization because we want
        # a factual, consistent summary — not creative writing.
        try:
            response = await self._groq_client.chat.completions.create(
                model=settings.groq_model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a conversation summarizer for an HR Payroll AI assistant. "
                            "Produce a concise summary of the conversation below. "
                            "Focus on:\n"
                            "- Key facts mentioned (employee names, codes, salaries, departments)\n"
                            "- Decisions made or actions taken\n"
                            "- Any pending questions or unresolved topics\n"
                            "- Important context the assistant would need to continue helping\n\n"
                            "Keep the summary under 300 words. Use bullet points for clarity."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Summarize this conversation history:\n\n{formatted_history}"
                        ),
                    },
                ],
            )

            summary = response.choices[0].message.content

        except Exception as exc:
            # If summarization fails (e.g., OpenAI API error), log the error
            # but DON'T crash. The conversation can continue with the full
            # history — it will just use more tokens than ideal.
            logger.error(
                "Failed to summarize thread %s: %s",
                thread_id,
                exc,
                exc_info=True,
            )
            return

        # Step 6: Store the summary as a system message.
        #
        # CONCEPT: Summary as System Message
        # We store the summary with role="system" so the LLM treats it as
        # authoritative background context (not as something a user said or
        # the assistant previously responded). This is important because:
        #   - System messages have the highest influence on LLM behavior
        #   - The LLM won't try to "respond" to a system message
        #   - It clearly delineates "this is context" vs "this is dialogue"
        await add_conversation_message(
            self.db,
            thread_id=thread_id,
            role="system",
            content=f"[Conversation Summary]\n{summary}",
            metadata={"type": "summary", "summarized_message_count": len(old_messages)},
        )

        # Step 7: Delete the old messages that were summarized.
        #
        # CONCEPT: Cleanup After Summarization
        # We delete the old messages to keep the database clean and ensure
        # that the next get_history() call returns the summary + recent
        # messages (not the full history plus the summary).
        #
        # We do this by directly deleting the ORM objects from the session.
        # Since these objects are already loaded and attached to the session,
        # SQLAlchemy can issue DELETE statements for each one.
        for msg in old_messages:
            await self.db.delete(msg)
        await self.db.commit()

        logger.info(
            "Summarized %d messages into summary for thread %s",
            len(old_messages),
            thread_id,
        )
