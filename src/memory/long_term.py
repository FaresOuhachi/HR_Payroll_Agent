"""
Semantic Long-Term Memory — Fact Storage & Retrieval via Vector Embeddings
=============================================================================
CONCEPT: Semantic Memory vs. Episodic Memory

In cognitive science, human memory is categorized into several types:

  Episodic Memory:
    Remembering specific events and conversations. "What did the user ask
    yesterday at 3pm?" This is what ConversationMemory handles — it stores
    the raw dialogue (who said what, when) and replays it as context.

  Semantic Memory:
    Remembering general knowledge, facts, and concepts — detached from
    the specific episode in which they were learned. "The company's leave
    policy allows 25 PTO days per year." You know this fact, but you might
    not remember exactly when or how you learned it.

This module implements SEMANTIC memory for the HR Payroll agent. Instead
of storing raw conversations, it extracts and stores discrete FACTS that
the agent learns during its interactions. These facts persist indefinitely
and can be recalled later based on meaning (not keywords).

EXAMPLES OF FACTS THE AGENT MIGHT STORE:
  - "Employee EMP007 prefers to be contacted via email, not Slack."
  - "The Finance department processes expense reports on the 15th of each month."
  - "The company switched health insurance providers from Aetna to Cigna in Jan 2025."
  - "Manager approval is required for salary adjustments above $10,000."

WHY STORE FACTS AS EMBEDDINGS?

  Traditional keyword search fails for semantic queries:
    Stored fact: "The company offers 25 days of paid time off per year."
    User query:  "How many vacation days do employees get?"

    A keyword search for "vacation days" would MISS the stored fact because
    the fact uses "paid time off" instead. But semantically, they mean the
    same thing.

  Vector embeddings solve this. When we store a fact, we convert it into
  a 1536-dimensional vector (using OpenAI's text-embedding-3-small model)
  that captures its MEANING. When we search, we convert the query to a
  vector and find the most similar fact vectors using cosine similarity.

    "paid time off" embedding → [0.12, -0.34, 0.56, ...]
    "vacation days"  embedding → [0.11, -0.33, 0.55, ...]
    Cosine similarity: 0.97 (very high!) → Match found!

  This is the same technique used by RAG (Retrieval-Augmented Generation)
  for document search — but here we apply it to agent-learned facts rather
  than pre-loaded documents.

HOW THIS DIFFERS FROM RAG (src/rag/):
  RAG (src/rag/):
    - Stores pre-loaded documents (policies, handbooks, guides)
    - Content is loaded at setup time by an admin
    - Documents are chunked and indexed in bulk
    - Read-only after initial ingestion

  Semantic Memory (this module):
    - Stores facts the agent discovers during conversations
    - Content is added dynamically as the agent operates
    - Individual facts are stored one at a time
    - Read-write throughout the agent's lifetime

  Both use the same underlying infrastructure (embeddings + vector store),
  but they serve different purposes. RAG is the agent's "textbook knowledge,"
  while semantic memory is the agent's "learned experience."

ARCHITECTURE:
  This module uses the RAG infrastructure from Phase 3:
    - src.rag.embeddings: Converts text → vector embeddings via OpenAI API
    - src.rag.vectorstore: Stores and queries vectors in PostgreSQL (pgvector)

  By reusing the RAG infrastructure, we avoid duplicating embedding and
  vector search logic. The only difference is the metadata — we tag
  semantic memory entries with source="semantic_memory" so they can be
  distinguished from RAG document chunks.
=============================================================================
"""

import logging
from datetime import datetime, timezone

from openai import AsyncOpenAI
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.db.models import Document

logger = logging.getLogger(__name__)


class SemanticMemory:
    """
    Long-term semantic memory backed by vector embeddings in PostgreSQL.

    CONCEPT: This class provides a simple key-value-like interface for
    storing and retrieving facts, but under the hood it uses vector
    similarity search to find relevant facts based on meaning.

    The workflow:
      1. Agent learns something important → store_fact("EMP007 prefers email")
      2. Text is converted to a 1536-dim vector via OpenAI embeddings
      3. Vector + text + metadata are stored in the 'documents' table
      4. Later, when the agent needs context → recall("contact preferences")
      5. Query text is converted to a vector
      6. pgvector finds the closest stored fact vectors (cosine similarity)
      7. The matching fact texts are returned to the agent

    WHY REUSE THE DOCUMENTS TABLE?
      The 'documents' table (from src/db/models.py) already has:
        - content (Text): The actual text
        - embedding (Vector(1536)): The vector representation
        - source (String): Where the content came from
        - metadata (JSONB): Flexible metadata

      Instead of creating a separate "facts" table with identical columns,
      we reuse 'documents' and differentiate by source="semantic_memory".
      This gives us the existing HNSW index for free, meaning vector
      searches over semantic memory facts are fast (O(log n)) without
      any additional index creation.

    USAGE:
        memory = SemanticMemory(db_session)

        # Store a fact the agent learned
        await memory.store_fact(
            fact="EMP007 (David Kim) prefers email over Slack for notifications.",
            metadata={"employee_code": "EMP007", "category": "preferences"},
        )

        # Later, recall relevant facts
        facts = await memory.recall("How should I contact David?", k=3)
        # Returns: ["EMP007 (David Kim) prefers email over Slack for notifications."]
    """

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    # The embedding model to use. text-embedding-3-small is OpenAI's most
    # cost-effective embedding model (costs ~$0.02 per 1M tokens) and produces
    # 1536-dimensional vectors. It's sufficient for our use case — we're
    # matching short facts, not nuanced legal documents.
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # The number of dimensions in the embedding vectors. This MUST match the
    # Vector(1536) column definition in the Document model.
    EMBEDDING_DIMENSIONS: int = 1536

    # Source identifier to distinguish semantic memory entries from RAG documents.
    SOURCE_TAG: str = "semantic_memory"

    def __init__(self, db: AsyncSession) -> None:
        """
        Initialize with a database session.

        Args:
            db: An async SQLAlchemy session. The caller manages the session
                lifecycle (typically FastAPI's dependency injection).
        """
        self.db = db
        self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    # =========================================================================
    # Private helpers
    # =========================================================================

    async def _embed(self, text: str) -> list[float]:
        """
        Convert text into a vector embedding using OpenAI's API.

        CONCEPT: Text Embeddings
        An embedding is a fixed-size vector of floating-point numbers that
        represents the "meaning" of a piece of text. Texts with similar
        meanings produce vectors that point in similar directions in
        high-dimensional space.

        Under the hood, OpenAI's embedding model (a trained neural network)
        reads the text and produces a 1536-dimensional vector. Each dimension
        captures some learned semantic feature — though individual dimensions
        aren't interpretable by humans.

        IMPORTANT: The same text always produces the same embedding (the model
        is deterministic). This is why we can store an embedding at write time
        and compare it at read time — the math is consistent.

        Args:
            text: The text to embed (a fact or a query).

        Returns:
            A list of 1536 floats representing the text's meaning.
        """
        response = await self._openai_client.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=text,
        )

        # The API returns a list of embedding objects (one per input text).
        # We only sent one text, so we take the first (and only) result.
        return response.data[0].embedding

    # =========================================================================
    # Public API
    # =========================================================================

    async def store_fact(self, fact: str, metadata: dict | None = None) -> None:
        """
        Store an important fact in long-term semantic memory.

        CONCEPT: Write Path
        When the agent learns something worth remembering, this method:
          1. Converts the fact text into a vector embedding
          2. Creates a Document record with source="semantic_memory"
          3. Stores both the text and the vector in PostgreSQL

        The HNSW index on the embedding column (defined in src/db/models.py)
        automatically indexes the new vector, making it instantly searchable
        via cosine similarity.

        WHAT MAKES A FACT WORTH STORING?
        Not every piece of information should be stored. Good candidates:
          - User preferences ("EMP007 prefers email contact")
          - Business rules discovered during interactions
          - Corrections ("EMP003's department is actually Marketing, not Sales")
          - Temporal facts ("Q4 bonus was 8% in 2024")
          - Frequently asked information (cache common queries)

        The agent's decision of WHAT to store is handled by the agent logic
        (in src/agents/), not by this module. This module simply provides
        the storage mechanism.

        Args:
            fact: The text of the fact to store. Should be a clear, self-contained
                  statement (not a question or partial phrase).
            metadata: Optional dict of structured metadata to store alongside the
                      fact. Useful for filtering and categorization. Examples:
                      - {"employee_code": "EMP007", "category": "preferences"}
                      - {"department": "Finance", "valid_until": "2025-12-31"}
                      - {"source_thread": "thread-abc-123"}
        """
        # Step 1: Generate the embedding vector.
        embedding = await self._embed(fact)

        # Step 2: Merge user metadata with default metadata.
        # We always tag semantic memory entries with standard fields so they
        # can be filtered and audited later.
        full_metadata = {
            "type": "semantic_memory",
            "stored_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }

        # Step 3: Create and persist the Document record.
        #
        # CONCEPT: Reusing the Document Model
        # We store facts in the same 'documents' table used by RAG, but with
        # source="semantic_memory". This lets us:
        #   a) Reuse the existing HNSW vector index
        #   b) Query semantic memory separately from RAG documents
        #   c) Optionally search BOTH at once (union of RAG + memory)
        document = Document(
            content=fact,
            embedding=embedding,
            source=self.SOURCE_TAG,
            section="fact",
            metadata_=full_metadata,
        )
        self.db.add(document)
        await self.db.commit()

        logger.info(
            "Stored semantic memory fact (%d chars, metadata keys: %s)",
            len(fact),
            list(full_metadata.keys()),
        )

    async def recall(self, query: str, k: int = 3) -> list[str]:
        """
        Retrieve the most relevant stored facts for a given query.

        CONCEPT: Read Path — Vector Similarity Search
        When the agent needs to recall stored knowledge, this method:
          1. Converts the query into a vector embedding
          2. Uses pgvector's cosine distance operator (<=> ) to find
             the k closest fact vectors in the database
          3. Returns the fact texts (not the vectors — the agent needs text)

        HOW COSINE SIMILARITY WORKS:
          Cosine similarity measures the angle between two vectors:
            - cos(0°) = 1.0  → identical direction → identical meaning
            - cos(90°) = 0.0 → perpendicular → unrelated
            - cos(180°) = -1.0 → opposite direction → opposite meaning

          pgvector's <=> operator computes cosine DISTANCE (1 - similarity),
          so smaller values mean MORE similar:
            - 0.0 = identical
            - 1.0 = unrelated
            - 2.0 = opposite

          We ORDER BY distance ASC and LIMIT k to get the k most similar facts.

        FILTERING BY SOURCE:
          We only search facts where source='semantic_memory', excluding
          RAG document chunks. This ensures the agent recalls its own
          learned knowledge, not document snippets. If you want to search
          both, you can modify the WHERE clause or make a separate call
          to the RAG system.

        Args:
            query: The search query (natural language). For example:
                   "What are David's contact preferences?"
            k: Number of facts to return. Defaults to 3.
               More facts = more context for the agent, but also more tokens.

        Returns:
            A list of fact strings, ordered by relevance (most relevant first).
            Returns an empty list if no facts are stored or none are relevant.
        """
        # Step 1: Embed the query.
        query_embedding = await self._embed(query)

        # Step 2: Query pgvector for the nearest neighbors.
        #
        # CONCEPT: Raw SQL for Vector Operations
        # SQLAlchemy's ORM doesn't natively support pgvector's <=> operator,
        # so we use a raw SQL expression wrapped in `text()`. This is one of
        # the few places where raw SQL is cleaner than the ORM.
        #
        # The query:
        #   SELECT content
        #   FROM documents
        #   WHERE source = 'semantic_memory'
        #   ORDER BY embedding <=> :query_embedding  -- cosine distance (smaller = closer)
        #   LIMIT :k
        #
        # The <=> operator leverages the HNSW index for fast approximate
        # nearest neighbor search — typically O(log n) instead of O(n).
        result = await self.db.execute(
            select(Document.content)
            .where(Document.source == self.SOURCE_TAG)
            .order_by(
                Document.embedding.cosine_distance(query_embedding)
            )
            .limit(k)
        )

        facts = [row[0] for row in result.fetchall()]

        logger.debug(
            "Recalled %d facts for query: '%s' (requested k=%d)",
            len(facts),
            query[:80],  # Truncate long queries in logs
            k,
        )

        return facts
