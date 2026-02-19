"""
Vector Store — pgvector Operations
=============================================================================
CONCEPT: Vector Databases and Similarity Search

Traditional databases excel at exact matching: "find all employees in Engineering"
uses an exact string comparison. But what about fuzzy, semantic queries like:
  "What is the company policy on taking time off?"

This question doesn't contain the word "PTO" or "leave", yet it should match
our leave policy document. This is where vector search comes in.

HOW VECTOR SIMILARITY SEARCH WORKS:
  1. Each document chunk is stored as a high-dimensional vector (1536 floats)
  2. The user's query is also converted to a vector
  3. We find the document vectors "closest" to the query vector
  4. "Closest" is measured using a distance metric (cosine similarity)

CONCEPT: Cosine Similarity
  Cosine similarity measures the angle between two vectors, ignoring their
  magnitude (length). It outputs a value between -1 and 1:
    - 1.0  = identical direction (semantically identical)
    - 0.0  = perpendicular (no semantic relationship)
    - -1.0 = opposite direction (semantically opposite)

  Formula: cos(theta) = (A . B) / (|A| * |B|)
    where A . B = sum of element-wise products (dot product)
    and |A| = sqrt(sum of squares) (vector magnitude)

  Example (simplified to 3D):
    A = [0.1, 0.8, 0.3]  ("leave policy")
    B = [0.2, 0.7, 0.4]  ("vacation rules")  → cosine ~ 0.98 (very similar!)
    C = [0.9, 0.1, 0.1]  ("tax brackets")    → cosine ~ 0.35 (not similar)

  In pgvector, the <=> operator computes cosine distance (1 - similarity),
  so LOWER values = MORE similar. We ORDER BY distance ASC to get the best
  matches first.

CONCEPT: HNSW Index (Hierarchical Navigable Small World)
  Without an index, finding the nearest vectors requires comparing the query
  against EVERY vector in the table — O(n) time. For 1 million documents,
  that's 1 million comparisons per query.

  HNSW is an approximate nearest neighbor (ANN) algorithm that builds a
  multi-layer graph of vectors:
    - Top layers: coarse navigation (few nodes, big jumps)
    - Bottom layers: fine-grained search (many nodes, small jumps)

  Think of it like searching for a restaurant in a city:
    1. First zoom to the right continent (top layer)
    2. Then the right country (middle layer)
    3. Then the right neighborhood (bottom layer)
    4. Then the exact street (base layer)

  HNSW achieves O(log n) search time with >95% recall (accuracy).
  The trade-off: index build time and memory usage increase.

  Key parameters (configured in our migration):
    - m=16: Each node connects to 16 neighbors (higher = more accurate, more memory)
    - ef_construction=200: Search width during index build (higher = better quality)

CONCEPT: pgvector
  pgvector is a PostgreSQL extension that adds vector operations directly
  to the database. This means we don't need a separate vector database
  (like Pinecone, Weaviate, or Milvus). Benefits:
    1. Simplicity — One database for everything (relational + vector)
    2. ACID transactions — Vector operations are transactional
    3. Familiar SQL — Use standard SQL with vector operators
    4. JOINs — Combine vector search with relational filters
       (e.g., "find similar documents WHERE source = 'leave_policy.md'")
=============================================================================
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.engine import async_session_maker

logger = logging.getLogger(__name__)


async def store_document(
    content: str,
    embedding: list[float],
    source: str,
    section: str = "",
    metadata: Optional[dict[str, Any]] = None,
    session: Optional[AsyncSession] = None,
) -> str:
    """
    Store a document chunk with its embedding in the vector store.

    CONCEPT: Vector Insertion
    Each document chunk gets stored as a row in the documents table with:
      - content: The raw text (for displaying in search results)
      - embedding: The 1536-dim vector (for similarity search)
      - source: Which file/URL this came from (for filtering and citation)
      - section: The heading/section within the document
      - metadata: Any additional info (page number, chunk index, etc.)

    We generate a UUID for each document to ensure uniqueness even if the
    same content is ingested twice. In a production system, you might want
    to deduplicate by content hash.

    Args:
        content:   The text content of this chunk
        embedding: The 1536-dimensional embedding vector
        source:    Source identifier (e.g., "leave_policy.md")
        section:   Section header (e.g., "Section 2: Sick Leave")
        metadata:  Additional metadata as a dict (stored as JSONB)
        session:   Optional database session. If not provided, creates a new one.
                   Passing an existing session is useful for batch operations
                   where you want all inserts in one transaction.

    Returns:
        The UUID of the newly created document record (as a string).
    """
    doc_id = str(uuid.uuid4())
    meta = metadata or {}

    # CONCEPT: Raw SQL with SQLAlchemy text()
    # We use raw SQL here instead of the ORM because pgvector's vector type
    # requires special casting (::vector). While SQLAlchemy's ORM supports
    # pgvector through the pgvector-python package, raw SQL gives us more
    # control over the exact query and makes the vector operations explicit.
    #
    # The :embedding parameter is cast to ::vector to tell PostgreSQL to
    # treat the array of floats as a vector type. Without this cast,
    # PostgreSQL would see it as a regular array.
    insert_query = text("""
        INSERT INTO documents (id, content, embedding, source, section, metadata, created_at)
        VALUES (
            :id,
            :content,
            :embedding::vector,
            :source,
            :section,
            :metadata::jsonb,
            :created_at
        )
    """)

    params = {
        "id": doc_id,
        "content": content,
        "embedding": str(embedding),  # Convert list to string for pgvector casting
        "source": source,
        "section": section,
        "metadata": str(meta).replace("'", '"'),  # Convert to valid JSON string
        "created_at": datetime.now(timezone.utc),
    }

    # CONCEPT: Session Management
    # If a session was provided (e.g., during batch ingestion), use it.
    # Otherwise, create a new session. This pattern allows:
    #   - Single inserts: each gets its own session/transaction
    #   - Batch inserts: share one session for atomicity (all or nothing)
    if session:
        await session.execute(insert_query, params)
        # Don't commit — let the caller decide when to commit
        # (important for batch operations)
    else:
        async with async_session_maker() as new_session:
            await new_session.execute(insert_query, params)
            await new_session.commit()

    logger.info(f"Stored document chunk: id={doc_id}, source={source}, section={section}")
    return doc_id


async def similarity_search(
    query_embedding: list[float],
    k: int = 5,
    source_filter: Optional[str] = None,
    score_threshold: Optional[float] = None,
    session: Optional[AsyncSession] = None,
) -> list[dict[str, Any]]:
    """
    Find the k most similar documents to a query embedding using cosine similarity.

    CONCEPT: k-Nearest Neighbors (kNN) Search
    Given a query vector, we want to find the k document vectors that are
    most similar (closest in cosine distance). The process:
      1. PostgreSQL computes cosine distance between query and every document
      2. HNSW index accelerates this to ~O(log n) instead of O(n)
      3. Results are ordered by distance (ascending = most similar first)
      4. We return the top k results

    CONCEPT: Cosine Distance vs Cosine Similarity
    pgvector's <=> operator returns cosine DISTANCE = 1 - cosine_similarity
      - Distance 0.0 = identical vectors (similarity 1.0)
      - Distance 1.0 = perpendicular vectors (similarity 0.0)
      - Distance 2.0 = opposite vectors (similarity -1.0)
    We convert back to similarity in the results for intuitive interpretation.

    CONCEPT: Source Filtering
    We can optionally filter by source document. This is powerful because it
    combines vector search with relational filtering in a single query:
      "Find chunks similar to 'overtime pay' but ONLY from compensation_policy.md"
    This is a major advantage of pgvector over standalone vector databases.

    Args:
        query_embedding: The 1536-dim vector of the search query
        k:              Number of results to return (default 5)
        source_filter:  Optional — only search within this source document
        score_threshold: Optional — minimum similarity score (0.0 to 1.0)
                        Documents below this threshold are filtered out.
        session:        Optional database session

    Returns:
        List of dicts, each containing:
          - id: Document UUID
          - content: The text chunk
          - source: Source file name
          - section: Section header
          - metadata: Additional metadata
          - similarity_score: Cosine similarity (0.0 to 1.0, higher = better)
    """
    # Build the SQL query dynamically based on filters
    # CONCEPT: Dynamic Query Building
    # We construct the WHERE clause conditionally. This avoids sending
    # unnecessary filter conditions to the database when they're not needed.

    where_clauses = []
    params: dict[str, Any] = {
        "query_embedding": str(query_embedding),
        "k": k,
    }

    if source_filter:
        where_clauses.append("source = :source_filter")
        params["source_filter"] = source_filter

    if score_threshold is not None:
        # Cosine distance < (1 - similarity_threshold) means similarity > threshold
        where_clauses.append("(embedding <=> :query_embedding::vector) < :distance_threshold")
        params["distance_threshold"] = 1.0 - score_threshold

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    # CONCEPT: The Core Similarity Search Query
    # This is where the magic happens. Let's break down the SQL:
    #
    # SELECT ... FROM documents
    #   → Scan the documents table
    #
    # embedding <=> :query_embedding::vector
    #   → Compute cosine distance between each document's embedding and the query
    #   → The <=> operator is pgvector's cosine distance operator
    #   → ::vector casts the parameter string to a vector type
    #
    # ORDER BY distance ASC
    #   → Sort by distance (lowest first = most similar)
    #
    # LIMIT :k
    #   → Return only the top k results
    #
    # 1 - (embedding <=> ...) AS similarity_score
    #   → Convert distance back to similarity for the response
    #
    # When the HNSW index exists, PostgreSQL automatically uses it to
    # accelerate the <=> computation. Without the index, it falls back
    # to a sequential scan (checking every row).

    search_query = text(f"""
        SELECT
            id,
            content,
            source,
            section,
            metadata,
            created_at,
            1 - (embedding <=> :query_embedding::vector) AS similarity_score
        FROM documents
        {where_sql}
        ORDER BY embedding <=> :query_embedding::vector ASC
        LIMIT :k
    """)

    async def _execute(s: AsyncSession) -> list[dict[str, Any]]:
        result = await s.execute(search_query, params)
        rows = result.fetchall()

        # CONCEPT: Result Mapping
        # Convert SQLAlchemy Row objects to plain dicts for easier consumption.
        # Each row contains the columns we selected, accessed by name.
        documents = []
        for row in rows:
            documents.append({
                "id": str(row.id),
                "content": row.content,
                "source": row.source,
                "section": row.section,
                "metadata": row.metadata,
                "similarity_score": round(float(row.similarity_score), 4),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            })

        return documents

    if session:
        results = await _execute(session)
    else:
        async with async_session_maker() as new_session:
            results = await _execute(new_session)

    logger.info(
        f"Similarity search returned {len(results)} results "
        f"(k={k}, source_filter={source_filter})"
    )

    if results:
        logger.debug(
            f"Top result: score={results[0]['similarity_score']:.4f}, "
            f"source={results[0]['source']}, section={results[0]['section']}"
        )

    return results


async def delete_documents_by_source(
    source: str,
    session: Optional[AsyncSession] = None,
) -> int:
    """
    Delete all document chunks from a specific source.

    CONCEPT: Re-ingestion
    When a policy document is updated, we need to:
      1. Delete all old chunks from that source
      2. Re-ingest the updated document
    This function handles step 1. Deleting by source ensures we remove ALL
    chunks from the old version, even if the number of chunks changed.

    Args:
        source: The source identifier to delete (e.g., "leave_policy.md")
        session: Optional database session

    Returns:
        The number of deleted rows
    """
    delete_query = text("""
        DELETE FROM documents WHERE source = :source
    """)

    async def _execute(s: AsyncSession) -> int:
        result = await s.execute(delete_query, {"source": source})
        return result.rowcount

    if session:
        count = await _execute(session)
    else:
        async with async_session_maker() as new_session:
            count = await _execute(new_session)
            await new_session.commit()

    logger.info(f"Deleted {count} document chunks from source: {source}")
    return count


async def count_documents(
    source: Optional[str] = None,
    session: Optional[AsyncSession] = None,
) -> int:
    """
    Count documents in the vector store, optionally filtered by source.

    Useful for monitoring ingestion progress and store health.

    Args:
        source:  Optional source filter
        session: Optional database session

    Returns:
        Number of document chunks matching the filter
    """
    if source:
        query = text("SELECT COUNT(*) FROM documents WHERE source = :source")
        params = {"source": source}
    else:
        query = text("SELECT COUNT(*) FROM documents")
        params = {}

    async def _execute(s: AsyncSession) -> int:
        result = await s.execute(query, params)
        return result.scalar_one()

    if session:
        return await _execute(session)
    else:
        async with async_session_maker() as new_session:
            return await _execute(new_session)
