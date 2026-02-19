"""
Hybrid Retriever — Combining Vector and Keyword Search
=============================================================================
CONCEPT: What is Hybrid Retrieval?

In a RAG system, retrieval is the step where we find relevant documents to
feed to the LLM. There are two main approaches:

1. VECTOR SEARCH (Semantic Search)
   - Converts the query to a vector, finds closest document vectors
   - Strength: Understands meaning ("time off" matches "PTO", "vacation")
   - Weakness: May miss exact terms ("EMP001", "Section 3.2", "$2,500")
   - Example: "What happens if I'm sick?" → finds "Sick Leave Allocation"

2. KEYWORD SEARCH (Lexical Search)
   - Searches for exact word matches using text patterns (ILIKE, full-text search)
   - Strength: Great for specific terms, codes, numbers, exact phrases
   - Weakness: Misses synonyms ("car" won't find "automobile")
   - Example: "Section 2.1" → finds the exact section (vector search might not)

CONCEPT: Why Combine Both?

Each approach has blind spots. Hybrid retrieval merges results from both methods
to get the best of both worlds:

  Query: "What is the minimum wage in Section 2.1?"

  Vector search finds: chunks about minimum wage, salary floors, compensation
  Keyword search finds: chunks containing "Section 2.1", "$2,500/month"

  Merged result: The chunk from "Section 2.1: Minimum Wage Requirements" appears
  at the top because it scores well in BOTH searches.

CONCEPT: Reciprocal Rank Fusion (RRF)

When combining results from two different search methods, how do we rank them?
We can't directly compare vector similarity scores with keyword match scores
(they're on different scales). RRF solves this elegantly:

  For each result, compute: score = 1 / (rank + k)

  where rank = position in the original result list, and k = a constant (usually 60)

  Then sum the RRF scores across all search methods. Documents that appear
  near the top of MULTIPLE lists get the highest combined scores.

  Example:
    Doc A: rank 1 in vector, rank 3 in keyword → 1/61 + 1/63 = 0.0164 + 0.0159 = 0.0323
    Doc B: rank 2 in vector, not in keyword    → 1/62 + 0     = 0.0161
    Doc C: rank 5 in vector, rank 1 in keyword → 1/65 + 1/61  = 0.0154 + 0.0164 = 0.0318

  Final ranking: A (0.0323) > C (0.0318) > B (0.0161)
  Doc A wins because it ranked well in BOTH searches!

CONCEPT: When to Use Each Strategy
  - Vector-only: Good for natural language questions ("What is the overtime policy?")
  - Keyword-only: Good for exact lookups ("EMP001", "Section 4.2")
  - Hybrid: Best for general-purpose search (recommended default)
=============================================================================
"""

import logging
import re
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.rag.embeddings import embedding_service
from src.rag.vectorstore import similarity_search
from src.db.engine import async_session_maker

logger = logging.getLogger(__name__)

# =============================================================================
# RRF constant (k parameter)
# =============================================================================
# CONCEPT: RRF Constant
# The k parameter controls how much we penalize lower-ranked results.
# Higher k = more equal weighting (ranks 1 and 10 score similarly)
# Lower k = more emphasis on top results (rank 1 dominates)
# 60 is the standard value from the original RRF paper (Cormack et al., 2009)
RRF_K = 60


class HybridRetriever:
    """
    A retriever that combines vector similarity search with keyword-based search
    to provide robust document retrieval for RAG.

    CONCEPT: Retriever as an Abstraction
    The retriever hides the complexity of multi-method search behind a simple
    interface: give me a query, get back relevant documents. The calling code
    (agent, API endpoint) doesn't need to know about embeddings, cosine
    similarity, or keyword matching — it just calls retrieve().

    This abstraction also makes it easy to:
      - Switch retrieval strategies without changing calling code
      - Add new search methods (e.g., full-text search, metadata filters)
      - A/B test different retrieval approaches
      - Log and monitor retrieval quality

    Usage:
        retriever = HybridRetriever()
        results = await retriever.retrieve("What is the sick leave policy?")
        for doc in results:
            print(f"{doc['source']} ({doc['score']:.3f}): {doc['content'][:100]}")
    """

    def __init__(
        self,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            vector_weight:  Weight for vector search results in RRF fusion.
                           Higher values favor semantic understanding.
            keyword_weight: Weight for keyword search results in RRF fusion.
                           Higher values favor exact term matching.

        CONCEPT: Configurable Weights
        The weight ratio controls the "personality" of the search:
          - 0.9/0.1 → Almost pure semantic search (good for general questions)
          - 0.5/0.5 → Equal balance (good for mixed queries)
          - 0.3/0.7 → Keyword-heavy (good for reference lookups)
        Default 0.7/0.3 favors semantic search, which works well for HR queries
        that tend to be natural language questions.
        """
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        source_filter: Optional[str] = None,
        strategy: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant document chunks for a given query.

        CONCEPT: Retrieval Pipeline
        This method orchestrates the full retrieval process:
          1. Analyze the query to choose the best strategy
          2. Run vector search (embed query → cosine similarity)
          3. Run keyword search (ILIKE pattern matching)
          4. Fuse results using Reciprocal Rank Fusion
          5. Return the top k results with combined scores

        Args:
            query:         The user's search query (natural language)
            k:             Number of results to return (default 5)
            source_filter: Optional — restrict search to a specific source
            strategy:      Search strategy to use:
                          - "hybrid" (default): combine vector + keyword
                          - "vector": vector search only (pure semantic)
                          - "keyword": keyword search only (pure lexical)

        Returns:
            List of result dicts, sorted by relevance score (descending):
              - content: The text chunk
              - source: Source document name
              - section: Section header
              - metadata: Additional metadata
              - score: Combined relevance score (0.0 to 1.0)
              - retrieval_method: Which method(s) found this result
        """
        logger.info(
            f"Retrieving for query: '{query[:80]}...' "
            f"(k={k}, strategy={strategy}, source_filter={source_filter})"
        )

        if strategy == "vector":
            return await self._vector_search(query, k, source_filter)
        elif strategy == "keyword":
            return await self._keyword_search(query, k, source_filter)
        else:
            return await self._hybrid_search(query, k, source_filter)

    async def _vector_search(
        self,
        query: str,
        k: int,
        source_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Pure vector similarity search.

        Steps:
          1. Embed the query text using OpenAI
          2. Find the k nearest document vectors using cosine similarity
          3. Format and return results

        CONCEPT: Query Embedding
        The query goes through the same embedding model as the documents.
        This ensures they live in the same vector space — a question about
        "sick leave" produces a vector near the document chunks about sick leave.
        """
        # Step 1: Embed the query
        query_embedding = await embedding_service.embed_text(query)

        # Step 2: Search for similar documents
        results = await similarity_search(
            query_embedding=query_embedding,
            k=k,
            source_filter=source_filter,
        )

        # Step 3: Format results with a normalized score
        formatted = []
        for doc in results:
            formatted.append({
                "content": doc["content"],
                "source": doc["source"],
                "section": doc["section"],
                "metadata": doc["metadata"],
                "score": doc["similarity_score"],
                "retrieval_method": "vector",
            })

        logger.info(f"Vector search returned {len(formatted)} results")
        return formatted

    async def _keyword_search(
        self,
        query: str,
        k: int,
        source_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Keyword-based search using PostgreSQL ILIKE pattern matching.

        CONCEPT: ILIKE Search
        PostgreSQL's ILIKE operator performs case-insensitive pattern matching.
        We wrap the query in % wildcards so "sick leave" matches:
          - "All employees receive 10 paid sick leave days..."
          - "Sick leave of 3+ consecutive days..."
          - "...regarding their SICK LEAVE allocation..."

        For multi-word queries, we search for EACH word independently and
        rank results by how many words they match. This handles queries like
        "overtime pay rate" where all three words might not appear together
        but a relevant chunk contains all of them.

        CONCEPT: Full-Text Search (FTS) Alternative
        PostgreSQL also offers full-text search (tsvector/tsquery), which is
        more sophisticated than ILIKE:
          - Handles stemming: "running" matches "run"
          - Handles stop words: ignores "the", "is", "a"
          - Supports ranking by term frequency

        For simplicity, we use ILIKE here. In a production system, you might
        upgrade to FTS with ts_rank for better keyword search quality.
        """
        # CONCEPT: Query Term Extraction
        # Split the query into individual words (ignoring short words and stop words)
        # This allows us to match documents that contain most query terms,
        # even if they don't appear as an exact phrase.
        query_terms = [
            term.strip().lower()
            for term in re.split(r'\s+', query)
            if len(term.strip()) > 2  # Skip very short words ("a", "is", "of")
        ]

        if not query_terms:
            return []

        # Build a query that searches for each term and counts matches
        # CONCEPT: Dynamic SQL Query Building
        # We construct WHERE conditions for each search term. A document that
        # matches MORE terms will have a higher relevance score.
        like_conditions = []
        params: dict[str, Any] = {"k": k}

        for i, term in enumerate(query_terms):
            param_name = f"term_{i}"
            like_conditions.append(f"(CASE WHEN content ILIKE :{param_name} THEN 1 ELSE 0 END)")
            params[param_name] = f"%{term}%"

        # The relevance score is the fraction of query terms that appear in the chunk
        relevance_expr = " + ".join(like_conditions)
        total_terms = len(query_terms)

        # At least one term must match
        any_match = " OR ".join(
            [f"content ILIKE :term_{i}" for i in range(len(query_terms))]
        )

        # Add optional source filter
        source_clause = ""
        if source_filter:
            source_clause = "AND source = :source_filter"
            params["source_filter"] = source_filter

        search_query = text(f"""
            SELECT
                id,
                content,
                source,
                section,
                metadata,
                ({relevance_expr})::float / {total_terms} AS relevance_score
            FROM documents
            WHERE ({any_match}) {source_clause}
            ORDER BY relevance_score DESC, created_at DESC
            LIMIT :k
        """)

        async with async_session_maker() as session:
            result = await session.execute(search_query, params)
            rows = result.fetchall()

        formatted = []
        for row in rows:
            formatted.append({
                "content": row.content,
                "source": row.source,
                "section": row.section,
                "metadata": row.metadata,
                "score": round(float(row.relevance_score), 4),
                "retrieval_method": "keyword",
            })

        logger.info(f"Keyword search returned {len(formatted)} results")
        return formatted

    async def _hybrid_search(
        self,
        query: str,
        k: int,
        source_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search combining vector and keyword results using RRF.

        CONCEPT: Fusion Process
        1. Run both vector and keyword searches (with expanded k for broader coverage)
        2. Assign RRF scores to results from each method
        3. Merge and sum scores for documents that appear in both result sets
        4. Sort by combined score and return top k

        The expanded k (we fetch 2x results from each method) ensures we have
        enough candidates for fusion. Some documents might rank low in one
        method but high in the other — we don't want to miss them.
        """
        # Fetch more results than needed from each method for better fusion
        expanded_k = min(k * 3, 20)  # Fetch 3x results, capped at 20

        # CONCEPT: Concurrent Search
        # We could run both searches concurrently with asyncio.gather() for
        # better latency. However, running them sequentially is simpler and
        # the keyword search is fast (no API call), so the improvement is small.
        vector_results = await self._vector_search(query, expanded_k, source_filter)
        keyword_results = await self._keyword_search(query, expanded_k, source_filter)

        # CONCEPT: Reciprocal Rank Fusion Implementation
        # We use a dict keyed by content (or a hash of content) to merge results.
        # Each result accumulates weighted RRF scores from each method.
        fused_scores: dict[str, dict[str, Any]] = {}

        # Process vector search results
        for rank, result in enumerate(vector_results):
            # RRF score: weighted / (rank + k)
            rrf_score = self._vector_weight / (rank + RRF_K)
            content_key = result["content"][:200]  # Use first 200 chars as key

            if content_key not in fused_scores:
                fused_scores[content_key] = {
                    "content": result["content"],
                    "source": result["source"],
                    "section": result["section"],
                    "metadata": result["metadata"],
                    "score": 0.0,
                    "retrieval_methods": set(),
                    "vector_score": result["score"],
                }

            fused_scores[content_key]["score"] += rrf_score
            fused_scores[content_key]["retrieval_methods"].add("vector")

        # Process keyword search results
        for rank, result in enumerate(keyword_results):
            rrf_score = self._keyword_weight / (rank + RRF_K)
            content_key = result["content"][:200]

            if content_key not in fused_scores:
                fused_scores[content_key] = {
                    "content": result["content"],
                    "source": result["source"],
                    "section": result["section"],
                    "metadata": result["metadata"],
                    "score": 0.0,
                    "retrieval_methods": set(),
                    "keyword_score": result["score"],
                }

            fused_scores[content_key]["score"] += rrf_score
            fused_scores[content_key]["retrieval_methods"].add("keyword")

        # Sort by fused score (descending) and take top k
        sorted_results = sorted(
            fused_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:k]

        # Format the final results
        formatted = []
        for result in sorted_results:
            methods = result.pop("retrieval_methods")
            # Normalize the score to a 0-1 range for consistency
            # The max possible RRF score is (vector_weight + keyword_weight) / RRF_K
            max_rrf = (self._vector_weight + self._keyword_weight) / RRF_K
            normalized_score = min(result["score"] / max_rrf, 1.0) if max_rrf > 0 else 0.0

            formatted.append({
                "content": result["content"],
                "source": result["source"],
                "section": result["section"],
                "metadata": result["metadata"],
                "score": round(normalized_score, 4),
                "retrieval_method": "+".join(sorted(methods)),
            })

        logger.info(
            f"Hybrid search returned {len(formatted)} results "
            f"(from {len(vector_results)} vector + {len(keyword_results)} keyword)"
        )

        return formatted


# =============================================================================
# Module-level convenience instance
# =============================================================================
# CONCEPT: Default Retriever Instance
# Like the embedding service, we provide a default retriever instance for
# easy access. Agents and API endpoints can simply import and use it:
#   from src.rag.retriever import retriever
#   results = await retriever.retrieve("What is the overtime rate?")
# =============================================================================
retriever = HybridRetriever()
