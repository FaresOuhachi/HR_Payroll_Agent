"""
Documents API — Ingest & Search Endpoints
=============================================================================
CONCEPT: RAG API Layer

These endpoints expose the RAG system's capabilities over HTTP, enabling:

  1. INGESTION (POST /documents/ingest)
     - Accept text content + source name
     - Run it through the ingestion pipeline (split → embed → store)
     - Return ingestion statistics

  2. SEARCH (POST /documents/search)
     - Accept a natural language query
     - Run hybrid retrieval (vector + keyword search)
     - Return ranked results with similarity scores

WHY POST instead of GET for search?
  While GET is typical for read operations, we use POST for search because:
    1. The query can be long (URLs have length limits, typically 2048 chars)
    2. We need structured parameters (strategy, filters, k) that fit better
       in a JSON body than URL query params
    3. Search operations are conceptually "creating a search" rather than
       "getting a resource" — some APIs even create persistent search objects

CONCEPT: Pydantic Request/Response Models
  Like the employee endpoints, we define explicit schemas for:
    - Request validation: Ensure the client sends valid data
    - Response serialization: Control exactly what the client receives
    - API documentation: FastAPI generates OpenAPI docs from these models
=============================================================================
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.rag.ingestion import ingest_text
from src.rag.retriever import retriever
from src.rag.vectorstore import count_documents

logger = logging.getLogger(__name__)


# =============================================================================
# API Router
# =============================================================================
# CONCEPT: APIRouter with Tags
# The prefix "/documents" means all routes in this file start with /documents.
# The tag "Documents" groups these endpoints in the Swagger UI for easy browsing.
router = APIRouter(prefix="/documents", tags=["Documents"])


# =============================================================================
# Request & Response Schemas
# =============================================================================

class IngestRequest(BaseModel):
    """
    Request body for document ingestion.

    CONCEPT: Explicit Schema Validation
    Pydantic validates every field before the route handler runs:
      - content must be a non-empty string (min_length=1)
      - source must be provided (required field)
      - section is optional with a sensible default
    If validation fails, FastAPI returns a 422 Unprocessable Entity response
    with details about which fields are invalid and why.
    """
    content: str = Field(
        ...,
        min_length=1,
        description="The text content to ingest into the knowledge base. "
                    "Can be raw text, Markdown, or structured content.",
        examples=["All employees receive 22 PTO days per year."],
    )
    source: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Source identifier for this content. Used for filtering "
                    "and citation. Example: 'leave_policy.md', 'hr_handbook'",
        examples=["leave_policy.md"],
    )
    section: str = Field(
        default="",
        max_length=255,
        description="Optional section label. If not provided, section headers "
                    "will be extracted automatically from the content.",
        examples=["Section 2: Sick Leave"],
    )


class IngestResponse(BaseModel):
    """
    Response after successful document ingestion.

    CONCEPT: Ingestion Feedback
    We return detailed statistics so the client knows exactly what happened:
      - How many chunks were created (useful for debugging chunk size)
      - Total character count (helps estimate storage usage)
      - Source name (confirmation of what was stored)
    """
    message: str = Field(description="Human-readable success message")
    source: str = Field(description="The source identifier used")
    total_chunks: int = Field(description="Number of chunks created")
    total_characters: int = Field(description="Total characters across all chunks")


class SearchRequest(BaseModel):
    """
    Request body for document search.

    CONCEPT: Search Parameters
    The search endpoint accepts several parameters to control retrieval:
      - query: The natural language question (required)
      - k: Number of results (controls recall vs precision trade-off)
      - source_filter: Narrow search to a specific document
      - strategy: Choose retrieval method (hybrid, vector, keyword)
    """
    query: str = Field(
        ...,
        min_length=1,
        description="Natural language search query. Examples: "
                    "'What is the sick leave policy?', "
                    "'How are tax brackets calculated?'",
        examples=["What is the sick leave policy?"],
    )
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return. Higher k increases recall "
                    "(finding more relevant docs) but may include less relevant ones.",
    )
    source_filter: Optional[str] = Field(
        default=None,
        description="Optional: Only search within documents from this source. "
                    "Example: 'leave_policy.md' to search only the leave policy.",
    )
    strategy: str = Field(
        default="hybrid",
        description="Search strategy: 'hybrid' (vector + keyword), "
                    "'vector' (semantic only), 'keyword' (exact match only)",
    )


class SearchResult(BaseModel):
    """A single search result with content and metadata."""
    content: str = Field(description="The text chunk that matched the query")
    source: str = Field(description="Which document this chunk came from")
    section: str = Field(description="The section heading within the document")
    score: float = Field(description="Relevance score (0.0 to 1.0, higher = better)")
    retrieval_method: str = Field(
        description="Which search method found this result: "
                    "'vector', 'keyword', or 'keyword+vector' (both)"
    )


class SearchResponse(BaseModel):
    """
    Response containing search results.

    CONCEPT: Structured Search Response
    We wrap results in a container that includes:
      - The original query (for debugging/logging)
      - Result count (might be less than requested k if few documents match)
      - The results themselves, each with content, source, and score
    """
    query: str = Field(description="The original search query")
    results: list[SearchResult] = Field(description="Ranked list of matching documents")
    total_results: int = Field(description="Number of results returned")
    strategy: str = Field(description="The search strategy that was used")


class DocumentStatsResponse(BaseModel):
    """Response for document store statistics."""
    total_documents: int = Field(description="Total document chunks in the store")


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest_document(request: IngestRequest):
    """
    Ingest text content into the RAG knowledge base.

    CONCEPT: Ingestion Endpoint
    This endpoint accepts raw text content and processes it through the
    full RAG ingestion pipeline:
      1. Split the text into overlapping chunks
      2. Generate embeddings for each chunk using OpenAI
      3. Store chunks + embeddings in PostgreSQL with pgvector

    The content is associated with a source name for later filtering.
    If content from the same source already exists, it will be replaced
    (re-ingestion is safe and idempotent).

    Example request:
    ```json
    {
        "content": "All employees receive 22 PTO days per year...",
        "source": "leave_policy_v2",
        "section": "PTO Allocation"
    }
    ```
    """
    logger.info(f"Ingestion request: source={request.source}, content_length={len(request.content)}")

    try:
        result = await ingest_text(
            content=request.content,
            source=request.source,
            section=request.section,
        )

        return IngestResponse(
            message=f"Successfully ingested {result['total_chunks']} chunks from '{request.source}'",
            source=result["source"],
            total_chunks=result["total_chunks"],
            total_characters=result["total_characters"],
        )

    except Exception as e:
        logger.error(f"Ingestion failed for source '{request.source}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}. "
                   "Check that OPENAI_API_KEY is set and the database is accessible.",
        )


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search the knowledge base for documents relevant to a query.

    CONCEPT: Retrieval Endpoint
    This endpoint takes a natural language query and returns the most
    relevant document chunks from the knowledge base. It supports three
    search strategies:

      - **hybrid** (default): Combines vector similarity search with keyword
        matching for the best overall results. Recommended for most queries.

      - **vector**: Pure semantic search. Best for natural language questions
        where exact terms may not appear in the documents.

      - **keyword**: Pure keyword matching. Best for searching specific terms,
        codes, or section numbers.

    Example request:
    ```json
    {
        "query": "What is the sick leave policy?",
        "k": 5,
        "strategy": "hybrid"
    }
    ```

    Example response:
    ```json
    {
        "query": "What is the sick leave policy?",
        "results": [
            {
                "content": "All employees receive 10 paid sick days per year...",
                "source": "leave_policy.md",
                "section": "2.1 Sick Leave Allocation",
                "score": 0.92,
                "retrieval_method": "keyword+vector"
            }
        ],
        "total_results": 5,
        "strategy": "hybrid"
    }
    ```
    """
    # Validate strategy
    valid_strategies = {"hybrid", "vector", "keyword"}
    if request.strategy not in valid_strategies:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid strategy '{request.strategy}'. "
                   f"Must be one of: {', '.join(sorted(valid_strategies))}",
        )

    logger.info(
        f"Search request: query='{request.query[:80]}', "
        f"k={request.k}, strategy={request.strategy}"
    )

    try:
        results = await retriever.retrieve(
            query=request.query,
            k=request.k,
            source_filter=request.source_filter,
            strategy=request.strategy,
        )

        search_results = [
            SearchResult(
                content=r["content"],
                source=r["source"],
                section=r["section"],
                score=r["score"],
                retrieval_method=r["retrieval_method"],
            )
            for r in results
        ]

        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            strategy=request.strategy,
        )

    except Exception as e:
        logger.error(f"Search failed for query '{request.query[:80]}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}. "
                   "Check that OPENAI_API_KEY is set and the knowledge base has been populated.",
        )


@router.get("/stats", response_model=DocumentStatsResponse)
async def get_document_stats():
    """
    Get statistics about the document knowledge base.

    CONCEPT: Health Monitoring
    This endpoint provides a quick way to check the state of the knowledge base:
      - How many document chunks are stored?
      - Is the store empty (has ingestion been run)?

    Useful for:
      - Monitoring dashboards
      - Pre-flight checks before running agents
      - Debugging "no results found" issues (store might be empty)
    """
    try:
        total = await count_documents()
        return DocumentStatsResponse(total_documents=total)
    except Exception as e:
        logger.error(f"Failed to get document stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document stats: {str(e)}",
        )
