"""
Embedding Generation Service
=============================================================================
CONCEPT: What Are Embeddings?

Embeddings are numerical representations of text that capture semantic meaning.
Instead of treating words as arbitrary symbols ("cat" has no relation to "kitten"),
embeddings map text into a high-dimensional vector space where:

  - Similar meanings are CLOSE together (small distance / high similarity)
  - Different meanings are FAR apart (large distance / low similarity)

Think of it like plotting words on a map:
  "salary" and "compensation" would be neighbors
  "salary" and "banana" would be far apart

HOW DO EMBEDDINGS CAPTURE MEANING?
  Neural networks (like OpenAI's models) learn to associate words and phrases
  that appear in similar contexts. During training on billions of text examples,
  the model learns patterns like:
    - "increased salary" and "raised compensation" appear in similar contexts
    - "PTO policy" and "vacation rules" are used interchangeably
  The result: texts with similar meaning get similar vector representations.

WHY 1536 DIMENSIONS?
  OpenAI's text-embedding-3-small produces 1536-dimensional vectors. Each
  dimension captures a different "feature" or "aspect" of the text's meaning.
  Think of each dimension as answering a fuzzy question about the text:
    - Dimension 42 might loosely correspond to "financial-ness"
    - Dimension 789 might loosely correspond to "formality"
    - Dimension 1200 might loosely correspond to "time-related-ness"

  In reality, individual dimensions don't have clean human-readable meanings,
  but collectively they create a rich representation. 1536 dimensions gives
  enough capacity to distinguish nuanced meanings while keeping costs and
  storage reasonable.

  Alternatives:
    - text-embedding-3-large: 3072 dimensions (more precise, but 2x storage)
    - text-embedding-3-small: 1536 dimensions (good balance of cost and quality)
    - text-embedding-ada-002: 1536 dimensions (older model, same dimensionality)

USAGE IN RAG:
  1. At ingestion time: we embed each document chunk and store the vector
  2. At query time: we embed the user's question
  3. We find document chunks whose vectors are closest to the question vector
  4. Those chunks become context for the LLM to generate an answer

COST CONSIDERATIONS:
  - text-embedding-3-small costs ~$0.02 per 1M tokens (very cheap!)
  - Batch processing reduces API calls and latency
  - Embeddings are deterministic: same input always produces same output
    so we can cache them
=============================================================================
"""

import logging
from typing import Optional

from openai import AsyncOpenAI

from src.config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# CONCEPT: OpenAI Client Initialization
#
# We use the async version of the OpenAI client (AsyncOpenAI) because our
# entire application is async (FastAPI + asyncpg). Using the sync client
# would block the event loop, preventing other requests from being handled
# while we wait for OpenAI's response.
#
# The client is created at module level (singleton pattern) so we reuse the
# same HTTP connection pool across all embedding requests.
# =============================================================================

# The model we use for generating embeddings
# text-embedding-3-small is OpenAI's recommended model for most use cases:
#   - Good quality semantic understanding
#   - Affordable pricing ($0.02 / 1M tokens)
#   - 1536 dimensions (matches our pgvector column)
EMBEDDING_MODEL = "text-embedding-3-small"

# The number of dimensions produced by the model
# This MUST match the Vector(1536) column definition in our documents table
EMBEDDING_DIMENSIONS = 1536

# Maximum number of texts that can be embedded in a single API call
# OpenAI's API supports batches, which is more efficient than individual calls:
#   - 1 API call for 100 texts vs 100 API calls for 100 texts
#   - Network overhead is the bottleneck, not computation
MAX_BATCH_SIZE = 100


class EmbeddingService:
    """
    Wrapper around OpenAI's embedding API for generating text embeddings.

    CONCEPT: Service Layer Pattern
    This class encapsulates all embedding logic in one place. The rest of the
    application doesn't need to know about OpenAI's API details — it just calls
    embed_text() or embed_batch() and gets back a list of floats.

    This makes it easy to swap the embedding provider later (e.g., switch to
    Cohere, use a local model, etc.) without changing any other code.

    Usage:
        service = EmbeddingService()
        vector = await service.embed_text("What is the leave policy?")
        # vector is a list of 1536 floats, e.g., [0.0123, -0.0456, ...]
    """

    def __init__(self, api_key: Optional[str] = None, model: str = EMBEDDING_MODEL):
        """
        Initialize the embedding service.

        Args:
            api_key: OpenAI API key. If not provided, reads from settings.
                     This parameter exists for testing — you can inject a
                     test key without modifying settings.
            model:   The embedding model to use. Defaults to text-embedding-3-small.
        """
        # CONCEPT: Configuration Injection
        # We allow passing the API key explicitly (for tests) but default to
        # the global settings. This follows the "dependency injection" principle:
        # components receive their dependencies from outside, making them testable.
        self._client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        self._model = model

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding vector for a single text string.

        CONCEPT: Single Embedding
        This is the simplest use case — embed one piece of text. Used when:
          - A user submits a query (embed the question to search for similar docs)
          - Ingesting a single document or chunk

        The returned list has exactly 1536 float values, where each value
        is typically between -1.0 and 1.0 (the exact range depends on the model).

        Args:
            text: The text to embed. Can be a short question or a long paragraph.
                  OpenAI's model handles up to 8191 tokens (~6000 words).

        Returns:
            A list of 1536 floats representing the semantic meaning of the text.

        Raises:
            openai.APIError: If the OpenAI API call fails (network, auth, etc.)
        """
        # CONCEPT: Input Preprocessing
        # Replace newlines with spaces to normalize the text. Embeddings are
        # somewhat sensitive to formatting — "hello\n\nworld" and "hello world"
        # should produce similar (but not identical) vectors. Normalizing helps
        # reduce this variance.
        cleaned_text = text.replace("\n", " ").strip()

        if not cleaned_text:
            # Return a zero vector for empty text rather than making an API call
            # This is a defensive measure — empty embeddings would match nothing
            logger.warning("Attempted to embed empty text, returning zero vector")
            return [0.0] * EMBEDDING_DIMENSIONS

        logger.debug(f"Embedding text ({len(cleaned_text)} chars): {cleaned_text[:80]}...")

        # Call OpenAI's embedding API
        # The response contains a list of embedding objects, one per input text
        response = await self._client.embeddings.create(
            input=[cleaned_text],
            model=self._model,
        )

        # Extract the embedding vector from the response
        # response.data is a list of Embedding objects; we sent 1 text so we get 1 back
        embedding = response.data[0].embedding

        logger.debug(
            f"Generated embedding: {len(embedding)} dimensions, "
            f"usage: {response.usage.total_tokens} tokens"
        )

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in a single API call.

        CONCEPT: Batch Embedding
        When ingesting many document chunks, embedding them one-by-one would
        require N separate API calls. Each call has network overhead (~100-200ms
        round trip). For 100 chunks:
          - One-by-one: 100 calls x 200ms = ~20 seconds of network wait
          - Batch: 1 call x 200ms = ~0.2 seconds of network wait (100x faster!)

        OpenAI's API accepts up to 2048 texts per batch, but we cap at 100 to
        avoid memory issues and timeouts. For larger batches, this method
        automatically splits into sub-batches.

        Args:
            texts: List of text strings to embed. Each will get its own vector.
                   The order of returned embeddings matches the input order.

        Returns:
            A list of embedding vectors (list of lists of floats).
            result[i] corresponds to texts[i].

        Example:
            texts = ["leave policy", "salary structure", "tax brackets"]
            embeddings = await service.embed_batch(texts)
            # embeddings[0] = vector for "leave policy"
            # embeddings[1] = vector for "salary structure"
            # embeddings[2] = vector for "tax brackets"
        """
        if not texts:
            return []

        # CONCEPT: Sub-batching
        # If we have more texts than the API supports in one call, we split
        # them into smaller batches and process them sequentially. We could
        # also process sub-batches concurrently with asyncio.gather(), but
        # sequential processing is safer (avoids rate limits).
        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), MAX_BATCH_SIZE):
            batch_end = min(batch_start + MAX_BATCH_SIZE, len(texts))
            batch = texts[batch_start:batch_end]

            # Clean each text in the batch
            cleaned_batch = [t.replace("\n", " ").strip() for t in batch]

            # Replace any empty strings with a placeholder to avoid API errors
            # We'll replace the resulting embeddings with zero vectors afterward
            empty_indices = set()
            for i, t in enumerate(cleaned_batch):
                if not t:
                    empty_indices.add(i)
                    cleaned_batch[i] = "empty"  # Placeholder (will be zeroed)

            logger.info(
                f"Embedding batch {batch_start // MAX_BATCH_SIZE + 1}: "
                f"{len(cleaned_batch)} texts "
                f"({sum(len(t) for t in cleaned_batch)} total chars)"
            )

            response = await self._client.embeddings.create(
                input=cleaned_batch,
                model=self._model,
            )

            # CONCEPT: Response Ordering
            # OpenAI returns embeddings in the same order as the input texts.
            # Each response.data[i].embedding corresponds to cleaned_batch[i].
            # We sort by index to be safe (the API guarantees order, but
            # defensive programming is good practice).
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_data]

            # Replace embeddings for empty texts with zero vectors
            for idx in empty_indices:
                batch_embeddings[idx] = [0.0] * EMBEDDING_DIMENSIONS

            all_embeddings.extend(batch_embeddings)

            logger.info(
                f"Batch embedded successfully. "
                f"Tokens used: {response.usage.total_tokens}"
            )

        return all_embeddings


# =============================================================================
# Module-level convenience instance
# =============================================================================
# CONCEPT: Singleton-like Access
# We expose a default instance so callers can do:
#   from src.rag.embeddings import embedding_service
#   vector = await embedding_service.embed_text("...")
#
# This avoids creating multiple OpenAI clients (each with its own HTTP pool).
# For testing, callers can create their own EmbeddingService with a mock key.
# =============================================================================
embedding_service = EmbeddingService()
