"""
Redis Caching Layer
=============================================================================
CONCEPT: Why Redis for an Agentic AI System?

Redis is an in-memory data store that provides sub-millisecond read/write
operations. In our HR Payroll Agent, Redis serves three critical purposes:

  1. EMBEDDING CACHE — Avoid redundant LLM API calls
  2. RESPONSE CACHE — Cache frequently-asked questions
  3. RATE LIMITING — Prevent abuse and control costs

EMBEDDING CACHE — Why Cache Embeddings?
  Every RAG (Retrieval-Augmented Generation) query requires converting the
  user's question into a vector embedding. This involves an API call to
  OpenAI's embedding endpoint:

    Without cache:
      User: "What is the leave policy?"   -> API call ($0.0001, ~200ms)
      User: "What is the leave policy?"   -> API call ($0.0001, ~200ms)  (duplicate!)
      User: "What is the leave policy?"   -> API call ($0.0001, ~200ms)  (duplicate!)

    With cache:
      User: "What is the leave policy?"   -> API call ($0.0001, ~200ms) + cache
      User: "What is the leave policy?"   -> Cache hit ($0, ~1ms)       (100x faster!)
      User: "What is the leave policy?"   -> Cache hit ($0, ~1ms)       (100x faster!)

  Benefits:
    - Cost reduction: Embedding API calls cost money; cached results are free
    - Latency reduction: ~200ms API call vs ~1ms Redis read
    - Rate limit protection: Fewer API calls = less risk of hitting rate limits

  We cache by hashing the input text (SHA256). Same text = same hash = cache hit.
  TTL (Time To Live) of 1 hour means embeddings are refreshed periodically
  (in case the embedding model is updated).

RATE LIMITING — How Does Redis-Based Rate Limiting Work?
  Rate limiting prevents any single user from overwhelming the system.
  This is especially important for AI agents because:
    - Each agent execution costs money (LLM API calls)
    - Unbounded usage could exhaust API quotas
    - A compromised account could drain the budget

  ALGORITHM: Sliding Window Counter (simplified)
    We use Redis keys with TTL (expiry) to count requests per user:

    Key pattern: "rate_limit:{user_id}"
    Value: number of requests in the current window

    1. User makes a request
    2. We INCR the key (atomic increment in Redis)
    3. If this is the first request (key didn't exist), SET expiry to window seconds
    4. If the count exceeds the limit, REJECT the request (return False)

    Why Redis for rate limiting?
      - INCR is atomic (no race conditions even with concurrent requests)
      - TTL handles automatic window reset (no cleanup needed)
      - Shared across all server instances (works in multi-process deployments)
      - Sub-millisecond operations (negligible overhead per request)

  Example timeline (limit=3, window=60s):
    t=0s:  User A makes request -> count=1 (allowed)
    t=10s: User A makes request -> count=2 (allowed)
    t=20s: User A makes request -> count=3 (allowed)
    t=30s: User A makes request -> count=4 (DENIED — over limit)
    t=60s: Key expires           -> count resets to 0
    t=61s: User A makes request -> count=1 (allowed again)

REDIS DATA STRUCTURES USED:
  - Strings: For cached values (embeddings, responses) and rate limit counters
  - TTL: For automatic expiration of cache entries and rate limit windows
  - INCR: For atomic counter increments (rate limiting)
=============================================================================
"""

import json
import hashlib
from typing import Any

import redis.asyncio as redis

from src.config import settings


class RedisCache:
    """
    Async Redis client for caching, rate limiting, and embedding storage.

    DESIGN DECISIONS:
      1. Lazy Connection: The Redis connection is NOT created in __init__.
         Instead, it's created on first use via connect(). This allows the
         application to start even if Redis is temporarily unavailable.

      2. Async: We use redis.asyncio for non-blocking I/O. While Redis
         operations are fast (~1ms), in a high-concurrency FastAPI app,
         blocking I/O would prevent other requests from being processed.

      3. Connection Pooling: redis.asyncio.from_url() creates a connection
         pool internally. Connections are reused across requests (similar
         to SQLAlchemy's connection pool for PostgreSQL).

      4. JSON Serialization: We store Python objects as JSON strings in Redis.
         Redis only supports strings, bytes, integers, and floats natively.
         For complex objects (lists, dicts, embeddings), we serialize to JSON.

    USAGE:
        # In your FastAPI lifespan or startup:
        cache = RedisCache()
        await cache.connect()

        # In your application code:
        await cache.set("key", {"data": "value"}, ttl=300)
        result = await cache.get("key")  # {"data": "value"}

        # In your shutdown:
        await cache.disconnect()
    """

    def __init__(self) -> None:
        """
        Initialize the RedisCache instance.

        NOTE: This does NOT create a Redis connection. Call connect() to
        establish the connection. This pattern allows:
          - Creating the cache object at module level (before event loop starts)
          - Connecting during FastAPI's lifespan startup
          - Graceful handling if Redis is unavailable at startup
        """
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        """
        Establish a connection to Redis.

        Uses the redis_url from settings (default: redis://localhost:6379/0).

        REDIS URL FORMAT:
          redis://[:password@]host[:port][/database_number]

          Examples:
            redis://localhost:6379/0          <- Local, no auth, database 0
            redis://:mypassword@redis:6379/0  <- With password
            rediss://redis-cloud:6380/0       <- TLS-encrypted (rediss://)

        CONNECTION POOL:
          redis.asyncio.from_url() creates a connection pool automatically.
          The pool maintains a set of open TCP connections to Redis and
          reuses them across requests. This avoids the overhead of creating
          a new TCP connection for every Redis operation (~1ms per new
          connection vs ~0.01ms for a pooled connection).

        DECODE_RESPONSES=True:
          By default, redis-py returns bytes (b"value"). With decode_responses,
          it returns strings ("value") automatically. This simplifies our code
          since we're storing JSON strings.
        """
        self._client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )

    async def disconnect(self) -> None:
        """
        Close the Redis connection and release all pooled connections.

        Call this during application shutdown (in FastAPI's lifespan).
        Failing to disconnect can leave orphaned TCP connections.
        """
        if self._client is not None:
            await self._client.close()
            self._client = None

    @property
    def client(self) -> redis.Redis:
        """
        Get the Redis client, raising an error if not connected.

        This property provides a safe way to access the client. Without
        this guard, you'd get cryptic AttributeError: 'NoneType' has no
        attribute 'get' if you forgot to call connect().
        """
        if self._client is None:
            raise RuntimeError(
                "Redis client is not connected. Call 'await cache.connect()' "
                "during application startup before using the cache."
            )
        return self._client

    # =========================================================================
    # General-Purpose Cache Operations
    # =========================================================================

    async def get(self, key: str) -> Any | None:
        """
        Get a value from the cache by key.

        PARAMETERS:
          key: The cache key (string). Convention: "namespace:identifier"
              Examples: "user:abc-123", "embedding:sha256hash", "response:query_hash"

        RETURNS:
          The deserialized Python object if the key exists, None otherwise.
          Returns None for both missing keys AND expired keys (Redis
          automatically deletes expired keys).

        HOW IT WORKS:
          1. Send GET command to Redis
          2. If the key exists, Redis returns the stored string
          3. We deserialize the JSON string back to a Python object
          4. If the key doesn't exist (or expired), Redis returns None
        """
        raw = await self.client.get(key)
        if raw is None:
            return None

        # Deserialize from JSON string back to Python object
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # If the stored value isn't valid JSON, return the raw string.
            # This handles cases where plain strings were stored directly.
            return raw

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Store a value in the cache with an optional TTL (time-to-live).

        PARAMETERS:
          key: The cache key.
          value: Any JSON-serializable Python object (dict, list, str, int, etc.).
              The value is serialized to a JSON string for storage.
          ttl: Time-to-live in seconds. After this many seconds, Redis
              automatically deletes the key. If None, the key persists
              indefinitely (until explicitly deleted or Redis restarts).

        WHY TTL?
          TTL prevents stale data from living forever in the cache:
            - Embedding cache: TTL=3600 (1 hour) — refresh if model updates
            - Response cache: TTL=300 (5 minutes) — balance freshness vs performance
            - Session data: TTL=1800 (30 minutes) — auto-expire idle sessions

          Without TTL, the cache would grow indefinitely and eventually
          consume all available memory. Redis has an eviction policy for
          when memory is full, but explicit TTL is more predictable.

        HOW TTL WORKS IN REDIS:
          Redis stores an expiration timestamp with the key. A background
          process periodically checks for expired keys and deletes them.
          Additionally, when you access an expired key, Redis deletes it
          on the spot (lazy deletion). This dual approach ensures expired
          keys are cleaned up without excessive CPU usage.
        """
        serialized = json.dumps(value)

        if ttl is not None:
            # SET key value EX ttl — sets the value AND expiration atomically
            await self.client.set(key, serialized, ex=ttl)
        else:
            # SET key value — no expiration
            await self.client.set(key, serialized)

    async def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        PARAMETERS:
          key: The cache key to delete.

        RETURNS:
          True if the key existed and was deleted, False if the key didn't exist.

        USE CASES:
          - Invalidate cached data after an update (e.g., employee salary changed)
          - Clear a user's rate limit counter (admin override)
          - Remove stale entries during maintenance
        """
        result = await self.client.delete(key)
        # Redis DEL returns the number of keys that were deleted (0 or 1)
        return result > 0

    # =========================================================================
    # Embedding Cache
    # =========================================================================

    async def cache_embedding(
        self,
        text_hash: str,
        embedding: list[float],
        ttl: int = 3600,
    ) -> None:
        """
        Cache a vector embedding for a given text hash.

        CONCEPT: Why Cache Embeddings?
        Embedding API calls are the most frequent external API calls in a
        RAG system. Every user query needs an embedding for similarity search.
        Caching avoids redundant calls for repeated or similar queries.

        KEY DESIGN:
          We use "embedding:{text_hash}" as the key. The text_hash is a
          SHA-256 hash of the input text. This ensures:
            - Same text always maps to the same key (deterministic)
            - Different texts get different keys (collision-resistant)
            - The key is fixed-length (64 hex chars) regardless of input length

        PARAMETERS:
          text_hash: SHA-256 hash of the input text. Generate with:
              hashlib.sha256(text.encode()).hexdigest()
          embedding: The vector embedding as a list of floats (e.g., 1536 floats
              for OpenAI's text-embedding-3-small model).
          ttl: Cache duration in seconds. Default 3600 (1 hour).
              Why 1 hour? Embedding models rarely change, but if they do,
              stale embeddings would return incorrect similarity results.
              1 hour balances cache hit rate vs freshness.

        STORAGE SIZE:
          A 1536-dimension embedding serialized as JSON is about 15-20KB.
          For 10,000 cached embeddings: ~200MB — easily fits in Redis.

        USAGE:
            import hashlib
            text = "What is the leave policy?"
            text_hash = hashlib.sha256(text.encode()).hexdigest()

            # Check cache first
            cached = await cache.get_cached_embedding(text_hash)
            if cached is not None:
                embedding = cached  # Cache hit!
            else:
                embedding = await openai_client.embed(text)
                await cache.cache_embedding(text_hash, embedding)
        """
        key = f"embedding:{text_hash}"
        await self.set(key, embedding, ttl=ttl)

    async def get_cached_embedding(self, text_hash: str) -> list[float] | None:
        """
        Retrieve a cached embedding by text hash.

        PARAMETERS:
          text_hash: SHA-256 hash of the input text.

        RETURNS:
          The cached embedding as a list of floats, or None if not cached.

        USAGE:
            text_hash = hashlib.sha256("What is the leave policy?".encode()).hexdigest()
            embedding = await cache.get_cached_embedding(text_hash)
            if embedding is None:
                # Cache miss — compute the embedding
                ...
        """
        key = f"embedding:{text_hash}"
        return await self.get(key)

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    async def rate_limit(
        self,
        user_id: str,
        limit: int = 100,
        window: int = 3600,
    ) -> bool:
        """
        Check if a user is within their rate limit.

        ALGORITHM: Fixed Window Counter
          1. Create a key "rate_limit:{user_id}" (or increment if it exists)
          2. If the key is new, set its TTL to `window` seconds
          3. If the counter exceeds `limit`, deny the request

        PARAMETERS:
          user_id: Unique identifier for the user (typically their UUID).
          limit: Maximum number of requests allowed in the window.
              Default: 100 requests per window.
              You might set different limits per role:
                - admin: 500 (needs to run batch operations)
                - manager: 200 (moderate usage)
                - employee: 100 (chatbot queries)
          window: Time window in seconds. Default: 3600 (1 hour).

        RETURNS:
          True if the request is WITHIN the limit (allowed).
          False if the request EXCEEDS the limit (should be denied).

        REDIS OPERATIONS:
          INCR: Atomically increment the counter. If the key doesn't exist,
                Redis creates it with value 1. Atomicity is crucial — even
                with 100 concurrent requests, INCR guarantees accurate counting.

          EXPIRE: Set the key's TTL. We only set this when count==1 (first
                  request in the window), so the window doesn't reset on
                  subsequent requests.

        EDGE CASE: Race Condition
          There's a tiny window between INCR and EXPIRE where the key exists
          without a TTL (if the server crashes between the two operations).
          In production, you'd use a Lua script to make INCR + EXPIRE atomic:

            local current = redis.call('INCR', KEYS[1])
            if current == 1 then
                redis.call('EXPIRE', KEYS[1], ARGV[1])
            end
            return current

          For our use case, this race condition is acceptable — the key would
          eventually be overwritten by a new window.

        USAGE:
            if not await cache.rate_limit(user_id=str(user.id), limit=100, window=3600):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Try again later.",
                    headers={"Retry-After": "3600"},
                )

        MONITORING:
          You can check a user's current count with:
            GET rate_limit:{user_id}  -> "42" (42 requests so far in this window)
          And the remaining TTL with:
            TTL rate_limit:{user_id}  -> 1823 (1823 seconds until window resets)
        """
        key = f"rate_limit:{user_id}"

        # Atomically increment the counter
        # If the key doesn't exist, INCR creates it with value 0 then
        # increments to 1. This is a single Redis roundtrip.
        current_count = await self.client.incr(key)

        # If this is the first request in the window, set the expiration.
        # We check == 1 (not <= 1) because INCR returns the NEW value.
        # The TTL ensures the counter automatically resets after `window` seconds.
        if current_count == 1:
            await self.client.expire(key, window)

        # Check if the user has exceeded their limit
        return current_count <= limit


# =============================================================================
# Module-Level Singleton
# =============================================================================
# We create a single RedisCache instance that can be imported anywhere.
# This follows the same pattern as `settings = Settings()` in config.py.
#
# IMPORTANT: The cache is NOT connected at import time. You must call
# `await redis_cache.connect()` during application startup.
#
# Usage:
#   from src.cache.redis_client import redis_cache
#   await redis_cache.connect()  # In startup
#   await redis_cache.set("key", "value", ttl=300)  # In application code
#   await redis_cache.disconnect()  # In shutdown
# =============================================================================
redis_cache = RedisCache()


def get_text_hash(text: str) -> str:
    """
    Compute a SHA-256 hash of the input text for use as a cache key.

    This is a convenience function for generating embedding cache keys.
    SHA-256 produces a 64-character hex string that is:
      - Deterministic: Same input always produces the same hash
      - Collision-resistant: Different inputs produce different hashes
      - Fixed-length: Always 64 characters, regardless of input length

    PARAMETERS:
      text: The input text to hash.

    RETURNS:
      A 64-character hexadecimal hash string.

    USAGE:
        from src.cache.redis_client import redis_cache, get_text_hash

        text = "What is the leave policy for new employees?"
        text_hash = get_text_hash(text)
        # text_hash = "a3f2b8c1d4e5..." (64 hex chars)

        cached = await redis_cache.get_cached_embedding(text_hash)
        if cached is None:
            embedding = await compute_embedding(text)
            await redis_cache.cache_embedding(text_hash, embedding)
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
